import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from models.utils import ACTLayer, Linear, MLPBase, Normal, RNNLayer, build_mlp, calculate_log_pi


class Actor(jit.ScriptModule):
    
    def __init__(self, ob_dim, ac_dim, h_dim, gain, use_rnn=True):
        super().__init__()
        self.base = MLPBase(ob_dim, h_dim)
        self.rnn = RNNLayer(h_dim, h_dim, use_rnn)
        self.act = ACTLayer(ac_dim, h_dim, gain)
        self.h_dim = h_dim

    @jit.script_method
    def _forward(self, obs, rnn_actor, masks, avails, deterministic:bool=False):
        features = self.base.forward(obs)
        features, rnn_actor = self.rnn.forward(features, rnn_actor, masks)
        actions, log_probs = self.act.forward(features, avails, deterministic)
        return actions, log_probs, rnn_actor

    @jit.script_method
    def _act_evaluate_actions(self, features, action, avails):
        logits = self.act.dist.forward(features, avails)
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        log_probs = self.act.dist.pd_log_probs(action, logits)
        dist_entropy = self.act.dist.pd_entropy(logits, probs)
        return log_probs, dist_entropy
    
    @jit.script_method
    def _evaluate_actions(self, obs, rnn_actor, actions, masks, avails):
        features = self.base.forward(obs)
        features, rnn_actor = self.rnn.forward(features, rnn_actor, masks)
        log_probs, dist_entropy = self._act_evaluate_actions(features, actions, avails)
        return log_probs, dist_entropy
    
    @jit.script_method
    def compute_pg_loss(self, obs, rnn_actor, actions, masks, avails, old_log_probs, advs, clip_param:float, ent_coef:float):
        log_probs, entropy = self._evaluate_actions(obs, rnn_actor, actions, masks, avails)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advs
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advs
        pg_loss = torch.sum(torch.minimum(surr1, surr2), dim=-1, keepdim=True)
        pg_loss = - pg_loss.mean()
        entropy = entropy.mean()
        loss = pg_loss - entropy * ent_coef
        loss.backward()
        return pg_loss.item(), entropy.item()


class Critic(jit.ScriptModule):
    
    def __init__(self, st_dim, h_dim, use_huber=False, use_norm=True, use_rnn=True):
        super().__init__()
        self.base = MLPBase(st_dim, h_dim)
        self.rnn = RNNLayer(h_dim, h_dim, use_rnn)
        self.v_out = Linear(h_dim, 1, 1)
        self.norm = ValueNorm(1, use_norm=use_norm)
        self.h_dim = h_dim
        self.use_huber = use_huber

    @jit.script_method
    def _forward(self, states, rnn_critic, masks):
        features = self.base.forward(states)
        features, rnn_critic = self.rnn.forward(features, rnn_critic, masks)
        values = self.v_out.forward(features)
        return values, rnn_critic

    @jit.script_method
    def compute_vf_loss(self, states, rnn_critic, masks, value_preds, returns, clip_param:float, vf_coef:float):
        values, _ = self._forward(states, rnn_critic, masks)
        value_pred_clipped = value_preds + (values - value_preds).clamp(-clip_param, clip_param)
        self.norm.update(returns)
        returns = self.norm.normalize(returns)
        if self.use_huber:
            vf_loss_clipped = F.huber_loss(returns, value_pred_clipped, reduction='none')
            vf_loss_original = F.huber_loss(returns, values, reduction='none')
        else:
            vf_loss_clipped = F.mse_loss(returns, value_pred_clipped, reduction='none')
            vf_loss_original = F.mse_loss(returns, values, reduction='none')
        vf_loss = torch.maximum(vf_loss_original, vf_loss_clipped)
        vf_loss = vf_loss.mean()
        loss = vf_loss * vf_coef
        loss.backward()
        return vf_loss.item()


class ValueNorm(jit.ScriptModule):

    def __init__(self, input_shape, beta=0.99999, epsilon=1e-5, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.epsilon = epsilon
        self.beta = beta
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    @jit.script_method
    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @jit.script_method
    def update(self, input_vector):
        if not self.use_norm:
            return
        with torch.no_grad():
            batch_mean = input_vector.mean(dim=0)
            batch_sq_mean = (input_vector ** 2).mean(dim=0)
            weight = self.beta
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    @jit.script_method
    def normalize(self, input_vector):
        if not self.use_norm:
            return input_vector
        mean, var = self.running_mean_var()
        mean = torch.unsqueeze(mean, 0)
        std = torch.sqrt(var).unsqueeze(0)
        out = (input_vector - mean) / std
        return out

    @jit.script_method
    def denormalize(self, input_vector):
        if not self.use_norm:
            return input_vector
        mean, var = self.running_mean_var()
        mean = torch.unsqueeze(mean, 0)
        std = torch.sqrt(var).unsqueeze(0)
        out = input_vector * std + mean
        return out


class ACTContinousLayer(jit.ScriptModule):
    
    def __init__(self, ac_dim, in_dim, gain):
        super().__init__()
        self.dist = Normal(in_dim, ac_dim, gain)
    
    @jit.script_method
    def forward(self, features, deterministic:bool=False):
        means = self.dist.forward(features)
        actions = self.dist.pd_mode(means) if deterministic else self.dist.pd_sample(means)
        scale = self.dist.get_scale(means)
        log_probs = self.dist.pd_log_probs(actions, means, scale)
        return actions, log_probs


class ActorContinuous(jit.ScriptModule):
    
    def __init__(self, ob_dim, ac_dim, h_dim, gain, use_rnn=True):
        super().__init__()
        self.base = MLPBase(ob_dim, h_dim)
        self.rnn = RNNLayer(h_dim, h_dim, use_rnn)
        self.act = ACTContinousLayer(ac_dim, h_dim, gain)
        self.h_dim = h_dim

    @jit.script_method
    def _forward(self, obs, rnn_actor, masks, deterministic:bool):
        features = self.base.forward(obs)
        features, rnn_actor = self.rnn.forward(features, rnn_actor, masks)
        actions, log_probs = self.act.forward(features, deterministic)
        return actions, log_probs, rnn_actor

    @jit.script_method
    def _evaluate_actions(self, obs, rnn_actor, actions, masks):
        features = self.base.forward(obs)
        features, rnn_actor = self.rnn.forward(features, rnn_actor, masks)
        means = self.act.dist.forward(features)
        scale = self.act.dist.get_scale(means)
        log_probs = self.act.dist.pd_log_probs(actions, means, scale)
        dist_entropy = self.act.dist.pd_entropy(scale)
        return log_probs, dist_entropy
    
    @jit.script_method
    def compute_pg_loss(self, obs, rnn_actor, actions, masks, avails, old_log_probs, advs, clip_param:float, ent_coef:float):
        log_probs, entropy = self._evaluate_actions(obs, rnn_actor, actions, masks)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advs
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advs
        pg_loss = torch.sum(torch.minimum(surr1, surr2), dim=-1, keepdim=True)
        pg_loss = - pg_loss.mean()
        entropy = entropy.mean()
        loss = pg_loss - entropy * ent_coef
        loss.backward()
        return pg_loss.item(), entropy.item()


class Discriminator(jit.ScriptModule):

    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(MLPBase(in_dim, h_dim), Linear(h_dim, 1, 0.01))

    @jit.script_method
    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], dim=-1))
    
    @jit.script_method
    def _calculate_reward(self, obs, actions):
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(obs, actions)).squeeze(-1)
    
    @jit.script_method
    def compute_loss(self, obs, actions, exp_actions):
        logits_pi = self.forward(obs, actions)
        logits_exp = self.forward(obs, exp_actions)
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss = loss_pi + loss_exp
        loss.backward()
        return loss_pi.item(), loss_exp.item()


class StateDependentPolicy(jit.ScriptModule):

    def __init__(self, ob_dim, ac_dim, h_dim, hidden_activation=nn.Tanh()):
        super().__init__()
        self.net = build_mlp(ob_dim, 2*ac_dim, [h_dim], hidden_activation)

    @jit.script_method
    def forward(self, states):
        means, _ = self.net(states).chunk(2, dim=-1)
        actions = torch.tanh(means)
        return actions

    @jit.script_method
    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        eps = torch.empty(means.shape).normal_().to(means.device)
        actions = torch.tanh(means + eps * log_stds.clamp(-20, 2).exp())
        return actions

    @jit.script_method
    def log_prob(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        log_stds = log_stds.clamp(-20, 2)
        eps = torch.empty(means.shape).normal_().to(means.device)
        actions = torch.tanh(means + eps * log_stds.exp())
        logp = calculate_log_pi(log_stds, eps, actions)
        return actions, logp.reshape(-1, 1)


class TwinnedStateActionFunction(jit.ScriptModule):

    def __init__(self, ob_dim, ac_dim, h_dim):
        super().__init__()
        self.q1 = build_mlp(ob_dim+ac_dim, 1, [h_dim])
        self.q2 = build_mlp(ob_dim+ac_dim, 1, [h_dim])

    @jit.script_method
    def forward(self, states, actions):
        states = torch.cat([states, actions], dim=-1)
        q1_task = jit.fork(self.q1.forward, states)
        q2_task = jit.fork(self.q2.forward, states)
        return q1_task.wait(), q2_task.wait()
    
    @jit.script_method
    def getQ(self, states, actions):
        Q = torch.cat(self.forward(states, actions), dim=1)
        Q, _ = torch.min(Q, dim=1, keepdim=True)
        return Q
