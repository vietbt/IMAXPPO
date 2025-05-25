import torch
import torch.jit as jit
import torch.nn.functional as F
from models.policies.policy import ActorContinuous, Critic, Discriminator, StateDependentPolicy, TwinnedStateActionFunction


class MAPILPolicyIQ(jit.ScriptModule):

    def __init__(self, ob_dim, s_dim, e_dim, h_dim):
        super().__init__()
        opp_dim = e_dim - s_dim
        self.opp_actor = StateDependentPolicy(ob_dim, opp_dim, h_dim)
        self.opp_critic = TwinnedStateActionFunction(ob_dim, opp_dim, h_dim)
        self.opp_critic_target = TwinnedStateActionFunction(ob_dim, opp_dim, h_dim).eval()
        self.opp_critic_target.load_state_dict(self.opp_critic.state_dict())
        self.ob_dim = ob_dim
        self.ac_dim: int = opp_dim
        self.s_dim: int = s_dim
        self.e_dim: int = e_dim

    @jit.script_method
    def getV(self, obs, ent_coef):
        action, log_prob = self.opp_actor.log_prob(obs)
        current_Q = self.opp_critic.getQ(obs, action)
        current_V = current_Q - ent_coef * log_prob
        return current_V

    @jit.script_method
    def get_targetV(self, obs, ent_coef):
        action, log_prob = self.opp_actor.log_prob(obs)
        target_Q = self.opp_critic_target.getQ(obs, action)
        target_V = target_Q - ent_coef * log_prob
        return target_V

    @jit.script_method
    def get_actions(self, obs, deterministic: bool=False):
        with torch.no_grad():
            return self.opp_actor.forward(obs) if deterministic else self.opp_actor.sample(obs)
    
    @jit.script_method
    def update_critic(self, obs, next_obs, actions, rewards, masks, ent_coef, gamma:float):
        with torch.no_grad():
            next_actions, next_log_prob = self.opp_actor.log_prob(next_obs)
            next_q_values = torch.cat(self.opp_critic_target.forward(next_obs, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            next_q_values = next_q_values - ent_coef * next_log_prob
            target_q_values = rewards + masks * gamma * next_q_values
        q1_values, q2_values = self.opp_critic.forward(obs, actions)
        critic_loss_1 = F.mse_loss(q1_values, target_q_values)
        critic_loss_2 = F.mse_loss(q2_values, target_q_values)
        critic_loss = 0.5 * (critic_loss_1 + critic_loss_2)
        critic_loss.backward()
        return critic_loss

    @jit.script_method
    def iq_loss(self, current_V, next_V, current_Q, masks, gamma:float):
        y = masks * gamma * next_V
        reward = current_Q - y
        loss = - reward.mean()
        loss += (current_V - y).mean()
        loss += 0.5 * (reward**2).mean()
        return loss

    @jit.script_method
    def update_iq_learn(self, obs, next_obs, actions, masks, ent_coef, gamma:float):
        current_V = self.getV(obs, ent_coef)
        with torch.no_grad():
            next_V = self.get_targetV(next_obs, ent_coef)
        futures = [self.iq_loss(current_V, next_V, current_Q, masks, gamma) for current_Q in self.opp_critic.forward(obs, actions)]
        critic_loss = torch.stack(futures).mean()
        critic_loss.backward()
        return critic_loss
    
    @jit.script_method
    def update_actor(self, obs, ent_coef):
        actions_pi, log_prob = self.opp_actor.log_prob(obs)
        q_values_pi = torch.cat(self.opp_critic.forward(obs, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = torch.mean(ent_coef * log_prob - min_qf_pi)
        actor_loss.backward()
        return actor_loss, log_prob.detach()


class MAPILPolicySup(jit.ScriptModule):

    def __init__(self, ob_dim, s_dim, e_dim, h_dim):
        super().__init__()
        opp_dim = e_dim - s_dim
        self.opp_actor = StateDependentPolicy(ob_dim, opp_dim, h_dim)
        self.ob_dim = ob_dim
        self.ac_dim: int = opp_dim
        self.s_dim: int = s_dim
        self.e_dim: int = e_dim

    @jit.script_method
    def get_actions(self, obs, deterministic: bool=False):
        with torch.no_grad():
            return self.opp_actor.forward(obs) if deterministic else self.opp_actor.sample(obs)
    
    @jit.script_method
    def update_actor(self, obs, action_exp):
        actions_pi = self.opp_actor.sample(obs)
        actor_loss = F.mse_loss(actions_pi, action_exp)
        actor_loss.backward()
        return actor_loss


class MAPILPolicyGAIL(jit.ScriptModule):
    
    def __init__(self, ob_dim, st_dim, s_dim, e_dim, h_dim, gain=0.01, use_huber=False, use_norm=True, use_rnn=True, **kwargs):
        super().__init__()
        ac_dim = e_dim - s_dim
        self.actor = ActorContinuous(ob_dim, ac_dim, h_dim, gain, use_rnn)
        self.critic = Critic(st_dim, h_dim, use_huber, use_norm, use_rnn)
        self.disc = Discriminator(ob_dim + ac_dim, h_dim)
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.ac_dim = ac_dim
        
    @jit.script_method
    def get_actions(self, states, obs, rnn_actor, rnn_critic, masks, deterministic:bool=False):
        actions, action_log_probs, rnn_actor = self.actor._forward(obs, rnn_actor, masks, deterministic)
        values, rnn_critic = self.critic._forward(states, rnn_critic, masks)
        return values, actions, action_log_probs, rnn_actor, rnn_critic

    @jit.script_method
    def get_values(self, states, rnn_critic, masks):
        values, _ = self.critic._forward(states, rnn_critic, masks)
        return values

    @jit.script_method
    def act(self, obs, rnn_actor, masks, deterministic:bool=True):
        actions, _, rnn_actor = self.actor._forward(obs, rnn_actor, masks, deterministic)
        return actions, rnn_actor

    @jit.script_method
    def calculate_reward(self, obs, actions):
        return self.disc._calculate_reward(obs, actions)