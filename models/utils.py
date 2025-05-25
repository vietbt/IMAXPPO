import math
from typing import List
import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F


class Linear(nn.Linear):
    
    def __init__(self, in_features, out_features, gain=math.sqrt(2), init_bias=0.0, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.orthogonal_(self.weight, gain=gain)
        self.bias.data.fill_(init_bias)


class MLPBase(jit.ScriptModule):

    def __init__(self, ob_dim, h_dim):
        super().__init__()
        self.feature_norm = nn.LayerNorm(ob_dim)
        self.fc = nn.Sequential(Linear(ob_dim, h_dim), nn.ReLU(), nn.LayerNorm(h_dim))

    @jit.script_method
    def forward(self, obs):
        obs_norm = self.feature_norm(obs)
        features = self.fc(obs_norm)
        return features


class RNNLayer(jit.ScriptModule):

    def __init__(self, in_dim, h_dim, use_rnn=True):
        super().__init__()
        self.use_rnn = use_rnn
        self.rnn = nn.GRU(in_dim, h_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        try:
            self.rnn = self.rnn.cuda()
            self.rnn.flatten_parameters()
        except:
            pass        
        self.norm = nn.LayerNorm(h_dim)

    @jit.script_method
    def forward_old(self, inputs, hxs, masks):
        n_envs = hxs.size(0)
        n_steps = inputs.size(0) // n_envs
        inputs = inputs.view(-1, n_envs, inputs.size(1))
        masks = masks.view(-1, n_envs, 1)
        hxs = hxs.transpose(0, 1).squeeze(0)
        all_out = []
        for i in range(n_steps):
            hxs = torch.gru_cell(inputs[i], hxs * masks[i], self.rnn.weight_ih_l0, self.rnn.weight_hh_l0, self.rnn.bias_ih_l0, self.rnn.bias_hh_l0)
            all_out.append(hxs)
        out = torch.stack(all_out).reshape(n_steps * n_envs, -1)
        hxs = torch.unsqueeze(hxs, 1)
        out = self.norm.forward(out)
        return out, hxs
    
    @jit.script_method
    def forward_all(self, inputs, hxs, masks):
        n_envs = hxs.size(0)
        n_steps = inputs.size(0) // n_envs

        hxs = hxs.transpose(0, 1)
        inputs = inputs.view(n_steps, n_envs, -1)
        masks = masks.view(n_steps, n_envs, 1)

        mask_ids: List[int] = (1 - masks).squeeze(-1).sum(-1).nonzero().squeeze(-1).tolist()
        mask_ids.append(n_steps)
        mask_ids.insert(0, 0)
        all_out: List[torch.Tensor] = []
        for i in range(len(mask_ids) - 1):
            s_id = mask_ids[i]
            e_id = mask_ids[i+1]
            if s_id < e_id:
                hxs = hxs * masks[s_id].unsqueeze(0)
                gru_out, hxs = self.rnn.forward(inputs[s_id:e_id], hxs)
                all_out.append(gru_out)
        hxs = hxs.transpose(0, 1)
        out = torch.cat(all_out).flatten(0, 1)
        return out, hxs

    @jit.script_method
    def forward(self, inputs, hxs, masks):
        if not self.use_rnn:
            return inputs, hxs
        if inputs.size(0) == hxs.size(0):
            hxs = hxs * masks.unsqueeze(-1)
            out, hxs = self.rnn.forward(inputs.unsqueeze(0), hxs.transpose(0, 1))
            hxs = hxs.transpose(0, 1)
            out = out.squeeze(0)
        else:
            out, hxs = self.forward_all(inputs, hxs, masks)
        out = self.norm.forward(out)
        return out, hxs


def build_mlp(input_dim, output_dim, hidden_units=[128, 128], hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(Linear(units, output_dim, 1))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


@jit.script
def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


class Categorical(jit.ScriptModule):

    def __init__(self, num_inputs, num_outputs, gain):
        super().__init__()
        self.linear = Linear(num_inputs, num_outputs, gain)
        self.min_real = torch.finfo(torch.float32).min

    @jit.script_method
    def forward(self, features, avails):
        logits = self.linear(features)
        logits[avails==0] = -float("inf")
        return logits
    
    @jit.script_method
    def pd_mode(self, probs):
        return probs.argmax(-1, keepdim=True)
    
    @jit.script_method
    def pd_sample(self, probs, logits):
        probs_2d = probs.reshape(-1, logits.size()[-1])
        samples_2d = torch.multinomial(probs_2d, 1, True).T
        return samples_2d.reshape(logits.size()[:-1]).unsqueeze(-1)

    @jit.script_method
    def pd_log_probs(self, value, logits):
        value = value.squeeze(-1)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        log_probs = log_pmf.gather(-1, value).squeeze(-1)
        log_probs = log_probs.view(value.size(0), -1).sum(-1)
        return log_probs.unsqueeze(-1)

    @jit.script_method
    def pd_entropy(self, logits, probs):
        logits = torch.clamp(logits, min=self.min_real)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)
    

class ACTLayer(jit.ScriptModule):
    
    def __init__(self, ac_dim, in_dim, gain):
        super().__init__()
        self.dist = Categorical(in_dim, ac_dim, gain)
    
    @jit.script_method
    def forward(self, features, avails, deterministic:bool=False):
        logits = self.dist.forward(features, avails)
        logits = logits - torch.logsumexp(logits, -1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        actions = self.dist.pd_mode(probs) if deterministic else self.dist.pd_sample(probs, logits) 
        log_probs = self.dist.pd_log_probs(actions, logits)
        return actions, log_probs


class Normal(jit.ScriptModule):

    def __init__(self, num_inputs, num_outputs, gain):
        super().__init__()
        self.linear = Linear(num_inputs, num_outputs, gain)
        self.log_std = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)

    @jit.script_method
    def get_scale(self, means):
        return torch.ones_like(means) * self.log_std.exp()
    
    @jit.script_method
    def forward(self, features):
        return self.linear(features)
    
    @jit.script_method
    def pd_mode(self, means):
        return means
    
    @jit.script_method
    def pd_sample(self, means):
        with torch.no_grad():
            std = torch.ones_like(means) * self.log_std.exp()
            return torch.normal(means, std)
    
    @jit.script_method
    def pd_log_probs(self, actions, means, scale):
        var = scale ** 2
        log_scale = scale.log()
        log_probs =  -((actions - means) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        return log_probs.sum(-1, keepdim=True)
    
    @jit.script_method
    def pd_entropy(self, scale):
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)
        return entropy
