import torch.jit as jit
from models.policies.policy import Actor, Critic


class MAPPOPolicy(jit.ScriptModule):

    def __init__(self, ob_dim, st_dim, ac_dim, h_dim, gain, use_huber=False, use_norm=True, use_rnn=True, **kwargs):
        super().__init__()
        self.actor = Actor(ob_dim, ac_dim, h_dim, gain, use_rnn)
        self.critic = Critic(st_dim, h_dim, use_huber, use_norm, use_rnn)
        
    @jit.script_method
    def get_actions(self, states, obs, rnn_actor, rnn_critic, masks, avails, deterministic:bool=False):
        actions, action_log_probs, rnn_actor = self.actor._forward(obs, rnn_actor, masks, avails, deterministic)
        values, rnn_critic = self.critic._forward(states, rnn_critic, masks)
        return values, actions, action_log_probs, rnn_actor, rnn_critic

    @jit.script_method
    def get_values(self, states, rnn_critic, masks):
        values, _ = self.critic._forward(states, rnn_critic, masks)
        return values

    @jit.script_method
    def act(self, obs, rnn_actor, masks, avails, deterministic:bool=True):
        actions, _, rnn_actor = self.actor._forward(obs, rnn_actor, masks, avails, deterministic)
        return actions, rnn_actor