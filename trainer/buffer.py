import torch
import numpy as np


class Buffer(object):

    def __init__(self, n_envs, n_agents, n_enemies, rnn_actor_dim, rnn_critic_dim, gamma, lam):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.rnn_actor_dim = rnn_actor_dim
        self.rnn_critic_dim = rnn_critic_dim
        self.lam = lam
        self.gamma = gamma
        self.obs, self.rewards, self.actions, self.values, self.masks, self.log_probs, self.states, self.avails, self.rnn_actor, self.rnn_critic = [], [], [], [], [], [], [], [], [], []
        self.infos = []
    
    def init(self, obs, states, avails):
        self._obs, self._states, self._avails = obs, states, avails
        self._masks = np.ones((self.n_envs, self.n_agents, 1))
        self._rnn_actor = np.zeros((self.n_envs, self.n_agents, 1, self.rnn_actor_dim))
        self._rnn_critic = np.zeros((self.n_envs, self.n_agents, 1, self.rnn_critic_dim))
        self.infos = []
    
    @property
    def data(self):
        return self._states, self._obs, self._rnn_actor, self._rnn_critic, self._masks, self._avails
    
    def insert(self, obs, states, avails, actions, values, log_probs, rewards, masks, rnn_actor, rnn_critic):
        self.obs.append(self._obs)
        self.states.append(self._states)
        self.avails.append(self._avails)
        self.masks.append(self._masks)
        self.actions.append(actions)
        self.values.append(values)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.rnn_actor.append(self._rnn_actor)
        self.rnn_critic.append(self._rnn_critic)
        self._rnn_actor = rnn_actor
        self._rnn_critic = rnn_critic
        self._obs, self._states, self._avails, self._masks = obs, states, avails, masks
    
    def compute_returns(self, last_values, denormalize):
        self.obs, self.states, self.avails, self.masks, self.actions, self.values, self.log_probs, self.rewards, self.rnn_actor, self.rnn_critic \
            = map(np.asarray, (self.obs, self.states, self.avails, self.masks, self.actions, self.values, self.log_probs, self.rewards, self.rnn_actor, self.rnn_critic))
        advs = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)
        n_steps = len(self.rewards)
        last_gae_lam = 0
        
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                nextnonterminal = self._masks
                nextvalues = last_values
            else:
                nextnonterminal = self.masks[step + 1]
                nextvalues = self.values[step + 1]

            nextvalues = denormalize(nextvalues)
            currentvalues = denormalize(self.values[step])
            self.returns[step] = self.rewards[step] + self.gamma * nextnonterminal * (nextvalues + self.lam * last_gae_lam)
            advs[step] = last_gae_lam = self.returns[step] - currentvalues
            
        if False:    # TODO
            self.active_masks = 1 - self.avails[...,:1]
            advs_copy = advs.copy()
            advs_copy[self.active_masks==0.0] = np.nan
            mean_advs = np.nanmean(advs_copy)
            std_advs = np.nanstd(advs_copy)
            self.advs = (advs - mean_advs) / (std_advs + 1e-8)
        else:
            self.advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    def reset(self):
        self.obs, self.rewards, self.actions, self.values, self.masks, self.log_probs, self.states, self.avails, self.rnn_actor, self.rnn_critic = [], [], [], [], [], [], [], [], [], []
        self.infos = []

    def recurrent_generator(self, n_minibatch):
        chunk_length = len(self.rewards) # TODO: check this settings
        batch_size = np.prod(self.rewards.shape)
        data_chunks = batch_size // chunk_length
        mini_batch_size = data_chunks // n_minibatch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(n_minibatch)]
        _cast = lambda x: x.transpose(1, 2, 0, 3).reshape(-1, x.shape[-1])
        obs, states, avails, masks, actions, values, log_probs, returns, advs \
            = map(_cast, (self.obs, self.states, self.avails, self.masks, self.actions, self.values, self.log_probs, self.returns, self.advs))
        rnn_actor = self.rnn_actor.transpose(1, 2, 0, 3, 4).reshape(-1, 1, self.rnn_actor_dim)
        rnn_critic = self.rnn_critic.transpose(1, 2, 0, 3, 4).reshape(-1, 1, self.rnn_critic_dim)
        for indices in sampler:
            ids = [list(range(id*chunk_length, (id+1)*chunk_length)) for id in indices]
            env_ids = [id*chunk_length for id in indices]
            _get_batch = lambda x: np.stack([x[id] for id in ids], 1).reshape(-1, x.shape[-1])
            _obs, _states, _avails, _masks, _actions, _values, _log_probs, _returns, _advs \
                = map(_get_batch, (obs, states, avails, masks, actions, values, log_probs, returns, advs))
            _rnn_actor = np.stack([rnn_actor[id] for id in env_ids]).reshape(-1, 1, self.rnn_actor_dim)
            _rnn_critic = np.stack([rnn_critic[id] for id in env_ids]).reshape(-1, 1, self.rnn_critic_dim)
            yield _obs, _states, _avails, _masks, _actions, _values, _log_probs, _returns, _advs, _rnn_actor, _rnn_critic


        
