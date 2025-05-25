import torch
import numpy as np
from tqdm import tqdm
from trainer.buffer import Buffer
from utils import convert_to_tensor
from models.agents.mappo import MAPPOPolicy
from models.agents.imaxppo import MAPILPolicyGAIL


class SMACRunner:

    def __init__(self, envs, policy: MAPPOPolicy, imitator, n_steps, gamma, lam, device, use_imitation, **kwargs):
        self.envs = envs
        self.n_steps = n_steps
        self.policy = policy
        self.imitator = imitator
        self.use_imitation = use_imitation
        self.buffer = Buffer(envs.n_envs, envs.n_agents, envs.n_enemies, policy.actor.h_dim, policy.critic.h_dim, gamma, lam)
        self.buffer_imitator = Buffer(envs.n_envs, envs.n_agents, envs.n_enemies, imitator.actor.h_dim, imitator.critic.h_dim, gamma, lam) if isinstance(imitator, MAPILPolicyGAIL) else None
        
        self._to_tensor = lambda x: convert_to_tensor(x, device=device)
        self._denormalizer = lambda x: policy.critic.norm.denormalize(self._to_tensor(x)).detach().cpu().numpy()
        self._denormalizer_imitator = lambda x: imitator.critic.norm.denormalize(self._to_tensor(x)).detach().cpu().numpy() if self.use_imitation else None
        obs, states, avails = envs.reset()
        self.buffer.init(obs, states, avails)
        self.buffer_imitator.init(obs, states, avails) if self.buffer_imitator else None
        self._converter = lambda x: self._to_tensor(np.concatenate(x))
        self._flatter = lambda x: np.array(np.split(x.detach().cpu().numpy(), envs.n_envs))
        
    @torch.no_grad()
    def prepare_buffer(self):
        _, states, _ = self.envs.get_current_states()
        states, rnn_critic, masks = map(self._converter, (states, self.buffer._rnn_critic, self.buffer._masks))
        next_values = self._flatter(self.policy.get_values(states, rnn_critic, masks))
        self.buffer.compute_returns(next_values, self._denormalizer)
        if isinstance(self.imitator, MAPILPolicyGAIL):
            rnn_critic_opp = self._converter(self.buffer_imitator._rnn_critic)
            next_values_opp = self._flatter(self.policy.get_values(states, rnn_critic_opp, masks))
            self.buffer_imitator.compute_returns(next_values_opp, self._denormalizer_imitator)
        return self.buffer, self.buffer_imitator

    def run(self, verbose=True):
        self.policy.eval()
        self.imitator.eval() if self.use_imitation else None
        self.buffer.reset()
        self.buffer_imitator.reset() if self.buffer_imitator else None
        for _ in tqdm(range(self.n_steps), f"Collecting ...", leave=False, mininterval=0.1, ncols=0, disable=not verbose):
            if isinstance(self.imitator, MAPILPolicyGAIL):
                values_opp, actions_opp, log_probs_opp, rnn_actor_opp, rnn_critic_opp = self.collect_imitator()
                self.buffer._obs = np.concatenate((self.buffer._obs, actions_opp), -1)
            else:
                actions_opp = self._flatter(self.imitator.get_actions(self._converter(self.buffer._obs)))
                self.buffer._obs = np.concatenate((self.buffer._obs, actions_opp), -1)
            values, actions, log_probs, rnn_actor, rnn_critic = self.collect()
            obs, states, avails, rewards, dones, infos = self.envs.step(actions)
            if any("abnormal" in info for info in infos):
                print("Environment error!")
                self.buffer.reset()
                self.buffer_imitator.reset() if self.buffer_imitator else None
                obs, states, avails = self.envs.reset()
                self.buffer.init(obs, states, avails)
                self.buffer_imitator.init(obs, states, avails) if self.buffer_imitator else None
                break
            if isinstance(self.imitator, MAPILPolicyGAIL):
                _obs = self._to_tensor(self.buffer_imitator._obs)
                _actions = self._to_tensor(actions_opp)
                rewards_opp = self.imitator.calculate_reward(_obs, _actions).unsqueeze(-1).detach().cpu().numpy()
                self.insert_imitator(obs, states, rewards_opp, dones, avails, values_opp, actions_opp, log_probs_opp, rnn_actor_opp, rnn_critic_opp)
            self.insert(obs, states, rewards, dones, avails, values, actions, log_probs, rnn_actor, rnn_critic)
            self.buffer.infos.extend(infos)
        if len(self.buffer.infos) == 0:
            return self.run(verbose)
        self.policy.train()
        self.imitator.train() if self.use_imitation else None
        return self.prepare_buffer()

    @torch.no_grad()
    def collect_imitator(self):
        return map(self._flatter, self.imitator.get_actions(*map(self._converter, self.buffer_imitator.data[:-1])))

    @torch.no_grad()
    def collect(self):
        return map(self._flatter, self.policy.get_actions(*map(self._converter, self.buffer.data)))

    def insert(self, obs, states, rewards, dones, avails, values, actions, log_probs, rnn_actor, rnn_critic):
        dones = np.all(dones, axis=1)
        rnn_actor[dones] = np.zeros((dones.sum(), self.buffer.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
        rnn_critic[dones] = np.zeros((dones.sum(), self.buffer.n_agents, 1, self.policy.critic.h_dim), dtype=np.float32)
        masks = np.ones((self.envs.n_envs, self.buffer.n_agents, 1), dtype=np.float32)
        masks[dones] = np.zeros((dones.sum(), self.buffer.n_agents, 1), dtype=np.float32)
        self.buffer.insert(obs, states, avails, actions, values, log_probs, rewards, masks, rnn_actor, rnn_critic)

    def insert_imitator(self, obs, states, rewards, dones, avails, values, actions, log_probs, rnn_actor, rnn_critic):
        dones = np.all(dones, axis=1)
        rnn_actor[dones] = np.zeros((dones.sum(), self.buffer_imitator.n_agents, 1, self.imitator.actor.h_dim), dtype=np.float32)
        rnn_critic[dones] = np.zeros((dones.sum(), self.buffer_imitator.n_agents, 1, self.imitator.critic.h_dim), dtype=np.float32)
        masks = np.ones((self.envs.n_envs, self.buffer_imitator.n_agents, 1), dtype=np.float32)
        masks[dones] = np.zeros((dones.sum(), self.buffer_imitator.n_agents, 1), dtype=np.float32)
        self.buffer_imitator.insert(obs, states, avails, actions, values, log_probs, rewards, masks, rnn_actor, rnn_critic)
