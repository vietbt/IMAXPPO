import time
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from trainer.buffer import Buffer
from utils import convert_to_tensor
from models.agents.mappo import MAPPOPolicy
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, wait
from models.agents.imaxppo import MAPILPolicySup, MAPILPolicyIQ, MAPILPolicyGAIL


def update_actor(actor, optim, data, noptepochs, cliprange, ent_coef, max_grad_norm):
    lossvals = []
    for _ in range(noptepochs):
        for obs, rnn_actor, actions, masks, avails, old_log_probs, advs in data:
            optim.zero_grad()
            lossvals.append(actor.compute_pg_loss(obs, rnn_actor, actions, masks, avails, old_log_probs, advs, cliprange, ent_coef))
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            optim.step()
    pg_loss, entropy = np.mean(lossvals, 0)
    return pg_loss, entropy


def update_critic(critic, optim, data, noptepochs, cliprange, vf_coef, max_grad_norm):
    lossvals = []
    for _ in range(noptepochs):
        for states, rnn_critic, masks, old_values, returns in data:
            optim.zero_grad()
            lossvals.append(critic.compute_vf_loss(states, rnn_critic, masks, old_values, returns, cliprange, vf_coef))
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optim.step()
    vf_loss = np.mean(lossvals)
    return vf_loss


def update_disc(disc, optim, obs_opp, action_opp, obs_exp, noptepochs):
    lossvals = []
    for _ in range(noptepochs):
        optim.zero_grad()
        lossvals.append(disc.compute_loss(obs_opp, action_opp, obs_exp))
        optim.step()
    loss_pi, loss_exp = np.mean(lossvals, 0)
    return loss_pi, loss_exp


class MATrainer():
    
    def __init__(self, policy: MAPPOPolicy, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, **kwargs):
        self.total_steps = 0
        self.device = device
        self.map_name = map_name
        self.cliprange = cliprange
        self.noptepochs = noptepochs
        self.nminibatches = nminibatches
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_imitation = use_imitation
        self.writer = SummaryWriter(log_dir)
        self._converter = partial(convert_to_tensor, device=device)
        self.policy = policy.to(device)
        self.lr = lr
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.optims = [self.actor_optimizer, self.critic_optimizer]
        self.start_time = None

    def set_learning_rate(self, rate):
        for optim in self.optims:
            for group in optim.param_groups:
                group['lr'] = self.lr * rate

    def update_writer(self, buffer, pg_loss, vf_loss, entropy):
        self.total_steps += self.n_steps
        self.writer.add_scalar(f"losses/pg_loss", pg_loss, self.total_steps)
        self.writer.add_scalar(f"losses/vf_loss", vf_loss, self.total_steps)
        self.writer.add_scalar(f"losses/entropy", entropy, self.total_steps)
        self.writer.add_scalar(f"losses/reward", np.mean(buffer.rewards), self.total_steps)
        if self.start_time is not None:
            end_time = time.time()
            fps = self.n_steps / (end_time - self.start_time)
            self.writer.add_scalar(f"losses/fps", fps, self.total_steps)
        else:
            fps = None
        self.start_time = time.time()
        logs = [info for info in buffer.infos if "go_count" in info]
        if len(logs) > 0 and fps is not None:
            dead_allies     = np.mean([info.get("dead_allies", -1) for info in logs])
            dead_enemies    = np.mean([info.get("dead_enemies", -1) for info in logs])
            go_count        = np.mean([info["go_count"] for info in logs])
            winrate         = np.mean([info["won"] for info in logs])
            self.writer.add_scalar(f"game/winrate", winrate, self.total_steps)
            if dead_allies > 0:
                self.writer.add_scalar(f"game/dead_allies", dead_allies, self.total_steps)
            if dead_enemies > 0:
                self.writer.add_scalar(f"game/dead_enemies", dead_enemies, self.total_steps)
            self.writer.add_scalar(f"losses/go_count", go_count, self.total_steps)
            loss_str = f"{abs(pg_loss):.2f}/{vf_loss:.2f}/{entropy:.2f}"
            print(f"| Map {self.map_name:.^20} | Loss: {loss_str} | Game: {winrate:.2f}/{dead_allies:.2f}/{dead_enemies:.2f} | FPS: {fps:.0f} |")
        else:
            winrate = 0
        return winrate

    def train(self, buffer: Buffer, *args, **kwargs):
        self.policy.train()
        data_actor, data_critic = [], []
        data_generator = buffer.recurrent_generator(self.nminibatches)
        for sample in data_generator:
            obs, states, avails, masks, actions, old_values, old_log_probs, returns, advs, rnn_actor, rnn_critic = map(self._converter, sample)
            data_actor.append((obs, rnn_actor, actions, masks, avails, old_log_probs, advs))
            data_critic.append((states, rnn_critic, masks, old_values, returns))
        with ThreadPoolExecutor() as executor:
            features = []
            features.append(executor.submit(update_actor, self.policy.actor, self.actor_optimizer, data_actor, self.noptepochs, self.cliprange, self.ent_coef, self.max_grad_norm))
            features.append(executor.submit(update_critic, self.policy.critic, self.critic_optimizer, data_critic, self.noptepochs, self.cliprange, self.vf_coef, self.max_grad_norm))
            wait(features)
            (pg_loss, entropy), vf_loss = [feat.result() for feat in features]
        return self.update_writer(buffer, pg_loss, vf_loss, entropy)
        
    @torch.no_grad()
    def eval(self, envs, n_episodes=32):
        if envs is None:
            return
        _flatter = lambda x: np.array(np.split(x.detach().cpu().numpy(), envs.n_envs))
        _converter = lambda x: convert_to_tensor(np.concatenate(x), device=self.device)
        logs = []
        obs, _, avails = envs.reset()
        rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
        masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
        self.policy.eval()
        while len(logs) < n_episodes:
            _obs, _rnn_actor, _masks, _avails = map(_converter, (obs, rnn_actor, masks, avails))
            actions, rnn_actor = self.policy.act(_obs, _rnn_actor, _masks, _avails, deterministic=True)
            actions, rnn_actor = map(_flatter, (actions, rnn_actor))
            obs, _, avails, _, dones, infos = envs.step(actions)
            if any("abnormal" in info for info in infos):
                print("Environment error!")
                rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
                masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
                obs, _, avails = envs.reset()
                continue
            logs.extend([info for info in infos if "go_count" in info])
            dones = np.all(dones, axis=1)
            rnn_actor[dones] = np.zeros((dones.sum(), envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
            masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros((dones.sum(), envs.n_agents, 1), dtype=np.float32)
        dead_allies     = np.mean([info.get("dead_allies", -1) for info in logs])
        dead_enemies    = np.mean([info.get("dead_enemies", -1) for info in logs])
        go_count        = np.mean([info["go_count"] for info in logs])
        winrate         = np.mean([info["won"] for info in logs])
        self.writer.add_scalar(f"game_eval/winrate", winrate, self.total_steps)
        if dead_allies > 0:
            self.writer.add_scalar(f"game_eval/dead_allies", dead_allies, self.total_steps)
        if dead_enemies > 0:
            self.writer.add_scalar(f"game_eval/dead_enemies", dead_enemies, self.total_steps)
        self.writer.add_scalar(f"losses/go_count_eval", go_count, self.total_steps)
        print(f"| Map {self.map_name:.^20} | Evaluate: {winrate:.2f}/{dead_allies:.2f}/{dead_enemies:.2f} |")
        return winrate


class MAPILTrainerGAIL(MATrainer):

    def __init__(self, policy, imitator: MAPILPolicyGAIL, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, **kwargs):
        super().__init__(policy, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, **kwargs)
        self.imitator = imitator.to(device)
        self.actor_imitator_optimizer = torch.optim.Adam(imitator.actor.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.critic_imitator_optimizer = torch.optim.Adam(imitator.critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.disc_optimizer = torch.optim.Adam(imitator.disc.parameters(), lr=1e-3)
        self.optims.extend([self.actor_imitator_optimizer, self.critic_imitator_optimizer, self.disc_optimizer])

    def train(self, buffer: Buffer, buffer_imitator: Buffer):
        self.policy.train()
        self.imitator.train()

        data_actor, data_critic = [], []
        data_generator = buffer.recurrent_generator(self.nminibatches)
        for sample in data_generator:
            obs, states, avails, masks, actions, old_values, old_log_probs, returns, advs, rnn_actor, rnn_critic = map(self._converter, sample)
            data_actor.append((obs, rnn_actor, actions, masks, avails, old_log_probs, advs))
            data_critic.append((states, rnn_critic, masks, old_values, returns))
        
        data_actor_opp, data_critic_opp = [], []
        data_generator = buffer_imitator.recurrent_generator(self.nminibatches)
        for sample in data_generator:
            obs, states, avails, masks, actions, old_values, old_log_probs, returns, advs, rnn_actor, rnn_critic = map(self._converter, sample)
            data_actor_opp.append((obs, rnn_actor, actions, masks, avails, old_log_probs, advs))
            data_critic_opp.append((states, rnn_critic, masks, old_values, returns))
    
        obs_opp, action_opp, avails_opp = map(self._converter, (buffer_imitator.obs, buffer_imitator.actions, buffer_imitator.avails))
        masks = avails_opp[..., 0] == 0.0
        obs_opp = obs_opp[masks]
        action_opp = action_opp[masks]
        obs_exp = obs_opp[1:,...,self.imitator.s_dim:self.imitator.e_dim]
        obs_opp = obs_opp[:-1]
        action_opp = action_opp[:-1]

        with ThreadPoolExecutor() as executor:
            features = []
            features.append(executor.submit(update_actor, self.policy.actor, self.actor_optimizer, data_actor, self.noptepochs, self.cliprange, self.ent_coef, self.max_grad_norm))
            features.append(executor.submit(update_critic, self.policy.critic, self.critic_optimizer, data_critic, self.noptepochs, self.cliprange, self.vf_coef, self.max_grad_norm))
            features.append(executor.submit(update_actor, self.imitator.actor, self.actor_imitator_optimizer, data_actor_opp, 1, self.cliprange, 0, self.max_grad_norm))
            features.append(executor.submit(update_critic, self.imitator.critic, self.critic_imitator_optimizer, data_critic_opp, 1, self.cliprange, self.vf_coef, self.max_grad_norm))
            features.append(executor.submit(update_disc, self.imitator.disc, self.disc_optimizer, obs_opp, action_opp, obs_exp, self.noptepochs))
            wait(features)
            (pg_loss, entropy), vf_loss, (pg_loss_opp, entropy_opp), vf_loss_opp, (loss_pi, loss_exp) = [feat.result() for feat in features]
        
        winrate = self.update_writer(buffer, pg_loss, vf_loss, entropy)
        self.writer.add_scalar(f"losses/loss_pi", loss_pi, self.total_steps)
        self.writer.add_scalar(f"losses/loss_exp", loss_exp, self.total_steps)
        self.writer.add_scalar(f"losses/pg_loss_opp", pg_loss_opp, self.total_steps)
        self.writer.add_scalar(f"losses/vf_loss_opp", vf_loss_opp, self.total_steps)
        self.writer.add_scalar(f"losses/entropy_opp", entropy_opp, self.total_steps)
        return winrate
    
    @torch.no_grad()
    def eval(self, envs, n_episodes=32):
        if envs is None:
            return
        _flatter = lambda x: np.array(np.split(x.detach().cpu().numpy(), envs.n_envs))
        _converter = lambda x: convert_to_tensor(np.concatenate(x), device=self.device)
        logs = []
        obs, _, avails = envs.reset()
        rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
        rnn_actor_opp = np.zeros((envs.n_envs, envs.n_agents, 1, self.imitator.actor.h_dim), dtype=np.float32)
        masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
        self.policy.eval()
        self.imitator.eval()
        while len(logs) < n_episodes:
            _obs, _rnn_actor, _rnn_actor_opp, _masks, _avails = map(_converter, (obs, rnn_actor, rnn_actor_opp, masks, avails))
            actions_opp, rnn_actor_opp = self.imitator.act(_obs, _rnn_actor_opp, _masks, deterministic=True)
            _obs = torch.cat((_obs, actions_opp), -1)
            actions, rnn_actor = self.policy.act(_obs, _rnn_actor, _masks, _avails, deterministic=True)
            actions, rnn_actor, rnn_actor_opp = map(_flatter, (actions, rnn_actor, rnn_actor_opp))
            obs, _, avails, _, dones, infos = envs.step(actions)
            if any("abnormal" in info for info in infos):
                print("Environment error!")
                rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
                rnn_actor_opp = np.zeros((envs.n_envs, envs.n_agents, 1, self.imitator.actor.h_dim), dtype=np.float32)
                masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
                obs, _, avails = envs.reset()
                continue
            logs.extend([info for info in infos if "go_count" in info])
            dones = np.all(dones, axis=1)
            rnn_actor[dones] = np.zeros((dones.sum(), envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
            rnn_actor_opp[dones] = np.zeros((dones.sum(), envs.n_agents, 1, self.imitator.actor.h_dim), dtype=np.float32)
            masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros((dones.sum(), envs.n_agents, 1), dtype=np.float32)
        dead_allies     = np.mean([info.get("dead_allies", -1) for info in logs])
        dead_enemies    = np.mean([info.get("dead_enemies", -1) for info in logs])
        go_count        = np.mean([info["go_count"] for info in logs])
        winrate         = np.mean([info["won"] for info in logs])
        self.writer.add_scalar(f"game_eval/winrate", winrate, self.total_steps)
        if dead_allies > 0:
            self.writer.add_scalar(f"game_eval/dead_allies", dead_allies, self.total_steps)
        if dead_enemies > 0:
            self.writer.add_scalar(f"game_eval/dead_enemies", dead_enemies, self.total_steps)
        self.writer.add_scalar(f"losses/go_count_eval", go_count, self.total_steps)
        print(f"| Map {self.map_name:.^20} | Evaluate: {winrate:.2f}/{dead_allies:.2f}/{dead_enemies:.2f} |")
        return winrate


class MAPILTrainerIQ(MATrainer):

    def __init__(self, policy, imitator: MAPILPolicyIQ, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, gamma, **kwargs):
        super().__init__(policy, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, **kwargs)
        self.imitator = imitator.to(device)
        self.target_entropy = -imitator.ac_dim
        self.log_ent_coef = torch.log(torch.ones(1, device=device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=lr)
        self.actor_opp_optim = torch.optim.Adam(imitator.opp_actor.parameters(), lr=lr)
        self.critic_opp_optim = torch.optim.Adam(imitator.opp_critic.parameters(), lr=lr)
        self.is_smac = any(name in map_name for name in ["protoss", "terran", "zerg"])
        self.gamma = gamma
        self.optims.extend([self.ent_coef_optimizer, self.actor_opp_optim, self.critic_opp_optim])

    def update_iq(self, mb_obs, mb_avails, mb_masks):
        obs_opp = torch.Tensor(mb_obs)[..., :self.imitator.ob_dim].to(self.device)
        avails_opp = torch.Tensor(mb_avails).to(self.device)
        masks_opp = torch.Tensor(mb_masks).bool().to(self.device)
        curr_obs_opp = obs_opp[:-1,...]
        next_obs_opp = obs_opp[1:,...]
        curr_masks_opp = masks_opp[:-1,...]
        if self.is_smac:
            masks = avails_opp[...,0] == 0.0
            masks = torch.bitwise_and(masks[:-1,...], masks[1:,...])
            curr_obs_opp = curr_obs_opp[masks]
            next_obs_opp = next_obs_opp[masks]
            curr_masks_opp = curr_masks_opp[masks]
        action_exp = next_obs_opp[...,self.imitator.s_dim:self.imitator.e_dim]
        curr_obs_opp = curr_obs_opp.reshape(-1, curr_obs_opp.shape[-1])
        next_obs_opp = next_obs_opp.reshape(-1, next_obs_opp.shape[-1])
        curr_masks_opp = curr_masks_opp.reshape(-1, curr_masks_opp.shape[-1])
        action_exp = action_exp.reshape(-1, action_exp.shape[-1])
        ent_coef = torch.exp(self.log_ent_coef.detach())
        try:
            self.critic_opp_optim.zero_grad()
            critic_loss = self.imitator.update_iq_learn(curr_obs_opp, next_obs_opp, action_exp, curr_masks_opp, ent_coef, self.gamma)
            self.critic_opp_optim.step()
        except:
            print("Error update_iq_learn!")
            critic_loss = torch.tensor(0.0)
        try:
            self.actor_opp_optim.zero_grad()
            actor_loss, log_prob = self.imitator.update_actor(curr_obs_opp, ent_coef)
            self.actor_opp_optim.step()
        except:
            print("Error update_actor!")
            actor_loss = torch.tensor(0.0)
        try:
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        except:
            print("Error ent_coef_loss!")
            ent_coef_loss = torch.tensor(0.0)
        return actor_loss, critic_loss, ent_coef_loss

    def train(self, buffer: Buffer, *args, **kwargs):
        super().train(buffer, *args, **kwargs)
        actor_loss, critic_loss, ent_coef_loss = self.update_iq(buffer.obs, buffer.avails, buffer.masks)
        self.writer.add_scalar(f"losses/opp_actor_loss", actor_loss.item(), self.total_steps)
        self.writer.add_scalar(f"losses/opp_critic_loss", critic_loss.item(), self.total_steps)
        self.writer.add_scalar(f"losses/opp_ent_coef_loss", ent_coef_loss.item(), self.total_steps)
        self.polyak_update(self.imitator.opp_critic.parameters(), self.imitator.opp_critic_target.parameters())

    @torch.no_grad()
    def polyak_update(self, params, target_params, tau=0.005):
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
    
    @torch.no_grad()
    def eval(self, envs, n_episodes=32):
        if envs is None:
            return
        _flatter = lambda x: np.array(np.split(x.detach().cpu().numpy(), envs.n_envs))
        _converter = lambda x: convert_to_tensor(np.concatenate(x), device=self.device)
        logs = []
        obs, _, avails = envs.reset()
        rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
        masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
        self.policy.eval()
        self.imitator.eval()
        while len(logs) < n_episodes:
            _obs, _rnn_actor, _masks, _avails = map(_converter, (obs, rnn_actor, masks, avails))
            actions_opp = self.imitator.get_actions(_obs, deterministic=False)
            _obs = torch.cat((_obs, actions_opp), -1)
            actions, rnn_actor = self.policy.act(_obs, _rnn_actor, _masks, _avails, deterministic=True)
            actions, rnn_actor = map(_flatter, (actions, rnn_actor))
            obs, _, avails, _, dones, infos = envs.step(actions)
            if any("abnormal" in info for info in infos):
                print("Environment error!")
                rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
                masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
                obs, _, avails = envs.reset()
                continue
            logs.extend([info for info in infos if "go_count" in info])
            dones = np.all(dones, axis=1)
            rnn_actor[dones] = np.zeros((dones.sum(), envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
            masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros((dones.sum(), envs.n_agents, 1), dtype=np.float32)
        dead_allies     = np.mean([info.get("dead_allies", -1) for info in logs])
        dead_enemies    = np.mean([info.get("dead_enemies", -1) for info in logs])
        go_count        = np.mean([info["go_count"] for info in logs])
        winrate         = np.mean([info["won"] for info in logs])
        self.writer.add_scalar(f"game_eval/winrate", winrate, self.total_steps)
        if dead_allies > 0:
            self.writer.add_scalar(f"game_eval/dead_allies", dead_allies, self.total_steps)
        if dead_enemies > 0:
            self.writer.add_scalar(f"game_eval/dead_enemies", dead_enemies, self.total_steps)
        self.writer.add_scalar(f"losses/go_count_eval", go_count, self.total_steps)
        print(f"| Map {self.map_name:.^20} | Evaluate: {winrate:.2f}/{dead_allies:.2f}/{dead_enemies:.2f} |")
        return winrate


class MAPILTrainerSUP(MATrainer):

    def __init__(self, policy, imitator: MAPILPolicySup, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, gamma, **kwargs):
        super().__init__(policy, map_name, n_agents, n_steps, log_dir, lr, eps, weight_decay, cliprange, noptepochs, nminibatches, vf_coef, ent_coef, max_grad_norm, device, use_imitation, **kwargs)
        self.imitator = imitator.to(device)
        self.actor_opp_optim = torch.optim.Adam(imitator.opp_actor.parameters(), lr=1e-2)
        self.is_smac = any(name in map_name for name in ["protoss", "terran", "zerg"])
        self.gamma = gamma
        self.optims.append(self.actor_opp_optim)

    def update_sup(self, mb_obs, mb_avails, mb_masks):
        obs_opp = torch.Tensor(mb_obs)[..., :self.imitator.ob_dim].to(self.device)
        avails_opp = torch.Tensor(mb_avails).to(self.device)
        masks_opp = torch.Tensor(mb_masks).bool().to(self.device)
        curr_obs_opp = obs_opp[:-1,...]
        next_obs_opp = obs_opp[1:,...]
        curr_masks_opp = masks_opp[:-1,...]
        if self.is_smac:
            masks = avails_opp[...,0] == 0.0
            masks = torch.bitwise_and(masks[:-1,...], masks[1:,...])
            curr_obs_opp = curr_obs_opp[masks]
            next_obs_opp = next_obs_opp[masks]
            curr_masks_opp = curr_masks_opp[masks]
        action_exp = next_obs_opp[...,self.imitator.s_dim:self.imitator.e_dim]
        curr_obs_opp = curr_obs_opp.reshape(-1, curr_obs_opp.shape[-1])
        next_obs_opp = next_obs_opp.reshape(-1, next_obs_opp.shape[-1])
        curr_masks_opp = curr_masks_opp.reshape(-1, curr_masks_opp.shape[-1])
        action_exp = action_exp.reshape(-1, action_exp.shape[-1])
        
        self.actor_opp_optim.zero_grad()
        actor_loss = self.imitator.update_actor(curr_obs_opp, action_exp)
        self.actor_opp_optim.step()
        
        return actor_loss

    def train(self, buffer: Buffer, *args, **kwargs):
        super().train(buffer, *args, **kwargs)
        actor_loss = self.update_sup(buffer.obs, buffer.avails, buffer.masks)
        self.writer.add_scalar(f"losses/opp_actor_loss", actor_loss.item(), self.total_steps)
    
    @torch.no_grad()
    def eval(self, envs, n_episodes=32):
        if envs is None:
            return
        _flatter = lambda x: np.array(np.split(x.detach().cpu().numpy(), envs.n_envs))
        _converter = lambda x: convert_to_tensor(np.concatenate(x), device=self.device)
        logs = []
        obs, _, avails = envs.reset()
        rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
        masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
        self.policy.eval()
        self.imitator.eval()
        while len(logs) < n_episodes:
            _obs, _rnn_actor, _masks, _avails = map(_converter, (obs, rnn_actor, masks, avails))
            actions_opp = self.imitator.get_actions(_obs, deterministic=False)
            _obs = torch.cat((_obs, actions_opp), -1)
            actions, rnn_actor = self.policy.act(_obs, _rnn_actor, _masks, _avails, deterministic=True)
            actions, rnn_actor = map(_flatter, (actions, rnn_actor))
            obs, _, avails, _, dones, infos = envs.step(actions)
            if any("abnormal" in info for info in infos):
                print("Environment error!")
                rnn_actor = np.zeros((envs.n_envs, envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
                masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
                obs, _, avails = envs.reset()
                continue
            logs.extend([info for info in infos if "go_count" in info])
            dones = np.all(dones, axis=1)
            rnn_actor[dones] = np.zeros((dones.sum(), envs.n_agents, 1, self.policy.actor.h_dim), dtype=np.float32)
            masks = np.ones((envs.n_envs, envs.n_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros((dones.sum(), envs.n_agents, 1), dtype=np.float32)
        dead_allies     = np.mean([info.get("dead_allies", -1) for info in logs])
        dead_enemies    = np.mean([info.get("dead_enemies", -1) for info in logs])
        go_count        = np.mean([info["go_count"] for info in logs])
        winrate         = np.mean([info["won"] for info in logs])
        self.writer.add_scalar(f"game_eval/winrate", winrate, self.total_steps)
        if dead_allies > 0:
            self.writer.add_scalar(f"game_eval/dead_allies", dead_allies, self.total_steps)
        if dead_enemies > 0:
            self.writer.add_scalar(f"game_eval/dead_enemies", dead_enemies, self.total_steps)
        self.writer.add_scalar(f"losses/go_count_eval", go_count, self.total_steps)
        print(f"| Map {self.map_name:.^20} | Evaluate: {winrate:.2f}/{dead_allies:.2f}/{dead_enemies:.2f} |")
        return winrate