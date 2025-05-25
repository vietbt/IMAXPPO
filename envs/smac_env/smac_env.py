import signal
import traceback
import numpy as np


def handler(signum, frame):
    raise Exception("Timeout")


class SMACWrapper:

    def __init__(self, config, seed):
        np.bool = bool
        self.config = config
        self.seed = seed
        self.timeout = 60
        self._reset()
        self.n_agents = self.env.n_agents
        self.n_enemies = self.env.n_enemies
        self.obs, self.states, self.avails = self.reset()
        self.ob_dim = self.obs.shape[-1]
        self.st_dim = self.states.shape[-1]
        self.ac_dim = self.avails.shape[-1]
    
    def _reset(self):
        try:
            self.env.close()
        except:
            pass
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
        self._env = StarCraftCapabilityEnvWrapper(**self.config)
        self.env._seed = self.seed
        self.initialized = False

    @property
    def env(self):
        return self._env.env
    
    def reset(self):
        try:
            if self.initialized:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(self.timeout)
            self._env.reset()
            out = self._get_current_states()
            if self.initialized:
                signal.alarm(0)
        except Exception:
            print(traceback.format_exc())
            self._reset()
            out = self.reset()
        self.initialized = True
        return out

    def get_env_info(self):
        return self.ob_dim, self.st_dim, self.ac_dim, self.n_agents, self.n_enemies, self.s_dim, self.e_dim
    
    def step(self, actions):
        try:
            reward, done, info = self._env.step(actions)
            assert not done or "battle_won" in info
        except Exception as e:
            print("Error!", e)
            reward = None
        if reward is None:
            print("Resetting ...")
            obs, states, avails = self.reset()
            rewards = [[0]] * self.n_agents
            dones = [True] * self.n_agents
            infos = [{"abnormal": True}]
        else:
            obs, states, avails = self._get_current_states()
            self.obs, self.states, self.avails = obs, states, avails
            my_info = {}
            if done:
                my_info["dead_allies"] = info["dead_allies"] / self.n_agents
                my_info["dead_enemies"] = info["dead_enemies"] / self.n_enemies
                my_info["go_count"] = self.env._episode_steps
                my_info["won"] = info["battle_won"]
                obs, states, avails = self.reset()
            rewards = [[reward]] * self.n_agents
            dones = [done] * self.n_agents
            infos = [my_info]
        return obs, states, avails, rewards, dones, infos

    def fix_state(self, states, agent_id):
        nf_al = self.env.get_ally_num_attributes()
        nf_en = self.env.get_enemy_num_attributes()

        ally_dim = self.n_agents * nf_al
        enemy_dim = self.n_enemies * nf_en

        st_allies = states[:ally_dim].reshape(self.n_agents, nf_al)
        st_enemies = states[ally_dim:ally_dim+enemy_dim]    #.reshape(self.n_enemies, nf_en)
        st_actions = states[ally_dim+enemy_dim:]            #.reshape(self.n_agents, self.env.n_actions)

        al_ids = [agent_id] + [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        st_allies = st_allies[al_ids]

        st_allies = st_allies.flatten()
        # st_enemies = st_enemies.flatten()
        # st_actions = st_actions.flatten()

        self.s_dim = len(st_allies)
        self.e_dim = len(st_allies) + len(st_enemies)

        states = np.concatenate((st_allies, st_enemies, st_actions))
        return states
    
    def fix_obs(self, obs):
        return obs
        # move_dim = self.env.get_obs_move_feats_size()
        # enemy_dim = self.env.get_obs_enemy_feats_size()
        # ally_dim = self.env.get_obs_ally_feats_size()
        # own_dim = self.env.get_obs_own_feats_size()

        # _enemy_dim = enemy_dim[0] * enemy_dim[1]
        # _ally_dim = ally_dim[0] * ally_dim[1]
        
        # move_feats, obs = obs[:move_dim], obs[move_dim:]
        # enemy_feats, obs = obs[:_enemy_dim].reshape(enemy_dim), obs[_enemy_dim:]
        # ally_feats, obs = obs[:_ally_dim].reshape(ally_dim), obs[_ally_dim:]
        # own_feats = obs[:own_dim]

        # # enemy_feats = np.pad(enemy_feats, ((0, MAX_ENEMIES-self.n_enemies), (0, 0)), constant_values=PAD_INDEX)
        # # ally_feats = np.pad(ally_feats, ((0, MAX_ALLIES-self.n_agents), (0, 0)), constant_values=PAD_INDEX)

        # feats = [move_feats, enemy_feats, ally_feats, own_feats]
        # obs = np.concatenate([feat.flatten() for feat in feats])
        # return obs
    
    def fix_avails(self, avails):
        return avails
        # ac_dim = self.env.n_actions_no_attack + MAX_ENEMIES
        # avails += [0] * (ac_dim - len(avails))
        # return avails
    
    def _get_current_states(self):
        states = self._env.get_state()
        obs = np.array([self.fix_obs(_obs) for _obs in self._env.get_obs()])
        states = np.array([self.fix_state(states, i) for i in range(self.n_agents)])
        avails = np.array([self.fix_avails(_avails) for _avails in self._env.get_avail_actions()])
        return obs, states, avails

    def get_current_states(self):
        return self.obs, self.states, self.avails
    