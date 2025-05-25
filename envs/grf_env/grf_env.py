import math
import random
import traceback
import types
import numpy as np


MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
PENALTY_Y, END_Y = 0.27, 0.42
LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, RELEASE_MOVE = 9, 10, 11, 12, 14
RELEASE_SPRINT, SLIDE, DRIBBLE, RELEASE_DRIBBLE = 15, 16, 17, 18


class FeatureEncoder:

    def __init__(self, env):
        self._env = env
        self._sticky_actions = env._sticky_actions
        self._action_set = env._action_set
        self._action_names = [a._name for a in env._action_set]
        self._sticky_names = [a._name for a in env._sticky_actions]
        self.agent_ids = None

    def encode(self, obs):
        if self.agent_ids is None:
            self.agent_ids = [id for id, agent in enumerate(obs.left_team) if agent.position[0] > 0]
        agent_feats = []
        for agent_id, agent in enumerate(obs.left_team):
            agent_feats.append(self.get_feats(agent, obs, agent_id))
        enemy_feats = [self.get_feats(enemy) for enemy in obs.right_team]
        agent_obs = []
        agent_avails = []
        for agent_id in self.agent_ids:
            player_state, ball_state, avail, feats = agent_feats[agent_id]
            pos_x, pos_y = feats[0], feats[1]
            left_team = []
            for ally_id in range(len(obs.left_team)):
                if ally_id == agent_id:
                    continue
                ally_pos_x, ally_pos_y, direc_x, direc_y, speed, tired = agent_feats[ally_id][-1]
                dist = math.hypot(ally_pos_x-pos_x, ally_pos_y-pos_y)
                left_team.append([ally_pos_x*2, ally_pos_y*2, direc_x, direc_y, speed, dist*2, tired])
            left_closest = min(left_team, key=lambda x: x[-2])
            right_team = []
            for enemy_id in range(len(obs.right_team)):
                enemy_pos_x, enemy_pos_y, direc_x, direc_y, speed, tired = enemy_feats[enemy_id][-1]
                dist = math.hypot(enemy_pos_x-pos_x, enemy_pos_y-pos_y)
                right_team.append([enemy_pos_x*2, enemy_pos_y*2, direc_x, direc_y, speed, dist*2, tired])
            right_closest = min(right_team, key=lambda x: x[-2])
            left_team = sum(left_team, [])
            right_team = sum(right_team, [])
            agent_obs.append(left_team + right_team + player_state + ball_state + left_closest + right_closest)
            agent_avails.append(avail)
            self.s_dim = len(left_team)
            self.e_dim = self.s_dim + len(right_team)
        return agent_obs, agent_avails

    def get_feats(self, agent, obs=None, agent_id=None):
        pos_x, pos_y = agent.position[0], agent.position[1]
        direc_x, direc_y = agent.direction[0], agent.direction[1]
        speed = math.hypot(direc_x, direc_y)
        tired = agent.tired_factor
        if obs is not None and agent_id in self.agent_ids:
            role = int(agent.role)
            role_onehot = self._encode_role_onehot(role)
            is_dribbling = self._env._env.sticky_action_state(self._sticky_actions[9]._backend_action, True, agent_id)
            is_sprinting = self._env._env.sticky_action_state(self._sticky_actions[8]._backend_action, True, agent_id)
            is_any_sticky = any(self._env._env.sticky_action_state(a._backend_action, True, agent_id) for a in self._sticky_actions[:8])
            ball_x, ball_y, ball_z = obs.ball_position[0], obs.ball_position[1], obs.ball_position[2]
            ball_x_relative = ball_x - pos_x
            ball_y_relative = ball_y - pos_y
            ball_x_speed, ball_y_speed, ball_z_speed = obs.ball_direction[0], obs.ball_direction[1], obs.ball_direction[2]
            ball_distance = math.hypot(ball_x_relative, ball_y_relative)
            ball_speed = math.hypot(ball_x_speed, ball_y_speed)
            ball_owned = obs.ball_owned_team != -1
            ball_owned_by_us = obs.ball_owned_team == 0
            ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)
            ball_far = 1 if ball_distance > 0.03 else 0
            avail = self._get_avail(obs, ball_distance, is_dribbling, is_sprinting, is_any_sticky)
            player_state = avail[9:] + [pos_x, pos_y, direc_x*100, direc_y*100, speed*100] + role_onehot + [ball_far, tired, is_dribbling, is_sprinting]
            ball_state = [ball_x, ball_y, ball_z] + ball_which_zone + [ball_x_relative, ball_y_relative] + [ball_x_speed*20, ball_y_speed*20, ball_z_speed*20, ball_speed*20, ball_distance, ball_owned, ball_owned_by_us]
        else:
            player_state = ball_state = avail = None
        feats = [pos_x, pos_y, direc_x*100, direc_y*100, speed*100, tired]
        return player_state, ball_state, avail, feats

    def _get_avail(self, obs, ball_distance, is_dribbling, is_sprinting, is_any_sticky):
        avail = [1] * len(self._action_names)
        ball_x, ball_y = obs.ball_position[0], obs.ball_position[1]
        game_mode = int(obs.game_mode)
        if obs.ball_owned_team == 1:
            avail[LONG_PASS] = avail[HIGH_PASS] = avail[SHORT_PASS] = avail[SHOT] = avail[DRIBBLE] = 0
        elif obs.ball_owned_team == -1 and ball_distance > 0.03 and game_mode == 0:  # GR ball and far from me
            avail[LONG_PASS] = avail[HIGH_PASS] = avail[SHORT_PASS] = avail[SHOT] = avail[DRIBBLE] = 0
        else:
            avail[SLIDE] = 0
            if ball_x > 0.85 and (ball_y < -0.34 or ball_y > 0.34):
                avail[LONG_PASS] = avail[SHORT_PASS] = avail[SHOT] = avail[DRIBBLE] = 0
        if not is_sprinting:
            avail[RELEASE_SPRINT] = 0
        if is_dribbling:
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0
        if not is_any_sticky:
            avail[RELEASE_MOVE] = 0
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif 0.64 <= ball_x <= 1.0 and -0.27 <= ball_y <= 0.27:
            avail[HIGH_PASS] = avail[LONG_PASS] = 0
        if game_mode == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1] + [0] * (len(self._action_names) - 1)
            avail[LONG_PASS] = avail[HIGH_PASS] = avail[SHORT_PASS] = 1
        elif game_mode == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1] + [0] * (len(self._action_names) - 1)
            avail[LONG_PASS] = avail[HIGH_PASS] = avail[SHORT_PASS] = 1
        elif game_mode == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1] + [0] * (len(self._action_names) - 1)
            avail[SHOT] = 1
        return avail

    def _encode_ball_which_zone(self, ball_x, ball_y):
        onehot = [0] * 6
        if -END_X <= ball_x < -PENALTY_X and -PENALTY_Y < ball_y < PENALTY_Y:
            id = 0
        elif -END_X <= ball_x < -MIDDLE_X and -END_Y < ball_y < END_Y:
            id = 1
        elif -MIDDLE_X <= ball_x <= MIDDLE_X and -END_Y < ball_y < END_Y:
            id = 2
        elif PENALTY_X < ball_x <= END_X and -PENALTY_Y < ball_y < PENALTY_Y:
            id = 3
        elif MIDDLE_X < ball_x <= END_X and -END_Y < ball_y < END_Y:
            id = 4
        else:
            id = 5
        onehot[id] = 1
        return onehot

    def _encode_role_onehot(self, role_num):
        result = [0] * 10
        result[role_num] = 1
        return result

    def calc_reward(self, ball_x, ball_y, is_win):
        ball_position_r = 0.0
        if -END_X <= ball_x < -PENALTY_X and -PENALTY_Y < ball_y < PENALTY_Y:
            ball_position_r = -2.0
        elif -END_X <= ball_x < -MIDDLE_X and -END_Y < ball_y < END_Y:
            ball_position_r = -1.0
        elif -MIDDLE_X <= ball_x <= MIDDLE_X and -END_Y < ball_y < END_Y:
            ball_position_r = 0.0
        elif PENALTY_X < ball_x <= END_X and -PENALTY_Y < ball_y < PENALTY_Y:
            ball_position_r = 2.0
        elif MIDDLE_X < ball_x <= END_X and -END_Y < ball_y < END_Y:
            ball_position_r = 1.0
        else:
            ball_position_r = 0.0
        win_reward = 5 if is_win else 0
        reward = win_reward + 0.003 * ball_position_r
        return reward


def _retrieve_observation(self):
    info = self._env.get_info()
    if info.is_in_play:
        agent_obs, agent_avails = self.obs_encoder.encode(info)
        self.agent_obs = np.array(agent_obs)
        self.agent_avails = np.array(agent_avails)
        self._observation = {}
        self._observation['left_agent_controlled_player'] = []
        self._observation['right_agent_controlled_player'] = []
        for i in range(self._env.config.left_agents):
            self._observation['left_agent_controlled_player'].append(info.left_controllers[i].controlled_player)
        for i in range(self._env.config.right_agents):
            self._observation['right_agent_controlled_player'].append(info.right_controllers[i].controlled_player)
        self._observation['game_mode'] = int(info.game_mode)
        self._observation['score'] = (info.left_goals, info.right_goals)
        self._observation['ball_owned_team'] = info.ball_owned_team
        self._observation['ball_owned_player'] = info.ball_owned_player
        self._observation['ball'] = (info.ball_position[0], info.ball_position[1])
        self._observation['left_team_x'] = [player.position[0] for player in info.left_team]
        self._step = info.step
    return info.is_in_play


class GRFWrapper:

    def __init__(self, map_name, seed):
        np.bool = bool
        np.random.seed(seed)
        random.seed(seed)
        self.env_name = map_name
        self.seed = seed
        self._reset()
        self.n_enemies = -1
    
    def _reset(self):
        try:
            self.env.close()
        except:
            pass
        import gfootball.env as football_env
        from gfootball.env.config import Config
        self.scenario_config = Config({'level': self.env_name}).ScenarioConfig()
        self.n_players = self.scenario_config.controllable_left_players
        self.agent_ids = None
        self._env = football_env.create_environment(
            env_name=self.env_name,
            number_of_left_players_agent_controls=self.n_players,
            representation="raw",
            other_config_options={
                "action_set": "v2",
                "game_engine_random_seed": 0
            }
        ).env._env
        self._env.obs_encoder = FeatureEncoder(self._env)
        self._env._retrieve_observation = types.MethodType(_retrieve_observation, self._env)
        self.go_count = 0
        self.reset()
        
    def reset(self):
        self._env.reset()
        self.go_count = 0
        return self._get_current_states()

    def get_env_info(self):
        return self.ob_dim, self.st_dim, self.ac_dim, self.n_agents, self.n_enemies, self.s_dim, self.e_dim

    def step(self, action_ids):
        self.go_count += 1
        actions = [19] * self.n_players
        for agent_id, action_id in zip(self.agent_ids, action_ids):
            actions[agent_id] = int(action_id)
        try:
            next_obs, _, done, _ = self._env.step(actions)
            scores = next_obs["score"]
            ball_x, ball_y = next_obs["ball"]
            done = done or scores[0] != scores[1]
            reward = self._env.obs_encoder.calc_reward(ball_x, ball_y, scores[0] > scores[1])

            if "counterattack" in self.env_name:
                reward = [reward] * self.n_agents
                agents_x = next_obs['left_team_x']
                agents_x = [agents_x[agent_id] for agent_id in self.agent_ids]
                done = done or ball_x < 0 or any(pos_x < 0 for pos_x in agents_x)
                for id, agent_id in enumerate(self.agent_ids):
                    pos_x = next_obs['left_team_x'][agent_id]
                    if pos_x < 0:
                        # reward[id] -= 0.1
                        done = True

            my_info = {}
            if done:
                my_info["go_count"] = self.go_count
                my_info["won"] = scores[0] > scores[1]
                self.reset()
            else:
                self._get_current_states()
        except Exception:
            print(traceback.format_exc())
            self.reset()
            reward, done, my_info = 0, False, {"abnormal": True}
        rewards = np.array([[reward]] * self.n_agents if not isinstance(reward, list) else [[r] for r in reward])
        dones = np.array([done] * self.n_agents)
        return self.obs, self.states, self.avails, rewards, dones, [my_info]

    def _get_current_states(self):
        self.agent_ids = self._env.obs_encoder.agent_ids
        self.n_agents = len(self.agent_ids)
        self.obs = self.states = self._env.agent_obs
        self.avails = self._env.agent_avails
        self.ob_dim = self.st_dim = self.obs.shape[-1]
        self.ac_dim = self.avails.shape[-1]
        self.s_dim = self._env.obs_encoder.s_dim
        self.e_dim = self._env.obs_encoder.e_dim
        return self.get_current_states()

    def get_current_states(self):
        return self.obs, self.states, self.avails