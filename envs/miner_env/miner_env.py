import math
import random
import signal
import traceback
import numpy as np


def handler(signum, frame):
    raise Exception("Timeout")


class MinerWrapper:

    def __init__(self, config, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.mode = config["mode"]
        self.n_agents = config["n_agents"]
        self.n_enemies = config["n_enemies"]
        self._reset()
    
    def _reset(self):
        from miner.MinerMultiAgentEnv import GameSocket
        self._env = GameSocket(self.n_agents+self.n_enemies)
        self._env.connect()
        self.mapIDs = list(self._env.maps.keys())
        self.reset()
    
    def reset(self, MAP_MAX_X=21, MAP_MAX_Y=9):
        from miner.bots import BlackPantherBot, DeepMindBot, GreedyBot
        mapID = self.mapIDs[np.random.randint(0, len(self.mapIDs))]
        posID_x = np.random.randint(MAP_MAX_X)
        posID_y = np.random.randint(MAP_MAX_Y)
        self._env.reset([mapID, posID_x, posID_y, 50, 100])
        if self.mode == "easy":
            bots = [GreedyBot(self._env), GreedyBot(self._env)]
        elif self.mode == "medium":
            bots = [GreedyBot(self._env), DeepMindBot(self._env)]
        elif self.mode == "hard":
            bots = [BlackPantherBot(self._env), DeepMindBot(self._env)]
        bots = bots * math.ceil(self.n_enemies / len(bots))
        random.shuffle(bots)
        self.bots = bots
        self.scores = {}
        self._env.stepCount = 0
        return self._get_current_states()
    
    def get_env_info(self):
        return self.ob_dim, self.st_dim, self.ac_dim, self.n_agents, self.n_enemies, self.s_dim, self.e_dim

    def fix_state(self, states, agent_id):
        al_ids = [agent_id] + [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        st_allies = [states[id] for id in al_ids]
        states = sum(st_allies, [])
        return states

    def get_avails(self):
        avails = []
        for user in self._env.bots:
            avails.append(self._env.get_avails(user))
        return avails
    
    def render(self):
        print("state:")
        new_data = []
        for data in self._env.map:
            new_data.append([f"{x}" for x in data])
        for user in self._env.bots:
            new_data[user.posy][user.posx] += "*"
        for data in new_data:
            print("|".join([f"{x:^6}" for x in data]))
        
        scores = {}
        for user in self._env.bots:
            scores[f"player_{user.playerId}"] = user.score
        print("scores:", scores)

    def get_state(self):
        max_x = self._env.userMatch.gameinfo.width
        max_y = self._env.userMatch.gameinfo.height
        data = np.zeros([max_x, max_y])
        for cell in self._env.userMatch.gameinfo.obstacles:
            if cell.type > 0:
                data[cell.posx, cell.posy] = - 1 / cell.type
        for cell in self._env.userMatch.gameinfo.golds:
            data[cell.posx, cell.posy] = cell.amount / 1000
        data = data.flatten()
        return data
    
    def get_obs(self, state, agent_id=0):
        agent = self._env.bots[agent_id]
        max_x = self._env.userMatch.gameinfo.width
        max_y = self._env.userMatch.gameinfo.height

        all_pos = [agent.posx/max_x, agent.posy/max_y]
        users = [user for user in self._env.bots if user.playerId != agent_id]
        for user in users:
            if user.playerId < self.n_agents:
                all_pos += [user.posx/max_x, user.posy/max_y]
        for user in users:
            if user.playerId >= self.n_agents:
                all_pos += [user.posx/max_x, user.posy/max_y]
        
        feats = np.concatenate((all_pos, state, [agent.energy/self._env.E, self._env.stepCount/self._env.maxStep]))
        self.s_dim = 2 * self.n_agents
        self.e_dim = len(all_pos)
        return feats

    def _get_current_states(self):
        state = self.get_state()
        self.obs = self.states = np.stack([self.get_obs(state, id) for id in range(self.n_agents)])
        self.avails = np.stack([self._env.get_avails(self._env.bots[id]) for id in range(self.n_agents)])
        self.ob_dim = self.obs.shape[-1]
        self.st_dim = self.states.shape[-1]
        self.ac_dim = self.avails.shape[-1]
        return self.get_current_states()

    def step(self, actions):
        actions = [int(action) for action in actions]
        self._env.stepCount += 1
        enemy_actions = [bot.get_action(self.n_agents + id) for id, bot in enumerate(self.bots)]
        self._env.step(actions + enemy_actions)

        rewards = []
        for agent in self._env.bots:
            score = agent.score - self.scores.get(agent.playerId, 0)
            self.scores[agent.playerId] = agent.score
            rewards.append(score)
        
        done = self._env.stepCount >= self._env.maxStep
        for user in self._env.bots:
            if user.status != 0:
                done = True
                break
        try:
            self._get_current_states()
        except:
            done = True
        my_info = {}
        if done:
            ally_score = np.mean([self.scores[id] for id in range(self.n_agents)])
            enemy_score = np.mean([self.scores[self.n_agents+id] for id in range(self.n_enemies)])
            my_info["go_count"] = self._env.stepCount
            my_info["won"] = ally_score > enemy_score
            self.reset()
        reward = np.sum(rewards[:self.n_agents])
        rewards = 0.01 * np.array([[reward]] * self.n_agents)
        dones = np.array([done] * self.n_agents)
        return self.obs, self.states, self.avails, rewards, dones, [my_info]

    def get_current_states(self):
        return self.obs, self.states, self.avails
