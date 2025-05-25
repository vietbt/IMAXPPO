import numpy as np
import contextlib


MAP_NAMES = {
    "academy": ["3_vs_1_with_keeper", "counterattack_easy", "counterattack_hard"],
    "miner": ["easy_2_vs_2", "medium_2_vs_2", "hard_2_vs_2"],
    "protoss": ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"],
    "terran": ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"],
    "zerg": ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"]
}


@contextlib.contextmanager
def silence_stderr():
    import os
    import sys
    stderr_fd = sys.stderr.fileno()
    orig_fd = os.dup(stderr_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, stderr_fd)
    try:
        yield
    finally:
        os.dup2(orig_fd, stderr_fd)
        os.close(orig_fd)
        os.close(null_fd)


def worker(config, EnvWrapper, seed, remote, parent_remote):
    parent_remote.close()
    with silence_stderr():
        game = EnvWrapper(config, seed)
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(game.step(data))
            elif cmd == 'reset':
                remote.send(game.reset())
            elif cmd == 'get_curr_state':
                state = game.get_current_states()
                remote.send(state)
            elif cmd == 'get_env_info':
                env_info = game.get_env_info()
                remote.send(env_info)
            elif cmd == 'close':
                game.env.close()
                remote.close()
                break
            else:
                print("Invalid command sent by remote")
                break


def read_map_info(map_name):
    map_type, params = map_name.lower().split("_", 1)
    if map_type in ["protoss", "terran", "zerg"]:
        n_agents, _, n_enemy = params.split("_")
        import yaml
        from envs.smac_env.smac_env import SMACWrapper
        with open(f"configs/sc2_gen_{map_type}.yaml", "r") as f:
            config = yaml.safe_load(f)["env_args"]
            config["capability_config"]["n_units"] = int(n_agents)
            config["capability_config"]["n_enemies"] = int(n_enemy)
        return config, SMACWrapper
    elif map_type == "academy":
        from envs.grf_env.grf_env import GRFWrapper
        return map_name, GRFWrapper
    elif map_type == "miner":
        from envs.miner_env.miner_env import MinerWrapper
        mode, n_agents, _, n_enemies = params.split("_")
        config = {
            "mode": mode,
            "n_agents": int(n_agents),
            "n_enemies": int(n_enemies),
        }
        return config, MinerWrapper
    else:
        raise


class VectorizedSMAC(object):

    def __init__(self, args, n_envs):
        from multiprocessing import Pipe, Process
        self.waiting = False
        self.closed = False
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        data = enumerate(zip(self.work_remotes, self.remotes))
        config, EnvWrapper = read_map_info(args.map_name)
        self.ps = [Process(target=worker, args=(config, EnvWrapper, args.seed+i, work_remote, remote)) for i, (work_remote, remote) in data]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.ob_dim, self.st_dim, self.ac_dim, self.n_agents, self.n_enemies, self.s_dim, self.e_dim = self.get_env_infos()
        
    def step_async(self, actions):
        for remote, _actions in zip(self.remotes, actions):
            remote.send(('step', _actions))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, states, avails, rewards, dones, infos = zip(*results)
        obs, states, avails, rewards, dones = map(np.stack, (obs, states, avails, rewards, dones))
        infos = [i for info in infos for i in info]
        return obs, states, avails, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def reset_async(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        self.waiting = True

    def reset_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, states, avails = zip(*results)
        obs, states, avails = map(np.stack, (obs, states, avails))
        return obs, states, avails

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def curr_states_async(self):
        for remote in self.remotes:
            remote.send(('get_curr_state', None))
        self.waiting = True
    
    def curr_infos_async(self):
        for remote in self.remotes:
            remote.send(('get_env_info', None))
        self.waiting = True
    
    def curr_infos_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def curr_states_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, states, avails = zip(*results)
        obs, states, avails = map(np.stack, (obs, states, avails))
        return obs, states, avails

    def get_current_states(self):
        self.curr_states_async()
        return self.curr_states_wait()
    
    def get_env_infos(self):
        self.remotes[0].send(('get_env_info', None))
        return self.remotes[0].recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
