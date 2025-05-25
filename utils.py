import os
import math
import torch
import numpy as np
from functools import partial
from tensorboard.backend.event_processing import event_accumulator


def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def convert_to_tensor(data, tensor="float", device="cuda", use_fp16=False):
    if data is not None:
        if not torch.is_tensor(data):
            if device == "cpu":
                if tensor == "float":
                    torch_data_func = torch.HalfTensor if use_fp16 else torch.FloatTensor
                else:
                    torch_data_func = torch.BoolTensor
            else:
                if tensor == "float":
                    torch_data_func = torch.cuda.HalfTensor if use_fp16 else torch.cuda.FloatTensor
                else:
                    torch_data_func = torch.cuda.BoolTensor
            data = torch_data_func(data)
        else:
            data = data.to(device)
    return data


def convert_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def policy_step(policy, obs, states, avails, rnn_actor, rnn_critic, dones, device, deterministic=False):
    convert = partial(convert_to_tensor, device=device)
    obs, states, rnn_actor, rnn_critic, dones = map(convert, (obs, states, rnn_actor, rnn_critic, dones))
    avails = convert_to_tensor(avails, "bool", device=device)
    actions, rnn_actor, neglogps, values, rnn_critic = policy.get_actions(obs, states, avails, rnn_actor, rnn_critic, dones, deterministic)
    return map(convert_to_numpy, (actions, rnn_actor, neglogps, values, rnn_critic))


@torch.no_grad()
def policy_get_values(policy, states, rnn_critic, dones, device):
    convert = partial(convert_to_tensor, device=device)
    states, rnn_critic, dones = map(convert, (states, rnn_critic, dones))
    dones = dones.unsqueeze(-1)
    values = policy.get_values(states, rnn_critic, dones)
    return convert_to_numpy(values)


def merge_data(data):
    obs, states, avails, returns, dones, actions, values, neglogpacs, rnn, ep_infos = zip(*data)
    obs, states, avails, returns, dones, actions, values, neglogpacs = map(np.concatenate, (obs, states, avails, returns, dones, actions, values, neglogpacs))
    rnn = np.concatenate(rnn, 1)
    ep_infos = sum(ep_infos, [])
    return obs, states, avails, returns, dones, actions, values, neglogpacs, rnn, ep_infos


def smooth(scalars, weight):
    last = 0
    smoothed = []
    for num_acc, next_val in enumerate(scalars):
        last = last * weight + (1 - weight) * next_val
        smoothed.append(last/(1-math.pow(weight, num_acc+1)))
    return smoothed


def read_tensorboard_events(folder):
    logs = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if "events" in file:
                event = event_accumulator.EventAccumulator(os.path.join(folder, file))
                event.Reload()
                tag = "game_eval/winrate"
                if tag not in event.scalars.Keys():
                    tag = "game/winrate"
                logs += [(x.step, x.value) for x in event.Scalars(tag)]
    logs.sort(key=lambda x: x[0])
    return logs


def create_trainer(args, envs):
    from models.agents.mappo import MAPPOPolicy
    ob_dim, st_dim, ac_dim = envs.ob_dim, envs.st_dim, envs.ac_dim
    n_agents, s_dim, e_dim = envs.n_agents, envs.s_dim, envs.e_dim
    # logs = read_tensorboard_events(args.log_dir)
    # if len(logs) > 0 and False:
    #     policy = torch.jit.load(f"{args.log_dir}/policy.pt")
    #     imitator = torch.jit.load(f"{args.log_dir}/imitator.pt")
    #     trainer = MAPILTrainerIQ(policy, imitator, n_agents=n_agents, **vars(args))
    #     trainer.total_steps = logs[-1][0]
    #     try:
    #         for optim, params in zip(trainer.optims, torch.load(f"{args.log_dir}/optims.pt")):
    #             optim.load_state_dict(params)
    #         print(f"Loading optimizers: {args.log_dir}/optims.pt")
    #     except:
    #         pass
    #     print(f"Loading pre-trained policy: {args.log_dir}/policy.pt at step {trainer.total_steps}.")
    if args.use_gail:
        from models.agents.imaxppo import MAPILPolicyGAIL
        from trainer.trainer import MAPILTrainerGAIL
        imitator = MAPILPolicyGAIL(ob_dim, st_dim, s_dim, e_dim, args.h_dim, use_rnn=False)
        policy = MAPPOPolicy(ob_dim + imitator.ac_dim, st_dim, ac_dim, **vars(args))
        trainer = MAPILTrainerGAIL(policy, imitator, n_agents=n_agents, **vars(args))
    elif args.use_imax:
        from models.agents.imaxppo import MAPILPolicyIQ
        from trainer.trainer import MAPILTrainerIQ
        imitator = MAPILPolicyIQ(ob_dim, s_dim, e_dim, args.h_dim)
        policy = MAPPOPolicy(ob_dim + imitator.ac_dim, st_dim, ac_dim, **vars(args))
        trainer = MAPILTrainerIQ(policy, imitator, n_agents=n_agents, **vars(args))
    
    else:
        from models.agents.imaxppo import MAPILPolicySup
        from trainer.trainer import MAPILTrainerSUP
        imitator = MAPILPolicySup(ob_dim, s_dim, e_dim, args.h_dim)
        policy = MAPPOPolicy(ob_dim + imitator.ac_dim, st_dim, ac_dim, **vars(args))
        trainer = MAPILTrainerSUP(policy, imitator, n_agents=n_agents, **vars(args))
    return policy, imitator, trainer
    
