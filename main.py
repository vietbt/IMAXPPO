
def main():
    from configs import args
    from envs.base_env.env import VectorizedSMAC
    envs = VectorizedSMAC(args, args.n_envs)
    args.seed += 1024
    eval_envs = VectorizedSMAC(args, args.n_eval_envs) if args.n_eval_envs > 0 else None
    
    import torch
    import random
    import numpy as np
    import setproctitle
    from utils import create_trainer
    from trainer.runner import SMACRunner

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    setproctitle.setproctitle(args.map_name)

    policy, imitator, trainer = create_trainer(args, envs)
    runner = SMACRunner(envs, policy, imitator, **vars(args))
    # print("policy:", policy)
    
    episode = 0
    while trainer.total_steps < args.n_timesteps:
        buffer, buffer_imitator = runner.run(args.verbose)
        if "miner" in args.map_name or not args.use_imax:
            rate = trainer.total_steps / args.n_timesteps
            trainer.set_learning_rate(np.exp(-5.0 * rate))
        trainer.train(buffer, buffer_imitator)
        if episode % 20 == 0 and "academy" not in args.map_name:
            # torch.jit.save(policy, f"{args.log_dir}/policy.pt")
            # if imitator is not None:
            #     torch.jit.save(imitator, f"{args.log_dir}/imitator.pt")
            # torch.save([x.state_dict() for x in trainer.optims], f"{args.log_dir}/optims.pt")
            trainer.eval(eval_envs, args.n_eval_steps)
        episode += 1
    envs.close()


if __name__ == "__main__":
    main()
