import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--map-name", type=str, default="protoss_5_vs_5")
parser.add_argument("--n-envs", type=int, default=8)
parser.add_argument("--n-eval-envs", type=int, default=8)
parser.add_argument("--n-steps", type=int, default=1024)
parser.add_argument("--n-eval-steps", type=int, default=32)
parser.add_argument("--n-timesteps", type=int, default=10e6)
parser.add_argument("--nminibatches", type=int, default=1)
parser.add_argument("--log-dir", type=str, default="logs")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--h-dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0)
parser.add_argument("--noptepochs", type=int, default=5)
parser.add_argument("--cliprange", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--gain", type=float, default=0.01)
parser.add_argument("--max-grad-norm", type=float, default=10.0)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--use-imax", action="store_true", default=False)
parser.add_argument("--use-gail", action="store_true", default=False)
parser.add_argument("--use-rnn", action="store_true", default=False)
parser.add_argument("--sc2-path", type=str, default="/mnt/d/StarCraftII/")

args = parser.parse_args()
os.environ["SC2PATH"] = args.sc2_path
args.log_dir = f"{args.log_dir}/{args.map_name}"
if args.use_gail:
    args.log_dir = f"{args.log_dir}_gail"
elif args.use_imax:
    args.log_dir = f"{args.log_dir}_imax"
else:
    args.log_dir = f"{args.log_dir}_sup"

args.use_imitation = args.use_gail or args.use_imax