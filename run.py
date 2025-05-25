import os
import sys
from multiprocessing import Pool


def run(cmd):
    print(cmd)
    os.system(cmd)


def main():
    name = sys.argv[1]
    if name == "20_vs_20":
        cmds = [
            f"python -u main.py --map-name=protoss_20_vs_20 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name=terran_20_vs_20 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name=zerg_20_vs_20 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
        ]
    elif name == "20_vs_23":
        cmds = [
            f"python -u main.py --map-name=protoss_20_vs_23 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name=terran_20_vs_23 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name=zerg_20_vs_23 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
        ]
    elif name == "miner":
        cmds = [
            f"python -u main.py --map-name=miner_easy_2_vs_2 --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-envs=32 --n-timesteps 2000000",
            f"python -u main.py --map-name=miner_medium_2_vs_2 --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-envs=32 --n-timesteps 2000000",
            f"python -u main.py --map-name=miner_hard_2_vs_2 --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-envs=32 --n-timesteps 2000000",
        ]
    elif name == "grf":
        cmds = [
            f"python -u main.py --map-name=academy_3_vs_1_with_keeper --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-timesteps=1000000 --ent-coef=0.0",
            f"python -u main.py --map-name=academy_counterattack_easy --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-timesteps=1000000 --ent-coef=0.0",
            f"python -u main.py --map-name=academy_counterattack_hard --sc2-path=/common/home/users/t/tvbui/StarCraftII/ --n-timesteps=1000000 --ent-coef=0.0",
        ]
    else:
        cmds = [
            f"python -u main.py --map-name={name}_5_vs_5 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name={name}_10_vs_10 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name={name}_10_vs_11 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
            f"python -u main.py --map-name={name}_20_vs_20 --sc2-path=/common/home/users/t/tvbui/StarCraftII/",
        ]
    # cmds = [f"{cmd} --use-gail" for cmd in cmds]
    with Pool(4) as p:
        p.map(run, cmds)


if __name__ == "__main__":
    main()