#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem 100GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=researchqostien
#SBATCH --job-name=IMAPPO_protoss_20_vs_23,terran_20_vs_23,zerg_20_vs_23
#SBATCH --gres=gpu:a40:1

module load CUDA/11.8.0
module load libGLU

srun python -u run.py protoss_20_vs_23,terran_20_vs_23,zerg_20_vs_23 --use-tensorboard --use-imitation --sc2-path=/common/home/users/t/tvbui/StarCraftII/