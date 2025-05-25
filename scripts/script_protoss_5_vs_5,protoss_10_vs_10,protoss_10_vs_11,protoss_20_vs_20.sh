#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem 100GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=researchqostien
#SBATCH --job-name=IMAPPO_protoss_5_vs_5,protoss_10_vs_10,protoss_10_vs_11,protoss_20_vs_20
#SBATCH --gres=gpu:1

module load CUDA/11.8.0
module load libGLU

srun python -u run.py protoss_5_vs_5,protoss_10_vs_10,protoss_10_vs_11,protoss_20_vs_20 --use-imitation --sc2-path=/common/home/users/t/tvbui/StarCraftII/