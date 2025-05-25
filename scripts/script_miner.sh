#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem 64GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=research-1-qos
#SBATCH --job-name=IMAPPO_miner
#SBATCH --gres=gpu:1

module load CUDA/11.8.0
module load libGLU

srun python -u run.py miner_easy_2_vs_2,miner_medium_2_vs_2,miner_hard_2_vs_2 --use-imitation --use-tensorboard --n-envs 128
