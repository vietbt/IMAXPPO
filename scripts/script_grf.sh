#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem 64GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=research-1-qos
#SBATCH --job-name=IMAPPO_grf
#SBATCH --gres=gpu:1

module load CUDA/11.8.0
module load libGLU

srun python -u run.py academy_3_vs_1_with_keeper,academy_counterattack_easy,academy_counterattack_hard --use-imitation --n-timesteps=1000000 --ent-coef=0.0