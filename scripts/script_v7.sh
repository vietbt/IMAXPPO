#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 100GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=researchqostien
#SBATCH --job-name=IQMIX
#SBATCH --gres=gpu:1
#SBATCH --constraint=48gb

module load CUDA/11.8.0
module load libGLU

srun python -u run.py grf