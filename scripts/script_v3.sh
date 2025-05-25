#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 100GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --partition=researchlong
#SBATCH --qos=researchqostien
#SBATCH --job-name=IMAX
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40

module load CUDA/11.8.0
module load libGLU

srun python -u run.py zerg