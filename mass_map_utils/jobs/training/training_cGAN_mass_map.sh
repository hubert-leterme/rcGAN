#!/bin/bash

#SBATCH --job-name=training_mmgan
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4g.20gb
#SBATCH --time=50:00:00
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
conda activate mmgan_python310
wandb login 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

cd /feynman/home/dap/lcs/hl285110/Documents/Code/rcGAN

srun python -u train.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
