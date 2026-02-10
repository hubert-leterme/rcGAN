#!/bin/bash

#SBATCH --job-name=valtest_mmgan
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4g.20gb
#SBATCH --time=8:00:00
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
conda activate mmgan_python310
wandb login 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

cd /feynman/home/dap/lcs/hl285110/Documents/Code/rcGAN

#Remember to change exp-name to the batch you want to validate
srun python -u ./scripts/mass_map/validate.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output 
srun python -u ./scripts/mass_map/test.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
srun python -u ./scripts/mass_map/plot.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output --num-figs 10
srun python -u ./scripts/mass_map/cosmos_plot.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
srun python -u ./mass_map_utils/scripts/metrics.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output