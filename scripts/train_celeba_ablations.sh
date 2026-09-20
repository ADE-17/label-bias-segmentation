#!/bin/bash
#SBATCH --job-name=celeba_ablations
#SBATCH --output=/path/to/output/logs/celeba_ablations-%A_%a.out
#SBATCH --error=/path/to/output/logs/celeba_ablations-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --export=ALL    
#SBATCH --array=0-4

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source /path/to/venv/bin/activate
export PYTHONHASHSEED=42

cd "$(dirname "$0")/.."   

FOLD=$SLURM_ARRAY_TASK_ID
echo "Running fold $FOLD"

EPOCHS=15

# Ablation 1: 0% Bias (Train style_cond model for 0% bias condition test)
python train_debias_asym.py --dataset celebamask --method style_cond --bias_ratio 0.00 --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_ablation_style_r0.00

# Ablation 2: 50% Bias (Train style_cond model for 50% bias condition test)
python train_debias_asym.py --dataset celebamask --method style_cond --bias_ratio 0.50 --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_ablation_style_r0.50

# Ablation 3: 50% Bias, Low Quality Model (ResNet18)
python train_biased.py --dataset celebamask --bias_mode erosion --bias_ratio 0.50 --erosion_radius 15 --encoder resnet18 --epochs $EPOCHS --fold $FOLD --exp_name celeba_ablation_erm_resnet18_r0.50

echo "Fold $FOLD completed at $(date +%F-%R:%S)"
