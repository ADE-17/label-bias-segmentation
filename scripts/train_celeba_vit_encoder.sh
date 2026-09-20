#!/bin/bash
#SBATCH --job-name=celeba_vit_mitig
#SBATCH --output=/path/to/output/logs/celeba_vit_mitig-%A_%a.out
#SBATCH --error=/path/to/output/logs/celeba_vit_mitig-%A_%a.err
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

# Get fold from SLURM array
FOLD=$SLURM_ARRAY_TASK_ID
echo "Running fold $FOLD"

EPOCHS=15
BIAS=0.50
ENCODER="mit_b2"
MODEL="unet"

echo "========================================="
echo "Running Bias Ratio: $BIAS with ViT ($ENCODER)"
echo "========================================="

# 1. ERM / Standard Biased Baseline (ViT)
# python train_biased.py \
#     --dataset celebamask \
#     --bias_mode erosion \
#     --bias_ratio $BIAS \
#     --erosion_radius 15 \
#     --model $MODEL \
#     --encoder $ENCODER \
#     --epochs $EPOCHS \
#     --fold $FOLD \
#     --exp_name celeba_erm_vit_r${BIAS}

# 2. Proposed: Asymmetric Loss (ViT)
# python train_debias_asym.py \
#     --dataset celebamask \
#     --method asymmetric \
#     --bias_ratio $BIAS \
#     --erosion_radius 15 \
#     --model $MODEL \
#     --encoder $ENCODER \
#     --epochs $EPOCHS \
#     --fold $FOLD \
#     --exp_name celeba_asym_vit_r${BIAS}

# 3. Proposed: Style-Conditioned FiLM (ViT)
python train_debias_asym.py \
    --dataset celebamask \
    --method style_cond \
    --bias_ratio $BIAS \
    --erosion_radius 15 \
    --model $MODEL \
    --encoder $ENCODER \
    --epochs $EPOCHS \
    --fold $FOLD \
    --exp_name celeba_style_vit_r${BIAS}

# 4. Proposed: Asym + Style (ViT)
python train_debias_asym.py \
    --dataset celebamask \
    --method asym_style \
    --bias_ratio $BIAS \
    --erosion_radius 15 \
    --model $MODEL \
    --encoder $ENCODER \
    --epochs $EPOCHS \
    --fold $FOLD \
    --exp_name celeba_asym_style_vit_r${BIAS}

echo "Fold $FOLD completed at $(date +%F-%R:%S)"
