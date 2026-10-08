#!/bin/bash
#SBATCH --job-name=train_fin
#SBATCH --output=/path/to/output/logs/train_fin-%A_%a.out
#SBATCH --error=/path/to/output/logs/train_fin-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

# ==========================================
# 1. Missing from CelebA 100% Bias
# ==========================================
DATASET="celebamask"
BIAS=1.0
EROSION=15

echo "--- CelebA 100% Bias (Noise/Ours) ---"

# Bootstrapping
python -m labelbias.train.erm --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --loss_mode bootstrapping_dice --epochs $EPOCHS --fold $FOLD --exp_name celeba_bootstrapped_r1.0

# Asym mask
python -m labelbias.train.scd --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --method asymmetric --epochs $EPOCHS --fold $FOLD --exp_name celeba_asym_r1.0

# Style-cond
python -m labelbias.train.scd --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --method style_cond --epochs $EPOCHS --fold $FOLD --exp_name celeba_style_r1.0

# Asym+Style
python -m labelbias.train.scd --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --method asym_style --epochs $EPOCHS --fold $FOLD --exp_name celeba_asym_style_r1.0

# Auto-film (Hybrid MoE)
python -m labelbias.train.auto_scd --dataset $DATASET --bias_ratio $BIAS --erosion_radius $EROSION --encoder resnet34 --hybrid_mode auto_film --warmup_epochs 5 --epochs $EPOCHS --fold $FOLD --exp_name celeba_hybrid_moe_r1.0


# ==========================================
# 2. PhC-U373 100% Bias (Fixing ViT -> ResNet)
# ==========================================
DATASET="phc"
BIAS=1.0

echo "--- PhC-U373 100% Bias (ResNet Fix) ---"

# Auto-film (Hybrid MoE) using ResNet34 instead of mit_b2
python -m labelbias.train.auto_scd --dataset $DATASET --bias_ratio $BIAS --encoder resnet34 --hybrid_mode auto_film --warmup_epochs 5 --epochs $EPOCHS --fold $FOLD --exp_name phc_hybrid_moe_r1.0

echo "Fold $FOLD completed at $(date +%F-%R:%S)"
