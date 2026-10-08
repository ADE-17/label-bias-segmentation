#!/bin/bash
#SBATCH --job-name=train_miss
#SBATCH --output=/path/to/output/logs/train_miss-%A_%a.out
#SBATCH --error=/path/to/output/logs/train_miss-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
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
ENCODER="resnet34"

# ==========================================
# 1. CelebA 50% Bias
# ==========================================
DATASET="celebamask"
BIAS=0.50
EROSION=15

echo "--- CelebA 50% Bias ---"
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode eo --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_eo_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode dp --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_dp_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode both --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_both_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method adversarial --adv_lr 1e-3 --grl_warmup 3 --grl_max_alpha 1.0 --lambda_debias 0.01 --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_adv_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method mmd_logit --lambda_debias 0.01 --mmd_normalize --balance_gender_in_batch --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_mmd_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method coral --lambda_debias 0.01 --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_coral_r${BIAS}
python -m labelbias.train.erm --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --loss_mode gce_dice --epochs $EPOCHS --fold $FOLD --exp_name celeba_gce_r${BIAS}

# ==========================================
# 2. CelebA 100% Bias
# ==========================================
BIAS=1.0

echo "--- CelebA 100% Bias ---"
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode eo --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_eo_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode dp --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_dp_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode both --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_both_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method adversarial --adv_lr 1e-3 --grl_warmup 3 --grl_max_alpha 1.0 --lambda_debias 0.01 --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_adv_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method mmd_logit --lambda_debias 0.01 --mmd_normalize --balance_gender_in_batch --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_mmd_r${BIAS}
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method coral --lambda_debias 0.01 --bias_ratio $BIAS --erosion_radius $EROSION --epochs $EPOCHS --fold $FOLD --exp_name celeba_coral_r${BIAS}
python -m labelbias.train.erm --dataset $DATASET --bias_mode erosion --bias_ratio $BIAS --erosion_radius $EROSION --loss_mode gce_dice --epochs $EPOCHS --fold $FOLD --exp_name celeba_gce_r${BIAS}

# ==========================================
# 3. PhC-U373 100% Bias
# ==========================================
DATASET="phc"
BIAS=1.0

echo "--- PhC-U373 100% Bias ---"
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode eo --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_eo_r1.0
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode dp --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_dp_r1.0
python -m labelbias.train.invariance --dataset $DATASET --method fairness --fairness_mode both --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_both_r1.0
python -m labelbias.train.invariance --dataset $DATASET --method adversarial --adv_lr 1e-3 --grl_warmup 3 --grl_max_alpha 1.0 --lambda_debias 0.01 --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_adv_r1.0
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method mmd_logit --lambda_debias 0.01 --mmd_normalize --balance_gender_in_batch --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_mmd_r1.0
python -m labelbias.train.invariance --dataset $DATASET --method domain_invariant --domain_method coral --lambda_debias 0.01 --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_coral_r1.0
python -m labelbias.train.erm --dataset $DATASET --bias_ratio $BIAS --loss_mode gce_dice --epochs $EPOCHS --fold $FOLD --exp_name phc_gce_r1.0

# Also run Bootstrapping for PhC (since it hasn't been run yet)
python -m labelbias.train.erm --dataset $DATASET --bias_ratio $BIAS --loss_mode bootstrapping_dice --epochs $EPOCHS --fold $FOLD --exp_name phc_bootstrapped_r1.0

# Asym mask for PhC
python -m labelbias.train.scd --dataset $DATASET --method asymmetric --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_asym_r1.0

# Style for PhC
python -m labelbias.train.scd --dataset $DATASET --method style_cond --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_style_r1.0

# Asym+Style for PhC
python -m labelbias.train.scd --dataset $DATASET --method asym_style --bias_ratio $BIAS --epochs $EPOCHS --fold $FOLD --exp_name phc_asym_style_r1.0

# Auto-film (Hybrid MoE) for PhC
python -m labelbias.train.auto_scd --dataset $DATASET --bias_ratio $BIAS --encoder mit_b2 --hybrid_mode auto_film --warmup_epochs 5 --epochs $EPOCHS --fold $FOLD --exp_name phc_hybrid_moe_r1.0

echo "Fold $FOLD completed at $(date +%F-%R:%S)"
