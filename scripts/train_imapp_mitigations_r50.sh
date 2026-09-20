#!/bin/bash
#SBATCH --job-name=imapp_mitigations_50
#SBATCH --output=/path/to/output/logs/imapp_mitigations_50_%A_%a.out
#SBATCH --error=/path/to/output/logs/imapp_mitigations_50_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4

# We are parallelizing across the 5 folds.
FOLD=$SLURM_ARRAY_TASK_ID

source /path/to/venv/bin/activate
export PYTHONHASHSEED=42
cd "$(dirname "$0")/.."

DATASET="imapp"
BIAS_RATIO="0.5"
EPOCHS=15
BS=32

echo "Starting training for FOLD $FOLD on $DATASET with bias $BIAS_RATIO"

# 1. ERM Baseline
python train_imapp_biased.py \
    --csv_path configs/imapp_processed.csv \
    --epochs $EPOCHS \
    --batch_size $BS \
    --bias_ratio $BIAS_RATIO \
    --output_dir /path/to/output \
    --exp_name imapp_baseline_r50 \
    --fold $FOLD

# 2. Fairness DP
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method fairness --fairness_mode dp \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_fairness_dp_r50 \
    --fold $FOLD

# 3. Fairness EO
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method fairness --fairness_mode eo \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_fairness_eo_r50 \
    --fold $FOLD

# 4. Fairness Both
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method fairness --fairness_mode both \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_fairness_both_r50 \
    --fold $FOLD

# 5. Adversarial
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method adversarial \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_adversarial_r50 \
    --fold $FOLD

# 6. MMD
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method domain_invariant --domain_method mmd_logit \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_mmd_r50 \
    --fold $FOLD

# 7. CORAL
python train_debiased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method domain_invariant --domain_method coral \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_coral_r50 \
    --fold $FOLD

# 8. GCE
python train_biased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --loss_mode gce_dice \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_gce_r50 \
    --fold $FOLD

# 9. Bootstrapping
python train_biased.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --loss_mode bootstrapping_dice \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_bootstrapping_r50 \
    --fold $FOLD

# 10. Auto-conditioned
python train_hybrid_moe.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --encoder resnet34 \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_hybrid_moe_r50 \
    --fold $FOLD

# 11. Style Conditioned
python train_debias_asym.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method style_cond \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_style_r50 \
    --fold $FOLD

# 12. Asymmetric Mask Conditioned
python train_debias_asym.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method asymmetric \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_asym_r50 \
    --fold $FOLD

# 13. Asymmetric + Style Conditioned
python train_debias_asym.py \
    --dataset $DATASET --bias_ratio $BIAS_RATIO \
    --method asym_style \
    --epochs $EPOCHS --batch_size $BS \
    --exp_name imapp_asym_style_r50 \
    --fold $FOLD

echo "Finished FOLD $FOLD for all IMA++ mitigations."
