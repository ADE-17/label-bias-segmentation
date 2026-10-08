#!/bin/bash
#SBATCH --job-name=celeba_mitigation
#SBATCH --output=/path/to/output/logs/celeba_mitig-%A_%a.out
#SBATCH --error=/path/to/output/logs/celeba_mitig-%A_%a.err
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

# We train for 20 epochs across 4 bias levels
EPOCHS=15

for BIAS in 0.25 0.50 0.75 1.0; do
    echo "========================================="
    echo "Running Bias Ratio: $BIAS"
    echo "========================================="
    
    # 1. Standard Baseline
    python -m labelbias.train.erm --dataset celebamask --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_baseline_r${BIAS}
    
    # 2. Bootstrapping Baseline
    python -m labelbias.train.erm --dataset celebamask --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --loss_mode bootstrapping_dice --epochs $EPOCHS --fold $FOLD --exp_name celeba_bootstrapped_r${BIAS}
    
    # 3. Hybrid MoE (auto_film)
    python -m labelbias.train.auto_scd --dataset celebamask --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --encoder mit_b2 --hybrid_mode auto_film --warmup_epochs 5 --epochs $EPOCHS --fold $FOLD --exp_name celeba_hybrid_moe_r${BIAS}
    
    # 4. MMD
    python -m labelbias.train.invariance --dataset celebamask --method domain_invariant --domain_method mmd_logit --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --lambda_debias 0.01 --mmd_normalize --balance_gender_in_batch --fold $FOLD --exp_name celeba_mmd_r${BIAS}
    
    # 5. CORAL
    python -m labelbias.train.invariance --dataset celebamask --method domain_invariant --domain_method coral --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --lambda_debias 0.01 --fold $FOLD --exp_name celeba_coral_r${BIAS}
    
    # 6. Adversarial (DANN)
    python -m labelbias.train.invariance --dataset celebamask --method adversarial --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --lambda_debias 0.01 --fold $FOLD --exp_name celeba_adv_r${BIAS}
    
    # 7. Fairness EO
    python -m labelbias.train.invariance --dataset celebamask --method fairness --fairness_mode eo --bias_mode erosion --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_eo_r${BIAS}
    
    # 8. Proposed: Asymmetric Loss
    python -m labelbias.train.scd --dataset celebamask --method asymmetric --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_asym_r${BIAS}
    
    # 9. Proposed: Style-Conditioned FiLM
    python -m labelbias.train.scd --dataset celebamask --method style_cond --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_style_r${BIAS}
    
    # 10. Proposed: Asym + Style
    python -m labelbias.train.scd --dataset celebamask --method asym_style --bias_ratio $BIAS --erosion_radius 15 --epochs $EPOCHS --fold $FOLD --exp_name celeba_asym_style_r${BIAS}

done

echo "Fold $FOLD completed at $(date +%F-%R:%S)"
