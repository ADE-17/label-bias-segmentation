#!/bin/bash
#SBATCH --job-name=eval_imapp
#SBATCH --output=/path/to/output/logs/eval_imapp_%j.out
#SBATCH --error=/path/to/output/logs/eval_imapp_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source /path/to/venv/bin/activate
export PYTHONHASHSEED=42
cd "$(dirname "$0")/.."

python evaluate_imapp.py \
    --experiments_dir /path/to/output/experiments \
    --experiments \
        imapp_baseline_r0 \
        imapp_fairness_dp_r0 \
        imapp_fairness_eo_r0 \
        imapp_fairness_both_r0 \
        imapp_adversarial_r0 \
        imapp_mmd_r0 \
        imapp_coral_r0 \
        imapp_gce_r0 \
        imapp_bootstrapping_r0 \
        imapp_hybrid_moe_r0 \
        imapp_style_r0 \
        imapp_asym_r0 \
        imapp_asym_style_r0 \
    --output_dir /path/to/output/evaluations_imapp \
    --dataset imapp \
    --data_root configs/imapp_processed.csv \
    --img_size 256
