#!/bin/bash
#SBATCH --job-name=eval_rem
#SBATCH --output=/path/to/output/logs/eval_rem-%j.out
#SBATCH --error=/path/to/output/logs/eval_rem-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

source /path/to/venv/bin/activate
cd "$(dirname "$0")/.."

echo "Starting Evaluation: CelebA 50% Bias (Remaining)"
python evaluate_experiments.py --dataset celebamask \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_celeba_mitig_50 \
    --experiments celeba_hybrid_moe_r0.50

echo "Starting Evaluation: CelebA 100% Bias (Remaining)"
python evaluate_experiments.py --dataset celebamask \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_celeba_mitig_100 \
    --experiments celeba_coral_r1.0 \
                  celeba_gce_r1.0 \
                  celeba_bootstrapped_r1.0 \
                  celeba_asym_r1.0 \
                  celeba_style_r1.0 \
                  celeba_asym_style_r1.0 \
                  celeba_hybrid_moe_r1.0

echo "Starting Evaluation: PhC-U373 100% Bias (All)"
python evaluate_experiments.py --dataset phc \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_phc_mitig_100 \
    --experiments phc_eo_r1.0 \
                  phc_dp_r1.0 \
                  phc_both_r1.0 \
                  phc_adv_r1.0 \
                  phc_mmd_r1.0 \
                  phc_coral_r1.0 \
                  phc_gce_r1.0 \
                  phc_bootstrapped_r1.0 \
                  phc_asym_r1.0 \
                  phc_style_r1.0 \
                  phc_asym_style_r1.0 \
                  phc_hybrid_moe_r1.0

echo "All Evaluations Completed!"
