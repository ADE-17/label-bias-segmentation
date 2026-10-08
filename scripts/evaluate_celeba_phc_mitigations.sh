#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=/path/to/output/logs/eval_all-%j.out
#SBATCH --error=/path/to/output/logs/eval_all-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

source /path/to/venv/bin/activate
cd "$(dirname "$0")/.."

echo "Starting Evaluation Group 1: CelebA 50% Bias"
python -m labelbias.evaluation.evaluate --dataset celebamask \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_celeba_mitig_50 \
    --experiments debias_fairness_eo_l0.1_bias_female_r50_e15 \
                  debias_fairness_dp_l0.1_bias_female_r50_e15 \
                  debias_fairness_both_l0.1_bias_female_r50_e15 \
                  debias_adversarial_l0.01_bias_female_r50_e15 \
                  debias_domain_invariant_mmd_logit_l0.01_bias_female_r50_e15 \
                  debias_domain_invariant_coral_l0.1_bias_female_r50_e15 \
                  gce_baseline_r50_e15 \
                  bootstrapped_baseline_r50_e15 \
                  debias_asymmetric_bw2_bias_female_r50_e15 \
                  debias_style_cond_bias_female_r50_e15 \
                  debias_asym_style_bw2_bias_female_r50_e15 \
                  debias_hybrid_auto_film_bias_female_r50_e15

echo "Starting Evaluation Group 2: CelebA 100% Bias"
python -m labelbias.evaluation.evaluate --dataset celebamask \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_celeba_mitig_100 \
    --experiments debias_fairness_eo_l0.1_bias_female_r100_e15 \
                  debias_fairness_dp_l0.1_bias_female_r100_e15 \
                  debias_fairness_both_l0.1_bias_female_r100_e15 \
                  debias_adversarial_l0.1_bias_female_r100_e15 \
                  debias_domain_invariant_mmd_logit_l0.01_bias_female_r100_e15 \
                  debias_domain_invariant_coral_l0.01_bias_female_r100_e15 \
                  gce_baseline_r100_e15 \
                  bootstrapped_baseline_r100_e15 \
                  debias_asymmetric_bw2_bias_female_r100_e15 \
                  debias_style_cond_bias_female_r100_e15 \
                  debias_asym_style_bw2_bias_female_r100_e15 \
                  debias_hybrid_auto_film_bias_female_r100_e15

echo "Starting Evaluation Group 3: PhC-U373 100% Bias"
python -m labelbias.evaluation.evaluate --dataset phc_u373 \
    --experiments_dir /path/to/output/experiments \
    --output_dir /path/to/output/evaluations_phc_mitig_100 \
    --experiments debias_fairness_eo_l0.1_phc_biased_r100 \
                  debias_fairness_dp_l0.1_phc_biased_r100 \
                  debias_fairness_both_l0.1_phc_biased_r100 \
                  debias_adversarial_l0.1_phc_biased_r100 \
                  debias_domain_invariant_mmd_logit_l0.1_phc_biased_r100 \
                  debias_domain_invariant_coral_l0.1_phc_biased_r100 \
                  gce_phc_biased_r100 \
                  bootstrapped_phc_biased_r100 \
                  debias_asymmetric_bw2_phc_biased_r100 \
                  debias_style_cond_phc_biased_r100 \
                  debias_asym_style_bw2_phc_biased_r100 \
                  debias_hybrid_auto_film_phc_biased_r100

echo "All evaluations complete!"
