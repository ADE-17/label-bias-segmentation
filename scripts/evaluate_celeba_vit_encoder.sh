#!/bin/bash

# Make sure you have the environment sourced!
# source /path/to/venv/bin/activate
# cd "$(dirname "$0")/.."

OUTPUT_DIR="/path/to/output/evaluations_celeba_vit_50"

EXPERIMENTS="celeba_erm_vit_r0.50 celeba_asym_vit_r0.50 celeba_style_vit_r0.50 celeba_asym_style_vit_r0.50"

echo "Starting evaluation of ViT 50% Bias experiments..."

python -m labelbias.evaluation.evaluate \
    --dataset celebamask \
    --experiments_dir /path/to/output/experiments \
    --experiments $EXPERIMENTS \
    --output_dir $OUTPUT_DIR

echo "Evaluation complete! Summary saved to $OUTPUT_DIR/all_experiments_summary.csv"
