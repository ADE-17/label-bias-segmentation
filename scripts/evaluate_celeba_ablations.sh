#!/bin/bash

# Output directories for ablations
OUT_CLEAN="/path/to/output/evaluations_ablation_force_clean_style"
OUT_BIASED="/path/to/output/evaluations_ablation_force_biased_style"
OUT_RESNET18="/path/to/output/evaluations_ablation_resnet18"

echo "=========================================================="
echo "Ablation 1 & 2: Conditioning on Clean vs Biased Styles"
echo "=========================================================="

# Force style = 0 (clean style)
echo "Evaluating with clean style (force_group_id=0)..."
python -m labelbias.evaluation.evaluate \
    --dataset celebamask \
    --experiments celeba_ablation_style_r0.00 celeba_ablation_style_r0.50 \
    --force_group_id 0 \
    --output_dir $OUT_CLEAN

# Force style = 1 (biased style)
echo "Evaluating with biased style (force_group_id=1)..."
python -m labelbias.evaluation.evaluate \
    --dataset celebamask \
    --experiments celeba_ablation_style_r0.00 celeba_ablation_style_r0.50 \
    --force_group_id 1 \
    --output_dir $OUT_BIASED

echo "=========================================================="
echo "Ablation 3: Low Model Quality (ResNet18)"
echo "=========================================================="

echo "Evaluating standard ERM with ResNet18..."
python -m labelbias.evaluation.evaluate \
    --dataset celebamask \
    --experiments celeba_ablation_erm_resnet18_r0.50 \
    --output_dir $OUT_RESNET18

echo "All ablation evaluations complete!"
