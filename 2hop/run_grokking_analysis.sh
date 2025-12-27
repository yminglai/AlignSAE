#!/bin/bash

# Complete grokking analysis pipeline
# 1. Collect activations from all epochs in parallel
# 2. Train SAE for each epoch in parallel  
# 3. Analyze binding emergence across epochs

set -e

echo "=========================================="
echo "2-Hop Grokking Analysis Pipeline"
echo "=========================================="

# Step 1: Collect activations from all epoch checkpoints
echo ""
echo "Step 1: Collecting activations from all epochs..."
bash 2hop/collect_grokking_activations_parallel.sh

if [ $? -ne 0 ]; then
    echo "❌ Activation collection failed!"
    exit 1
fi

echo ""
echo "✓ Activation collection complete"
echo ""
read -p "Press Enter to continue to SAE training..."

# Step 2: Train SAEs for all epochs
echo ""
echo "Step 2: Training SAEs for all epochs..."
bash 2hop/train_grokking_saes_parallel.sh

if [ $? -ne 0 ]; then
    echo "❌ SAE training failed!"
    exit 1
fi

echo ""
echo "✓ SAE training complete"
echo ""
read -p "Press Enter to continue to analysis..."

# Step 3: Analyze grokking across all trained SAEs
echo ""
echo "Step 3: Analyzing binding emergence..."
python 2hop/07_analyze_grokking_per_epoch.py \
    --sae_base_dir "2hop/sae_per_epoch" \
    --output_dir "2hop/grokking_analysis_results"

if [ $? -ne 0 ]; then
    echo "❌ Grokking analysis failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to: 2hop/grokking_analysis_results"
echo "  - grokking_curves.png"
echo "  - binding_matrices_evolution.png"
echo "  - grokking_results.json"
echo "=========================================="
