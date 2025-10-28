#!/bin/bash
"""
SAE Training and Evaluation Pipeline
Trains SAE models for each transformer layer and evaluates their performance.
"""

# Configuration
BASE_DIR="/home/mlyang721/bio_data"
LM_MODEL="${BASE_DIR}/models/base_sft/checkpoint-step-10000"
OUTPUT_BASE="${BASE_DIR}/models/sae_per_layer"
RESULTS_BASE="${BASE_DIR}/results/sae_per_layer"

# Training parameters (same as user's command)
N_FREE=10000
EPOCHS_STAGE1=1
EPOCHS_STAGE2=1
LAMBDA_ORTHO="1e-2"
CUDA_DEVICE=1

# Layers to train (0-11)
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11)

# Start with layer 6 for testing
# LAYERS=(6)

echo "=== SAE Training and Evaluation Pipeline ==="
echo "Training layers: ${LAYERS[*]}"
echo "Base LM model: $LM_MODEL"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$RESULTS_BASE"

# Function to train SAE for a specific layer
train_sae_layer() {
    local layer=$1
    local activation_file="${BASE_DIR}/data/activations/train_activations_layer${layer}.pkl"
    local output_dir="${OUTPUT_BASE}/layer${layer}"

    echo "=== Training SAE for Layer $layer ==="
    echo "Activation file: $activation_file"
    echo "Output directory: $output_dir"
    echo ""

    # Create output directory
    mkdir -p "$output_dir"

    # Train SAE
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${BASE_DIR}/scripts/04_train_sae.py" \
        --activation_file "$activation_file" \
        --output_dir "$output_dir" \
        --mode joint \
        --n_free $N_FREE \
        --epochs_stage1 $EPOCHS_STAGE1 \
        --epochs_stage2 $EPOCHS_STAGE2 \
        --lambda_ortho $LAMBDA_ORTHO

    echo "Training completed for layer $layer"
    echo ""
}

# Function to evaluate SAE for a specific layer
evaluate_sae_layer() {
    local layer=$1
    local sae_checkpoint="${OUTPUT_BASE}/layer${layer}/sae_final.pt"
    local activation_file="${BASE_DIR}/data/activations/train_activations_layer${layer}.pkl"
    local results_dir="${RESULTS_BASE}/layer${layer}"

    echo "=== Evaluating SAE for Layer $layer ==="
    echo "SAE checkpoint: $sae_checkpoint"
    echo "Results directory: $results_dir"
    echo ""

    # Create results directory
    mkdir -p "$results_dir"

    # Evaluate SAE
    python "${BASE_DIR}/scripts/05_evaluate_sae.py" \
        --sae_checkpoint "$sae_checkpoint" \
        --lm_model "$LM_MODEL" \
        --activation_file "$activation_file" \
        --layer $layer \
        --output_dir "$results_dir"

    echo "Evaluation completed for layer $layer"
    echo ""
}

# Main pipeline
echo "Starting SAE training and evaluation pipeline..."
echo ""

# Train and evaluate each layer
for layer in "${LAYERS[@]}"; do
    echo "Processing layer $layer..."

    # Train SAE
    train_sae_layer $layer

    # Evaluate SAE
    evaluate_sae_layer $layer

    echo "Completed layer $layer"
    echo "=========================================="
    echo ""
done

echo "=== Pipeline Summary ==="
echo "Trained and evaluated SAEs for layers: ${LAYERS[*]}"
echo "Results saved to: $RESULTS_BASE"
echo ""

# Generate summary report
echo "=== Generating Summary Report ==="
SUMMARY_FILE="${RESULTS_BASE}/pipeline_summary.txt"
echo "SAE Per-Layer Training and Evaluation Summary" > "$SUMMARY_FILE"
echo "Generated on: $(date)" >> "$SUMMARY_FILE"
echo "Layers trained: ${LAYERS[*]}" >> "$SUMMARY_FILE"
echo "LM Model: $LM_MODEL" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for layer in "${LAYERS[@]}"; do
    results_file="${RESULTS_BASE}/layer${layer}/binding_accuracy_results.json"
    if [ -f "$results_file" ]; then
        echo "Layer $layer:" >> "$SUMMARY_FILE"
        # Extract key metrics using jq if available, otherwise use grep
        if command -v jq &> /dev/null; then
            jq -r ".train.slot_binding_acc, .test_ood.slot_binding_acc, .diagonal_accuracy, .swap_controllability.best_success_rate" "$results_file" | \
            awk 'NR==1 {train_acc=$0} NR==2 {test_acc=$0} NR==3 {diag_acc=$0} NR==4 {swap_acc=$0} END {printf "  Train Slot Acc: %.3f\n  Test-OOD Slot Acc: %.3f\n  Diagonal Acc: %.3f\n  Swap Success: %.3f\n\n", train_acc, test_acc, diag_acc, swap_acc}' >> "$SUMMARY_FILE"
        else
            echo "  (Install jq for detailed metrics)" >> "$SUMMARY_FILE"
        fi
    else
        echo "Layer $layer: No results file found" >> "$SUMMARY_FILE"
    fi
done

echo "Summary report saved to: $SUMMARY_FILE"
echo ""
echo "Pipeline completed successfully!"</content>
<parameter name="filePath">/home/mlyang721/bio_data/sae_pipeline.sh