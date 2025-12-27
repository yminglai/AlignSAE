#!/bin/bash

# Parallel Swap Intervention for Layer 6
# Tests multiple alpha values in parallel using independent GPU processes

LM_MODEL="2hop/_trained_model_4k_mixed/checkpoint_epoch_100"
SAE_CHECKPOINT="2hop/sae_large/sae_best.pt"
VAL_DATA="2hop/_dataset/_gen/val_two_hop_qa_data_4k.jsonl"
FACTS_DB="2hop/facts_database/entity_relations.json"
LAYER=6
MAX_NEW_TOKENS=1
OUTPUT_DIR="2hop/swap_results/layer6_1token"

# Alpha values to test
ALPHAS=(0.0 1.0 2.0 5.0 10.0 15.0 20.0 50.0 100.0 200.0 500.0 1000.0)

echo "=========================================="
echo "Swap Intervention - Parallel Execution"
echo "=========================================="
echo "Layer: $LAYER"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output directory: $OUTPUT_DIR"
echo "Number of alpha values: ${#ALPHAS[@]}"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch all alpha evaluations in parallel
pids=()
for alpha in "${ALPHAS[@]}"; do
    echo "Launching alpha=$alpha in background..."
    python 2hop/04_swap_intervention.py \
        --alpha "$alpha" \
        --lm_model "$LM_MODEL" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --val_data "$VAL_DATA" \
        --facts_db "$FACTS_DB" \
        --layer "$LAYER" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda &
    
    pids+=($!)
done

echo ""
echo "All processes launched. Waiting for completion..."
echo "PIDs: ${pids[@]}"

# Wait for all background processes
for pid in "${pids[@]}"; do
    wait "$pid"
    echo "Process $pid completed"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo "Merging results..."

# Merge and plot results
python 2hop/05_merge_and_plot_swap.py \
    --input_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/swap_summary.json"

echo ""
echo "✓ Done! Results saved to $OUTPUT_DIR"
echo "✓ Summary plot: $OUTPUT_DIR/swap_intervention_summary.png"
