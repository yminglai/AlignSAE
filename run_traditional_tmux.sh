#!/bin/bash
# Train and Analyze Traditional SAE in Tmux (12 parallel windows)

SESSION_NAME="traditional_sae"

# Configuration
N_HIDDEN=10000
EPOCHS=420  # Match your supervised SAE (20 + 400)
BATCH_SIZE=256
LR=0.001
LAMBDA_SPARSE=0.001
SAMPLES_PER_RELATION=500

echo "=========================================================="
echo "Training Traditional SAE in Tmux (12 layers in parallel)"
echo "=========================================================="
echo ""
echo "Configuration:"
echo "  Hidden features: $N_HIDDEN"
echo "  Epochs: $EPOCHS (matching supervised SAE: 20+400)"
echo "  Session name: $SESSION_NAME"
echo ""

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session with first window for layer 0
tmux new-session -d -s $SESSION_NAME -n "layer0"

# Create windows for layers 1-11
for layer in {1..11}; do
    tmux new-window -t $SESSION_NAME -n "layer${layer}"
done

# Function to send commands to a window
send_to_window() {
    local layer=$1
    local window="layer${layer}"
    
    # Training command
    tmux send-keys -t "$SESSION_NAME:$window" "
echo '=========================================='
echo 'Training Traditional SAE - Layer $layer'
echo '=========================================='
echo ''

python scripts/04_train_traditional_sae.py \\
    --activation_file data/activations/train_activations_layer${layer}.pkl \\
    --output_dir models/traditional_sae/layer${layer} \\
    --n_hidden $N_HIDDEN \\
    --epochs $EPOCHS \\
    --batch_size $BATCH_SIZE \\
    --lr $LR \\
    --lambda_sparse $LAMBDA_SPARSE \\
    --device cuda

echo ''
echo '✓ Training complete for layer $layer'
echo ''
echo '=========================================='
echo 'Analyzing Traditional SAE - Layer $layer'
echo '=========================================='
echo ''

python scripts/analyze_feature_coactivation.py \\
    --sae_dir models/traditional_sae \\
    --lm_model models/base_sft/final \\
    --qa_file data/generated/qa_test_id.jsonl \\
    --output_dir results/traditional_sae_analysis \\
    --layers $layer \\
    --samples_per_relation $SAMPLES_PER_RELATION \\
    --top_k_features 50 \\
    --activation_threshold 0.1

echo ''
echo '✓✓ Layer $layer COMPLETE (trained + analyzed) ✓✓'
echo ''
" C-m
}

# Send commands to all windows
echo "Setting up tmux windows..."
for layer in {0..11}; do
    send_to_window $layer
    echo "  ✓ Layer $layer window configured"
done

echo ""
echo "=========================================================="
echo "All layers running in tmux!"
echo "=========================================================="
echo ""
echo "To view progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Switch between layers:"
echo "  Ctrl+b, then number 0-9 for layers 0-9"
echo "  Ctrl+b, then 'n' for next window"
echo "  Ctrl+b, then 'p' for previous window"
echo "  Ctrl+b, then 'w' to see all windows"
echo ""
echo "Detach (leave running):"
echo "  Ctrl+b, then 'd'"
echo ""
echo "Kill all:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "Expected runtime: ~10-15 minutes per layer"
echo "All 12 layers in parallel: ~15-20 minutes total"
echo ""
