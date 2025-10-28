#!/bin/bash
# Launch SAE Pipeline in Parallel Tmux Sessions
# Distributes layer training across multiple GPUs for parallel processing.

BASE_DIR="/home/mlyang721/bio_data"
SCRIPT="${BASE_DIR}/sae_pipeline.py"

# GPU configuration
GPUS=(1 2 3)
LAYERS_PER_GPU=4

# Calculate layer ranges for each GPU
get_layer_range() {
    local gpu_index=$1
    local start_layer=$((gpu_index * LAYERS_PER_GPU))
    local end_layer=$((start_layer + LAYERS_PER_GPU - 1))

    # Cap at layer 11 (12 layers total: 0-11)
    if [ $end_layer -gt 11 ]; then
        end_layer=11
    fi

    echo "${start_layer} ${end_layer}"
}

echo "=== Launching SAE Pipeline in Parallel ==="
echo "Base directory: $BASE_DIR"
echo "GPUs: ${GPUS[*]}"
echo "Layers per GPU: $LAYERS_PER_GPU"
echo ""

# Launch pipeline for each GPU in a tmux session
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    read start_layer end_layer <<< $(get_layer_range $i)

    session_name="sae_gpu${gpu}"

    echo "Launching GPU $gpu (layers $start_layer-$end_layer) in tmux session: $session_name"

    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null

    # Create new tmux session with first layer
    tmux new-session -d -s "$session_name" -n "layer${start_layer}"
    tmux send-keys -t "$session_name:0" "cd $BASE_DIR && python $SCRIPT --cuda_device $gpu --start_layer $start_layer --end_layer $start_layer" C-m

    # Create panes for remaining layers
    pane_index=1
    for ((layer=start_layer+1; layer<=end_layer; layer++)); do
        # Split window and run the next layer
        tmux split-window -v -t "$session_name:0.$((pane_index-1))"
        tmux send-keys -t "$session_name:0.$pane_index" "cd $BASE_DIR && python $SCRIPT --cuda_device $gpu --start_layer $layer --end_layer $layer" C-m
        pane_index=$((pane_index+1))
    done

    # Set the layout to tiled for better viewing
    tmux select-layout -t "$session_name" tiled

    num_panes=$((end_layer-start_layer+1))
    echo "✓ Started tmux session: $session_name with $num_panes panes"
    echo "  Attach with: tmux attach -t $session_name"
    echo "  To navigate panes: Ctrl-b + arrow keys"
    echo ""
done

echo "=== All Sessions Started ==="
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null || echo "No tmux sessions found"
echo ""
echo "To monitor a specific GPU:"
echo "  tmux attach -t sae_gpu1  # For GPU 1"
echo "  tmux attach -t sae_gpu2  # For GPU 2"
echo "  tmux attach -t sae_gpu3  # For GPU 3"
echo ""
echo "To detach from tmux: Ctrl-b + d"
echo "To kill a session: tmux kill-session -t sae_gpu1"
