#!/bin/bash
# Monitor all SAE training sessions

echo "=== SAE Training Monitor ==="
echo ""

# Check tmux sessions
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null | grep sae || echo "No SAE sessions found"
echo ""

# Check GPU processes
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Check if tmux sessions are still running
tmux_active=$(tmux list-sessions 2>/dev/null | grep -c sae_gpu || echo "0")

# Check training progress by looking at checkpoint files
echo "Training Progress:"
all_complete=true
for layer in {0..11}; do
    best_file="/home/mlyang721/bio_data/models/sae_per_layer/layer${layer}/sae_best.pt"
    final_file="/home/mlyang721/bio_data/models/sae_per_layer/layer${layer}/sae_final.pt"
    
    if [ -f "$final_file" ] && [ "$tmux_active" -eq 0 ]; then
        # Training completed for this layer (final checkpoint exists and no tmux sessions)
        epoch_info=$(python3 -c "
import torch
try:
    ckpt = torch.load('$final_file', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    slot_acc = ckpt.get('best_slot_acc', 'unknown')
    print(f'FINAL - Epoch: {epoch}, Best Slot Acc: {slot_acc:.3f}')
except Exception as e:
    print(f'Final checkpoint exists - Epoch: unknown')
" 2>/dev/null)
        echo "Layer $layer: ✅ Training complete - $epoch_info"
    elif [ -f "$best_file" ]; then
        # Training active for this layer
        all_complete=false
        epoch_info=$(python3 -c "
import torch
try:
    ckpt = torch.load('$best_file', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    slot_acc = ckpt.get('best_slot_acc', 'unknown')
    print(f'Epoch: {epoch}, Best Slot Acc: {slot_acc:.3f}')
except Exception as e:
    print('Checkpoint exists - Epoch: unknown')
" 2>/dev/null)
        echo "Layer $layer: ✓ Training active - $epoch_info"
    else
        # Training not started
        all_complete=false
        echo "Layer $layer: ⏳ Starting..."
    fi
done
echo ""

# Check if all training is complete
if [ "$all_complete" = true ]; then
    echo "🎉 ALL TRAINING COMPLETE! Starting evaluation..."
    echo ""
    
    # Check if evaluation has already been done
    eval_complete=true
    for layer in {0..11}; do
        results_file="/home/mlyang721/bio_data/results/sae_per_layer/layer${layer}/binding_accuracy_results.json"
        if [ ! -f "$results_file" ]; then
            eval_complete=false
            break
        fi
    done
    
    if [ "$eval_complete" = true ]; then
        echo "✅ Evaluation also complete! Check results in /home/mlyang721/bio_data/results/sae_per_layer/"
        echo ""
        echo "To view summary:"
        echo "  python sae_pipeline.py --generate_summary"
    else
        echo "🚀 Running evaluation on all layers..."
        # Run evaluation in background
        nohup python /home/mlyang721/bio_data/sae_pipeline.py --evaluate_all > /home/mlyang721/bio_data/evaluation.log 2>&1 &
        echo "Evaluation started in background. Check /home/mlyang721/bio_data/evaluation.log for progress."
    fi
else
    echo "⏳ Training still in progress..."
fi
echo ""

echo "Quick Commands:"
echo "  ./monitor_sae_training.sh    # Show this status"
echo "  tmux attach -t sae_gpu1      # Monitor GPU 1 (layers 0-3)"
echo "  tmux attach -t sae_gpu2      # Monitor GPU 2 (layers 4-7)"
echo "  tmux attach -t sae_gpu3      # Monitor GPU 3 (layers 8-11)"
echo ""
echo "To kill all training:"
echo "  tmux kill-session -t sae_gpu1 sae_gpu2 sae_gpu3"
echo ""

# Show current time
echo "Last updated: $(date)"
