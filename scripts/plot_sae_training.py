"""
Plot SAE training history and binding metrics.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_training_history(history_file, output_dir):
    """Plot training curves from history JSON."""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    epochs = [h['epoch'] for h in history]
    stages = [h.get('stage', 1) for h in history]
    
    losses = [h['loss'] for h in history]
    l_recon = [h['L_recon'] for h in history]
    l_sparse = [h['L_sparse'] for h in history]
    l_align = [h['L_align'] for h in history]
    l_indep = [h['L_indep'] for h in history]
    l_ortho = [h.get('L_ortho', 0) for h in history]  # Handle backward compatibility
    l_value = [h['L_value'] for h in history]
    
    slot_accs = [h['slot_acc'] for h in history]
    value_accs = [h['value_acc'] for h in history]
    
    # Plot losses
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss components
    axes[0, 0].plot(epochs, losses, label='Total Loss', linewidth=2)
    axes[0, 0].plot(epochs, l_recon, label='Reconstruction', alpha=0.7)
    axes[0, 0].plot(epochs, l_sparse, label='Sparsity', alpha=0.7)
    axes[0, 0].plot(epochs, l_align, label='Alignment', alpha=0.7)
    axes[0, 0].plot(epochs, l_indep, label='Independence', alpha=0.7)
    axes[0, 0].plot(epochs, l_ortho, label='Orthogonality', alpha=0.7)
    axes[0, 0].plot(epochs, l_value, label='Value Prediction', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Accuracies
    axes[0, 1].plot(epochs, slot_accs, label='Slot Accuracy', color='blue', linewidth=2)
    axes[0, 1].plot(epochs, value_accs, label='Value Accuracy', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracies')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss breakdown by stage
    stage1_mask = np.array(stages) == 1
    stage2_mask = np.array(stages) == 2
    
    axes[1, 0].plot(np.array(epochs)[stage1_mask], np.array(losses)[stage1_mask], 
                    label='Stage 1 (Reconstruction)', color='green', linewidth=2)
    axes[1, 0].plot(np.array(epochs)[stage2_mask], np.array(losses)[stage2_mask], 
                    label='Stage 2 (Full)', color='purple', linewidth=2)
    axes[1, 0].axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Stage Transition')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total Loss')
    axes[1, 0].set_title('Loss by Training Stage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Alignment loss progression
    axes[1, 1].plot(epochs, l_align, label='Alignment Loss', color='orange', linewidth=2)
    axes[1, 1].axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Stage Transition')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Alignment Loss')
    axes[1, 1].set_title('1-to-1 Binding Loss Progression')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training plots to {output_dir / 'training_history.png'}")

def plot_binding_metrics(eval_results_file, output_dir):
    """Plot binding accuracy metrics."""
    with open(eval_results_file, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    # Extract metrics
    train_slot = results['train']['slot_binding_acc']
    train_answer = results['train']['answer_acc']
    ood_slot = results['test_ood']['slot_binding_acc']
    ood_answer = results['test_ood']['answer_acc']
    
    train_per_rule_slot = [results['train']['per_rule'][r]['slot_binding_acc'] for r in rule_names]
    train_per_rule_answer = [results['train']['per_rule'][r]['answer_acc'] for r in rule_names]
    ood_per_rule_slot = [results['test_ood']['per_rule'][r]['slot_binding_acc'] for r in rule_names]
    ood_per_rule_answer = [results['test_ood']['per_rule'][r]['answer_acc'] for r in rule_names]
    
    # Plot overall binding accuracy
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall comparison
    splits = ['Train', 'Test-OOD']
    slot_accs = [train_slot, ood_slot]
    answer_accs = [train_answer, ood_answer]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0].bar(x - width/2, slot_accs, width, label='Slot Binding Acc', alpha=0.8, color='blue')
    axes[0].bar(x + width/2, answer_accs, width, label='Answer Acc', alpha=0.8, color='red')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Overall Binding Accuracy: Question → Relation → Answer')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])
    axes[0].axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Per-rule comparison
    x_rules = np.arange(len(rule_names))
    
    axes[1].bar(x_rules - width/2, train_per_rule_slot, width, label='Train Slot', alpha=0.8, color='lightblue')
    axes[1].bar(x_rules + width/2, ood_per_rule_slot, width, label='OOD Slot', alpha=0.8, color='darkblue')
    axes[1].set_ylabel('Slot Binding Accuracy')
    axes[1].set_title('Per-Rule Slot Binding Accuracy')
    axes[1].set_xticks(x_rules)
    axes[1].set_xticklabels(rule_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1.0])
    axes[1].axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binding_accuracy_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved binding plots to {output_dir / 'binding_accuracy_plots.png'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_file', type=str, default='models/sae_large/training_history.json')
    parser.add_argument('--eval_results_file', type=str, default='results/sae_eval/binding_accuracy_results.json')
    parser.add_argument('--output_dir', type=str, default='results/sae_eval')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(args.history_file).exists():
        plot_training_history(args.history_file, output_dir)
    else:
        print(f"Training history file not found: {args.history_file}")
    
    if Path(args.eval_results_file).exists():
        plot_binding_metrics(args.eval_results_file, output_dir)
    else:
        print(f"Evaluation results file not found: {args.eval_results_file}")

if __name__ == "__main__":
    main()