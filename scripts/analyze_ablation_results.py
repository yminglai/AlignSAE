#!/usr/bin/env python3
"""
Analyze and visualize ablation study results.

Generates:
1. Summary table comparing all configs
2. Loss component importance plot
3. Free-bank width scaling plot
4. Comprehensive JSON/CSV outputs
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_config_results(config_dir, layer):
    """Load all metrics for a configuration."""
    layer_dir = config_dir / f"layer{layer}"
    
    results = {
        'config_name': config_dir.name,
        'exists': False,
        'train_slot_acc': None,
        'test_ood_slot_acc': None,
        'diagonal_acc': None,
        'swap_success': None,
        'best_alpha': None,
        'swap_success_at_alpha_2': None,
        'reconstruction_mse': None,
    }
    
    # Load binding accuracy results
    binding_file = layer_dir / 'binding_accuracy_results.json'
    if binding_file.exists():
        with open(binding_file) as f:
            data = json.load(f)
        results['exists'] = True
        results['train_slot_acc'] = data.get('train', {}).get('slot_binding_acc')
        results['test_ood_slot_acc'] = data.get('test_ood', {}).get('slot_binding_acc')
        results['diagonal_acc'] = data.get('diagonal_accuracy')
        results['reconstruction_mse'] = data.get('reconstruction_mse')
    
    # Load swap controllability from extended evaluation
    swap_file = layer_dir / 'swap_controllability_all_alphas.json'
    if swap_file.exists():
        with open(swap_file) as f:
            swap_data = json.load(f)
        
        # Find best alpha and its success rate
        best_alpha = None
        best_success = 0.0
        alpha_2_success = None
        
        for alpha_key, alpha_result in swap_data.items():
            if not alpha_key.startswith('alpha_'):
                continue
            
            alpha_val = float(alpha_key.replace('alpha_', ''))
            success_rate = alpha_result.get('success_rate', 0.0)
            
            # Track alpha=2.0 specifically (reviewer's reference point)
            if abs(alpha_val - 2.0) < 0.01:
                alpha_2_success = success_rate
            
            # Track overall best
            if success_rate > best_success:
                best_success = success_rate
                best_alpha = alpha_val
        
        results['best_alpha'] = best_alpha
        results['swap_success'] = best_success
        results['swap_success_at_alpha_2'] = alpha_2_success
    
    return results

def extract_config_type(config_name):
    """Determine if config is loss ablation or width ablation."""
    if config_name.startswith('width_'):
        return 'width', int(config_name.split('_')[1])
    elif config_name == 'baseline':
        return 'loss', 0
    elif config_name.startswith('no_'):
        return 'loss', config_name
    else:
        return 'unknown', config_name

def plot_loss_ablations(loss_results, output_dir):
    """Plot impact of removing each loss component."""
    
    # Order: baseline first, then ablations
    config_order = ['baseline', 'no_sparse', 'no_align', 'no_indep', 'no_ortho', 'no_value']
    config_labels = ['Baseline\n(All)', 'No Sparse\n(λ_s=0)', 'No Align\n(λ_a=0)', 
                     'No Indep\n(λ_i=0)', 'No Ortho\n(λ_⊥=0)', 'No Value\n(λ_v=0)']
    
    # Filter and order results
    ordered_results = []
    for config in config_order:
        if config in loss_results:
            ordered_results.append(loss_results[config])
    
    if not ordered_results:
        print("No loss ablation results found")
        return
    
    # Extract metrics - use alpha=2.0 as reference (reviewer's benchmark)
    metrics = {
        'Train Slot Acc': [r['train_slot_acc'] for r in ordered_results],
        'Test OOD Slot Acc': [r['test_ood_slot_acc'] for r in ordered_results],
        'Diagonal Acc': [r['diagonal_acc'] for r in ordered_results],
        'Swap Success (α=2)': [r['swap_success_at_alpha_2'] for r in ordered_results],
    }
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss Component Ablation Study - Impact on Performance', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(ordered_results))]
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Handle None values
        valid_values = [v if v is not None else 0 for v in values]
        
        bars = ax.bar(range(len(valid_values)), valid_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=0, ha='center')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, valid_values)):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Highlight degradation
        if len(valid_values) > 1 and valid_values[0] is not None:
            baseline_val = valid_values[0]
            for i in range(1, len(valid_values)):
                if valid_values[i] is not None:
                    drop = baseline_val - valid_values[i]
                    if drop > 0.05:  # Significant drop
                        ax.text(i, valid_values[i]/2, f'↓{drop:.2f}', 
                               ha='center', va='center', fontsize=10, 
                               fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ablation_loss_components.png")

def plot_width_scaling(width_results, output_dir):
    """Plot impact of free-bank width."""
    
    # Sort by width
    sorted_configs = sorted(width_results.items(), key=lambda x: x[1]['width'])
    widths = [item[1]['width'] for item in sorted_configs]
    
    metrics = {
        'Train Slot Acc': [item[1]['train_slot_acc'] for item in sorted_configs],
        'Test OOD Slot Acc': [item[1]['test_ood_slot_acc'] for item in sorted_configs],
        'Diagonal Acc': [item[1]['diagonal_acc'] for item in sorted_configs],
        'Swap Success (α=2)': [item[1]['swap_success_at_alpha_2'] for item in sorted_configs],
    }
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Free-Bank Width Ablation - Scaling Analysis', 
                 fontsize=16, fontweight='bold')
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Handle None values
        valid_pairs = [(w, v) for w, v in zip(widths, values) if v is not None]
        if not valid_pairs:
            continue
        
        valid_widths, valid_values = zip(*valid_pairs)
        
        ax.plot(valid_widths, valid_values, 'o-', linewidth=2, markersize=10, 
               color='#3498db', markerfacecolor='#3498db', markeredgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Free-Bank Width (n_free)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add value labels
        for w, v in zip(valid_widths, valid_values):
            ax.text(w, v + 0.03, f'{v:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Annotate knee point if exists
        if len(valid_values) >= 3:
            # Find where performance plateaus (difference < 0.02)
            for i in range(len(valid_values) - 1):
                if valid_values[i+1] - valid_values[i] < 0.02:
                    ax.axvline(x=valid_widths[i], color='red', linestyle='--', 
                             alpha=0.5, linewidth=2)
                    ax.text(valid_widths[i], 0.5, f'Plateau at\n{valid_widths[i]}', 
                           ha='center', fontsize=9, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_width_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ablation_width_scaling.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--layer', type=int, default=6)
    args = parser.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Analyzing Ablation Results")
    print("="*70)
    print(f"Directory: {ablation_dir}")
    print(f"Layer: {args.layer}")
    print()
    
    # Collect all results
    all_results = []
    loss_results = {}
    width_results = {}
    
    for config_dir in sorted(ablation_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue
        
        print(f"Loading: {config_dir.name}")
        results = load_config_results(config_dir, args.layer)
        
        if results['exists']:
            all_results.append(results)
            
            config_type, config_value = extract_config_type(config_dir.name)
            if config_type == 'loss':
                loss_results[config_dir.name] = results
            elif config_type == 'width':
                results['width'] = config_value
                width_results[config_dir.name] = results
    
    print(f"\nFound {len(all_results)} completed configurations")
    print(f"  - Loss ablations: {len(loss_results)}")
    print(f"  - Width ablations: {len(width_results)}")
    print()
    
    # Generate summary table
    print("="*70)
    print("Summary Table")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    df = df.sort_values('config_name')
    
    # Format for display
    display_cols = ['config_name', 'train_slot_acc', 'test_ood_slot_acc', 
                   'diagonal_acc', 'swap_success', 'swap_success_at_alpha_2', 
                   'best_alpha', 'reconstruction_mse']
    df_display = df[display_cols].copy()
    
    for col in ['train_slot_acc', 'test_ood_slot_acc', 'diagonal_acc', 
                'swap_success', 'swap_success_at_alpha_2']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    df_display['reconstruction_mse'] = df_display['reconstruction_mse'].apply(
        lambda x: f"{x:.6f}" if x is not None else "N/A")
    
    print(df_display.to_string(index=False))
    print()
    
    # Save to CSV
    csv_path = output_dir / 'ablation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save to JSON
    json_path = output_dir / 'ablation_summary.json'
    with open(json_path, 'w') as f:
        json.dump({
            'layer': args.layer,
            'loss_ablations': loss_results,
            'width_ablations': width_results,
            'all_configs': all_results
        }, f, indent=2)
    print(f"Saved: {json_path}")
    print()
    
    # Generate plots
    print("="*70)
    print("Generating Visualizations")
    print("="*70)
    
    if loss_results:
        plot_loss_ablations(loss_results, output_dir)
    else:
        print("  No loss ablation results to plot")
    
    if width_results:
        plot_width_scaling(width_results, output_dir)
    else:
        print("  No width ablation results to plot")
    
    print()
    print("="*70)
    print("✓ Analysis Complete!")
    print("="*70)
    print(f"Results: {output_dir}/")

if __name__ == '__main__':
    main()
