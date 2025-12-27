"""
Step 5: Merge and Plot Swap Intervention Results
Combines parallel evaluation results and generates visualization
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


RELATIONS = [
    'accuses', 'admires', 'blames', 'boss_of', 'classmate_of',
    'competes_with', 'cousin_of', 'endorsed_by', 'follows', 'forgives',
    'friend_of', 'has_crush_on', 'mentor_of', 'neighbor_of', 'owes_debt_to',
    'protects', 'reports_to', 'subscribes_to', 'warns', 'works_with'
]


def load_all_results(input_dir):
    """Load all alpha_X.json files from directory (handles nested structure)"""
    results = []
    input_path = Path(input_dir)
    
    # Check both flat and nested structure
    json_files = list(input_path.glob('alpha_*.json'))
    if not json_files:
        # Try nested structure
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                nested_files = list(subdir.glob('alpha_*.json'))
                json_files.extend(nested_files)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    results.sort(key=lambda x: x['alpha'])
    return results


def plot_swap_results(results, output_dir):
    """Generate summary plots for swap intervention"""
    alphas = [r['alpha'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]
    
    # Find peak performance
    peak_idx = np.argmax(success_rates)
    peak_alpha = alphas[peak_idx]
    peak_rate = success_rates[peak_idx]
    
    # Get baseline (alpha=0)
    baseline_rate = success_rates[0] if alphas[0] == 0.0 else 0.0
    improvement = peak_rate - baseline_rate
    relative_improvement = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Success rate vs alpha
    axes[0].plot(alphas, success_rates, marker='o', linewidth=2, markersize=6)
    axes[0].axhline(y=baseline_rate, color='red', linestyle='--', alpha=0.5,
                    label=f'Baseline (α=0): {baseline_rate:.1f}%')
    axes[0].axvline(x=peak_alpha, color='green', linestyle='--', alpha=0.5)
    axes[0].scatter([peak_alpha], [peak_rate], color='green', s=200, zorder=5,
                    label=f'Peak (α={peak_alpha}): {peak_rate:.1f}%')
    
    axes[0].set_xlabel('Alpha (Intervention Strength)', fontsize=12)
    axes[0].set_ylabel('Swap Success Rate (%)', fontsize=12)
    axes[0].set_title(f'Relation Swap Intervention\n'
                     f'Improvement: +{improvement:.1f}% ({relative_improvement:.1f}% increase)',
                     fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('symlog')
    
    # Plot 2: Per-relation breakdown at peak alpha
    peak_result = results[peak_idx]
    by_relation = peak_result.get('by_relation', {})
    
    if by_relation:
        relations_sorted = sorted(by_relation.items(),
                                 key=lambda x: x[1].get('success_rate', 0),
                                 reverse=True)
        rel_names = [r[0] for r in relations_sorted]
        rel_rates = [r[1].get('success_rate', 0) * 100 for r in relations_sorted]
        
        y_pos = np.arange(len(rel_names))
        bars = axes[1].barh(y_pos, rel_rates, color='steelblue', alpha=0.7)
        axes[1].axvline(x=peak_rate, color='red', linestyle='--', alpha=0.5,
                       label=f'Overall: {peak_rate:.1f}%')
        
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(rel_names, fontsize=9)
        axes[1].set_xlabel('Success Rate (%)', fontsize=12)
        axes[1].set_title(f'Per-Relation Success Rate (α={peak_alpha})', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, axis='x', alpha=0.3)
        
        for i, (bar, rate) in enumerate(zip(bars, rel_rates)):
            if rate > 5:
                axes[1].text(rate - 1, i, f'{rate:.1f}%',
                           va='center', ha='right', fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'swap_intervention_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()
    
    return {
        'peak_alpha': peak_alpha,
        'peak_rate': peak_rate,
        'baseline_rate': baseline_rate,
        'improvement': improvement,
        'relative_improvement': relative_improvement
    }


def generate_summary(results, stats, output_file):
    """Generate summary JSON with all results"""
    summary = {
        'statistics': stats,
        'all_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge and plot swap results")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_all_results(args.input_dir)
    print(f"Found {len(results)} alpha evaluations")
    
    print("Generating plots...")
    stats = plot_swap_results(results, args.input_dir)
    
    print("Creating summary...")
    generate_summary(results, stats, args.output_file)
    
    print(f"\n{'='*60}")
    print(f"SWAP INTERVENTION SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (α=0):     {stats['baseline_rate']:.1f}%")
    print(f"Peak (α={stats['peak_alpha']}):      {stats['peak_rate']:.1f}%")
    print(f"Improvement:        +{stats['improvement']:.1f}%")
    print(f"Relative increase:  +{stats['relative_improvement']:.1f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
