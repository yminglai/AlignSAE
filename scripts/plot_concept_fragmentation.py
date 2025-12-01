#!/usr/bin/env python3
"""
Visualize concept fragmentation metrics comparison between AlignSAE and Traditional SAE.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    input_file = Path('results/sae_comparison/concept_fragmentation_summary.csv')
    output_dir = Path('results/sae_comparison')
    
    df = pd.read_csv(input_file)
    
    # ACL-style formatting
    rc = {
        'font.size': 18,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 26,
        'axes.linewidth': 1.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
    
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        layers = df['layer'].values
        
        # Plot 1: Effective Features (lower is better)
        ax = axes[0]
        ax.plot(layers, df['align_eff_feat'], marker='o', linewidth=2.5, 
                markersize=7, label='AlignSAE', color='#2ca02c')
        ax.plot(layers, df['trad_eff_feat'], marker='s', linewidth=2.5, 
                markersize=7, label='Traditional SAE', color='#d62728')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Effective Number of Features')
        ax.set_title('Concept Fragmentation: EffFeat(c)')
        ax.set_xticks(layers)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.6)
        ax.legend(frameon=True)
        ax.text(0.05, 0.95, 'Lower is better\n(≈1 = single feature)', 
                transform=ax.transAxes, fontsize=14, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot 2: Top-1 Concentration (higher is better)
        ax = axes[1]
        ax.plot(layers, df['align_top1_conc'], marker='o', linewidth=2.5,
                markersize=7, label='AlignSAE', color='#2ca02c')
        ax.plot(layers, df['trad_top1_conc'], marker='s', linewidth=2.5,
                markersize=7, label='Traditional SAE', color='#d62728')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Top-1 Concentration')
        ax.set_title('Concept Concentration: Top1Conc(c)')
        ax.set_xticks(layers)
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.6)
        ax.legend(frameon=True)
        ax.text(0.05, 0.95, 'Higher is better\n(≈1 = concentrated)', 
                transform=ax.transAxes, fontsize=14, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Concept-to-Feature Fragmentation Analysis', y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'concept_fragmentation_comparison.png', dpi=400, bbox_inches='tight')
        fig.savefig(output_dir / 'concept_fragmentation_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")
    print("  - concept_fragmentation_comparison.png")
    print("  - concept_fragmentation_comparison.pdf")


if __name__ == '__main__':
    main()
