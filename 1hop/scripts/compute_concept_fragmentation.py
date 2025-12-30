#!/usr/bin/env python3
"""
Compute concept fragmentation metrics for AlignSAE vs Traditional SAE.

Quantifies "one concept → many features" problem:
- EffFeat(c): effective number of features per concept (entropy-based)
- Top1Conc(c): concentration on the top-1 feature

Generates a comparison table across all 12 layers.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

RELATION_NAMES = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']


def load_activation_matrix(result_dir, layer_idx):
    """Load the [6 concepts × n_features] activation matrix."""
    path = Path(result_dir) / f'layer{layer_idx:02d}_activation_matrix.npy'
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')
    return np.load(path)


def compute_fragmentation_metrics(A):
    """
    Given A[c, k] = avg activation of concept c on feature k,
    compute:
      - EffFeat(c): effective number of features (exp of entropy)
      - Top1Conc(c): max fraction of activation on a single feature
    
    Args:
        A: [n_concepts, n_features]
    Returns:
        dict with keys 'eff_feat' and 'top1_conc', each a list of length n_concepts
    """
    n_concepts = A.shape[0]
    eff_feat = []
    top1_conc = []

    # 截断负数，保证只统计正激活
    A_pos = np.maximum(A, 0)

    for c in range(n_concepts):
        # Normalize over features (only positive activations)
        total = A_pos[c, :].sum() + 1e-12
        B_c = A_pos[c, :] / total

        # Top-1 concentration
        top1 = B_c.max()

        # Effective number of features (exp of entropy)
        B_pos = B_c[B_c > 1e-12]
        entropy = -np.sum(B_pos * np.log(B_pos))
        eff = np.exp(entropy)

        eff_feat.append(eff)
        top1_conc.append(top1)

    return {
        'eff_feat': eff_feat,
        'top1_conc': top1_conc
    }


def main():
    parser = argparse.ArgumentParser(description='Compute concept fragmentation metrics')
    parser.add_argument('--align_dir', type=str, 
                        default='results/1to1_feature_coactivation_analysis',
                        help='Directory with AlignSAE activation matrices')
    parser.add_argument('--trad_dir', type=str,
                        default='results/traditional_sae_analysis',
                        help='Directory with Traditional SAE activation matrices')
    parser.add_argument('--output_dir', type=str,
                        default='results/sae_comparison',
                        help='Output directory for results')
    parser.add_argument('--layers', type=str, default='0-11',
                        help='Layer range (e.g., 0-11) or comma-separated list')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse layer list
    if '-' in args.layers:
        start, end = args.layers.split('-')
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]
    
    # Collect results
    results = []
    
    for layer in layers:
        print(f"Processing layer {layer}...")
        
        # Load AlignSAE matrix
        align_matrix = load_activation_matrix(args.align_dir, layer)
        # AlignSAE: last 6 columns are relation slots, first n_free are free features
        # For concept fragmentation, we focus on the last 6 slots (supervised)
        n_free_align = align_matrix.shape[1] - 6
        align_rel_slots = align_matrix[:, n_free_align:]
        
        # Load Traditional SAE matrix
        trad_matrix = load_activation_matrix(args.trad_dir, layer)
        
        # Compute metrics
        align_metrics = compute_fragmentation_metrics(align_rel_slots)
        trad_metrics = compute_fragmentation_metrics(trad_matrix)
        
        # Aggregate across concepts
        for c_idx, c_name in enumerate(RELATION_NAMES):
            results.append({
                'layer': layer,
                'concept': c_name,
                'align_eff_feat': align_metrics['eff_feat'][c_idx],
                'align_top1_conc': align_metrics['top1_conc'][c_idx],
                'trad_eff_feat': trad_metrics['eff_feat'][c_idx],
                'trad_top1_conc': trad_metrics['top1_conc'][c_idx],
            })
    
    # Build DataFrame
    df = pd.DataFrame(results)
    
    # Compute layer averages
    layer_summary = df.groupby('layer').agg({
        'align_eff_feat': 'mean',
        'align_top1_conc': 'mean',
        'trad_eff_feat': 'mean',
        'trad_top1_conc': 'mean',
    }).reset_index()
    
    # Save detailed results
    df.to_csv(output_dir / 'concept_fragmentation_detailed.csv', index=False)
    layer_summary.to_csv(output_dir / 'concept_fragmentation_summary.csv', index=False)
    
    # Print summary table
    print("\n" + "="*80)
    print("Concept Fragmentation Metrics: AlignSAE vs Traditional SAE")
    print("="*80)
    print("\nLayer-wise Averages (across 6 concepts):\n")
    
    print(f"{'Layer':<6} | {'AlignSAE EffFeat':<10} | {'AlignSAE Top1C':<10} | {'Trad EffFeat':<10} | {'Trad Top1C':<10}")
    print("-" * 60)

    for _, row in layer_summary.iterrows():
        print(f"{int(row['layer']):<6} | "
              f"{row['align_eff_feat']:>10.3f} | "
              f"{row['align_top1_conc']:>10.3f} | "
              f"{row['trad_eff_feat']:>10.3f} | "
              f"{row['trad_top1_conc']:>10.3f}")
    
    print("\n" + "="*80)
    print(f"Results saved to {output_dir}")
    print(f"  - concept_fragmentation_detailed.csv (per concept)")
    print(f"  - concept_fragmentation_summary.csv (per layer)")
    print("="*80)
    
    # Also save as JSON for easy parsing
    summary_dict = layer_summary.to_dict(orient='records')
    with open(output_dir / 'concept_fragmentation_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    print("\nInterpretation:")
    print("  - EffFeat(c): Effective number of features representing concept c.")
    print("                Low is better (≈1 means single-feature representation).")
    print("  - Top1Conc(c): Fraction of activation on the dominant feature.")
    print("                 High is better (≈1 means highly concentrated).")


if __name__ == '__main__':
    main()
