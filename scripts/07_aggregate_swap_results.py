#!/usr/bin/env python3
"""
Aggregate per-layer swap evaluation results and plot a heatmap (layers vs alpha).

Usage: python3 scripts/07_aggregate_swap_results.py

Produces:
- plots/swap_controllability_heatmap.png
- results/sae_eval_swap_extended/swap_summary.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    base = Path('results/sae_eval_swap_extended')
    layers = []
    alpha_set = set()
    # collect per-layer files
    per_layer = {}
    for layer_dir in sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('layer')], key=lambda x: int(x.name.replace('layer',''))):
        layer_idx = int(layer_dir.name.replace('layer',''))
        file = layer_dir / 'swap_controllability_all_alphas.json'
        if not file.exists():
            # try earlier filename
            candidates = list(layer_dir.glob('swap_alpha_*.json'))
            if len(candidates) == 0:
                print(f"Warning: no results found for {layer_dir}")
                continue
            # load all alpha files and combine
            combined = {}
            for c in candidates:
                d = json.load(open(c))
                # each file is per-alpha
                # try to infer alpha from filename or keys
                try:
                    alpha = float(c.stem.split('_')[-1])
                except Exception:
                    alpha = None
                if alpha is not None:
                    combined[alpha] = d
            per_layer[layer_idx] = combined
            for a in combined.keys():
                alpha_set.add(a)
        else:
            d = json.load(open(file))
            # d['all_results'] maps alpha->result
            allr = d.get('all_results', d.get('all_results', {}))
            # keys might be strings
            processed = {}
            for k, v in allr.items():
                try:
                    ak = float(k)
                except Exception:
                    ak = float(str(k))
                processed[ak] = v
                alpha_set.add(ak)
            per_layer[layer_idx] = processed

    if len(per_layer) == 0:
        print('No per-layer results found in', base)
        return

    alphas = sorted(alpha_set)
    layers_sorted = sorted(per_layer.keys())

    # build matrix
    mat = np.full((len(layers_sorted), len(alphas)), np.nan, dtype=float)
    for i, L in enumerate(layers_sorted):
        for j, a in enumerate(alphas):
            info = per_layer.get(L, {}).get(a)
            if info is None:
                mat[i, j] = np.nan
            else:
                mat[i, j] = float(info.get('overall_success_rate', np.nan))

    # save CSV
    df = pd.DataFrame(mat, index=[f'layer{L}' for L in layers_sorted], columns=[str(a) for a in alphas])
    out_csv = base / 'swap_summary.csv'
    df.to_csv(out_csv)
    print('Saved summary CSV to', out_csv)

    # Plot heatmap
    plots = Path('plots')
    plots.mkdir(exist_ok=True)
    plt.figure(figsize=(max(8, len(alphas)*0.6), max(6, len(layers_sorted)*0.4)))
    sns.set(style='white')
    ax = sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label':'Swap success rate'}, vmin=0.0, vmax=1.0)
    ax.set_title('Swap Controllability: Layers vs Alpha')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Layer')
    plt.tight_layout()
    out_png = plots / 'swap_controllability_heatmap.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print('Saved heatmap to', out_png)


if __name__ == '__main__':
    main()
