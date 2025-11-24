#!/usr/bin/env python3
"""Generate side-by-side visualizations comparing supervised 1-to-1 SAE vs.
traditional SAE analysis outputs.

Outputs two figures per layer:
1. top features comparison (supervised vs traditional) with shared color scale
2. relation binding comparison (supervised relation slots vs greedy-assigned
   pseudo slots from traditional SAE free features)
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RELATION_NAMES = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']


def load_activation_matrix(result_dir, layer_idx):
    path = Path(result_dir) / f'layer{layer_idx:02d}_activation_matrix.npy'
    if not path.exists():
        raise FileNotFoundError(f'Missing activation matrix for layer {layer_idx} in {result_dir}')
    return np.load(path)


def get_topk_free_features(matrix, n_free, top_k):
    free = matrix[:, :n_free]
    avg = free.mean(axis=0)
    idx = np.argsort(avg)[-top_k:][::-1]
    return free[:, idx], [f'F{i}' for i in idx]


def prepare_supervised_topk(matrix, n_free, n_relation, top_k):
    free = matrix[:, :n_free]
    avg = free.mean(axis=0)
    top_free = max(0, top_k - n_relation)
    idx = np.argsort(avg)[-top_free:][::-1]
    data = free[:, idx]
    labels = [f'F{i}' for i in idx]
    if n_relation > 0:
        rel_slots = matrix[:, n_free: n_free + n_relation]
        data = np.concatenate([data, rel_slots], axis=1)
        labels.extend([f'Slot{i}' for i in range(n_relation)])
    return data, labels


def greedy_relation_assignment(free_matrix, n_assign=6):
    matrix = free_matrix.copy()
    assignments = {}
    used_rows = set()
    used_cols = set()

    for _ in range(min(n_assign, matrix.shape[0], matrix.shape[1])):
        masked = matrix.copy()
        masked[list(used_rows), :] = -np.inf
        masked[:, list(used_cols)] = -np.inf
        row, col = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[row, col] == -np.inf:
            break
        assignments[row] = col
        used_rows.add(row)
        used_cols.add(col)
    return assignments


def build_traditional_binding(free_matrix):
    assignments = greedy_relation_assignment(free_matrix)
    binding = np.zeros((free_matrix.shape[0], free_matrix.shape[0]))
    col_labels = []
    for rel in range(free_matrix.shape[0]):
        feat_idx = assignments.get(rel)
        if feat_idx is None:
            binding[:, rel] = 0.0
            col_labels.append('None')
        else:
            binding[:, rel] = free_matrix[:, feat_idx]
            col_labels.append(f'F{feat_idx}')
    return binding, col_labels


def plot_topk(layer_idx, sup_data, sup_labels, trad_data, trad_labels, vmax, output_dir, cmap, n_relation=6):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    datasets = [
        ('Supervised 1-to-1 SAE', sup_data, sup_labels, True),
        ('Traditional SAE', trad_data, trad_labels, False)
    ]
    for ax, (title, data, labels, has_slots) in zip(axes, datasets):
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f'Layer {layer_idx}: {title}\nTop {data.shape[1]} Free Features', fontsize=14, fontweight='bold')
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(RELATION_NAMES)))
        ax.set_yticklabels(RELATION_NAMES)
        ax.set_xlabel('Feature Index (sorted by avg activation)')
        ax.set_ylabel('Queried Relation')
        if has_slots and n_relation > 0:
            slot_start = data.shape[1] - n_relation - 0.5
            slot_end = data.shape[1] - 0.5
            ax.axvspan(slot_start, slot_end, color='red', alpha=0.08)
            ax.axvline(slot_start, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text((slot_start + slot_end) / 2, -0.5, 'Relation Slots', color='red',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Colorbar removed as per the patch requirement
    plt.tight_layout()
    out_path = Path(output_dir) / f'layer{layer_idx:02d}_top_features_comparison.png'
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_binding(layer_idx, sup_binding, trad_binding, trad_labels, vmax, output_dir, cmap):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plots = [
        ('Supervised Relation Slots', sup_binding, [f'Slot {i}' for i in range(sup_binding.shape[1])]),
        ('Traditional (Greedy Assigned Slots)', trad_binding, trad_labels)
    ]
    for ax, (title, data, xlabels) in zip(axes, plots):
        sns.heatmap(data, ax=ax, cmap=cmap, vmin=0, vmax=vmax,
                    annot=True, fmt='.2f', cbar=False, square=True)
        ax.set_title(f'Layer {layer_idx}: {title}', fontsize=13, fontweight='bold')
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_yticklabels(RELATION_NAMES, rotation=0)
        ax.set_xlabel('Target Slot / Assigned Feature')
        ax.set_ylabel('Queried Relation')
    # Colorbar removed to keep layout clean
    plt.tight_layout()
    out_path = Path(output_dir) / f'layer{layer_idx:02d}_relation_binding_comparison.png'
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare supervised vs traditional SAE visualizations')
    parser.add_argument('--supervised_dir', required=True, help='Path to supervised analysis results (npys)')
    parser.add_argument('--traditional_dir', required=True, help='Path to traditional analysis results (npys)')
    parser.add_argument('--layers', type=str, default='0-11', help='Range like start-end or comma list')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/sae_comparison')
    parser.add_argument('--color_percentile', type=float, default=99.5, help='Percentile for vmax clipping')
    parser.add_argument('--cmap', type=str, default='magma')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if '-' in args.layers:
        start, end = args.layers.split('-')
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    for layer in layers:
        sup_matrix = load_activation_matrix(args.supervised_dir, layer)
        trad_matrix = load_activation_matrix(args.traditional_dir, layer)
        sup_n_free = sup_matrix.shape[1] - 6
        trad_n_free = trad_matrix.shape[1]

        sup_top, sup_labels = prepare_supervised_topk(sup_matrix, sup_n_free, 6, args.top_k)
        trad_top, trad_labels = get_topk_free_features(trad_matrix, trad_n_free, args.top_k)
        combined = np.concatenate([sup_top.flatten(), trad_top.flatten()])
        vmax = np.percentile(combined, args.color_percentile)
        plot_topk(layer, sup_top, sup_labels, trad_top, trad_labels, vmax, output_dir, args.cmap, n_relation=6)

        sup_binding = sup_matrix[:, sup_n_free:]
        trad_binding, trad_labels = build_traditional_binding(trad_matrix[:, :trad_n_free])
        combined_binding = np.concatenate([sup_binding.flatten(), trad_binding.flatten()])
        vmax_binding = np.percentile(combined_binding, args.color_percentile)
        plot_binding(layer, sup_binding, trad_binding, trad_labels, vmax_binding, output_dir, args.cmap)


if __name__ == '__main__':
    main()
