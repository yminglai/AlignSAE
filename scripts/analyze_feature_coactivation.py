#!/usr/bin/env python3
"""
Analyze SAE Feature Co-Activation Patterns Across All Layers

Creates 12 dense heatmaps (one per layer) showing:
- Y-axis: 6 relation types (which relation is being queried)
- X-axis: 10,006 features (10,000 free + 6 relation slots)
- Values: Average activation when asking about each relation

This reveals which features activate for each relation type, showing the
distributed representation structure learned by the SAE.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_sae(checkpoint_path, device):
    """Load trained SAE model (either supervised or traditional)."""
    import importlib.util
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    d_model = checkpoint['d_model']
    args = checkpoint.get('args', {})
    
    # Detect SAE type
    is_traditional = args.get('is_traditional', False)
    
    if is_traditional:
        # Traditional SAE: only n_hidden features, no relation slots
        n_hidden = args.get('n_hidden', 10000)
        
        # Import TraditionalSAE class
        train_path = Path(__file__).parent / '04_train_traditional_sae.py'
        if not train_path.exists():
            raise RuntimeError(f'Cannot find traditional SAE module at {train_path}')
        
        spec = importlib.util.spec_from_file_location('train_traditional_sae', train_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        SAEClass = getattr(mod, 'TraditionalSAE', None)
        if SAEClass is None:
            raise RuntimeError('Could not find TraditionalSAE class')
        
        model = SAEClass(d_model=d_model, n_hidden=n_hidden)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Return args with n_free set to n_hidden and n_relation=0
        args['n_free'] = n_hidden
        args['n_relation'] = 0
        
    else:
        # Supervised 1-to-1 SAE: n_free + n_relation features
        n_free = args.get('n_free', 10000)
        n_relation = args.get('n_relation', 6)
        
        # Import supervised SAE class
        train_sae_path = Path(__file__).parent / '04_train_sae.py'
        if not train_sae_path.exists():
            raise RuntimeError(f'Cannot find training module at {train_sae_path}')
        
        spec = importlib.util.spec_from_file_location('train_sae', train_sae_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        SAEClass = getattr(mod, 'LargeSupervisedSAE', None) or getattr(mod, 'SupervisedSAE', None)
        if SAEClass is None:
            raise RuntimeError('Could not find SAE class in scripts/04_train_sae.py')
        
        model = SAEClass(d_model=d_model, n_free=n_free, n_relation=n_relation)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    
    return model, args


def collect_activations_by_relation(sae, lm_model, tokenizer, qa_by_relation, layer_idx, device):
    """
    Collect average SAE activations for each relation type.
    
    Args:
        qa_by_relation: Dict mapping rule_idx -> list of QA pairs
        
    Returns:
        activation_matrix: [n_relations, n_features] average activations
        all_activations: Dict mapping rule_idx -> list of activation arrays (for consistency analysis)
    """
    n_relation = 6
    n_features = None
    
    # Store activations for each relation
    relation_activations = {}
    all_activations = {}  # Store all individual activations for consistency analysis
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for rule_idx in range(n_relation):
            if rule_idx not in qa_by_relation:
                continue
            
            qa_list = qa_by_relation[rule_idx]
            activations_list = []
            
            for qa in tqdm(qa_list, desc=f"Layer {layer_idx}, Relation {rule_idx}"):
                prompt = f"Q: {qa['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                
                # Get model hidden states
                outputs = lm_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                last_position = inputs['input_ids'].shape[1] - 1
                h = hidden_states[0, last_position, :]
                
                # Get SAE activations
                z, _ = sae(h.unsqueeze(0))
                z_np = z[0].cpu().numpy()  # [n_features]
                
                if n_features is None:
                    n_features = len(z_np)
                
                activations_list.append(z_np)
            
            # Store all activations for consistency analysis
            all_activations[rule_idx] = np.array(activations_list)  # [n_samples, n_features]
            
            # Average across samples
            avg_activation = np.mean(activations_list, axis=0)
            relation_activations[rule_idx] = avg_activation
    
    # Stack into matrix: [n_relations, n_features]
    activation_matrix = np.zeros((n_relation, n_features))
    for rule_idx in range(n_relation):
        if rule_idx in relation_activations:
            activation_matrix[rule_idx, :] = relation_activations[rule_idx]
    
    return activation_matrix, all_activations


def plot_dense_heatmap(layer_idx, activation_matrix, output_dir, n_free=10000, n_relation=6, is_traditional=False):
    """
    Create dense heatmap for one layer.
    
    Args:
        layer_idx: Layer index
        activation_matrix: [6, n_features] - rows=relations, cols=features
        output_dir: Output directory
        is_traditional: True if traditional SAE (no relation slots)
    """
    rule_names = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']
    
    n_total_features = activation_matrix.shape[1]
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(24, 8))
    
    # Main heatmap
    ax_main = plt.subplot(1, 1, 1)
    
    # Plot heatmap
    im = ax_main.imshow(activation_matrix, aspect='auto', cmap='hot', 
                        interpolation='nearest', vmin=0)

    # Add colorbar and push it farther away so it doesn't overlap the plot
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.02, pad=0.04)
    cbar.set_label('Mean Activation', rotation=270, labelpad=20, fontsize=12)
    
    # Y-axis: Relations
    ax_main.set_yticks(range(6))
    ax_main.set_yticklabels(rule_names, fontsize=12)
    ax_main.set_ylabel('Queried Relation', fontsize=14, fontweight='bold')
    
    # X-axis: Features
    n_total = activation_matrix.shape[1]
    
    if is_traditional:
        # Traditional SAE: all features are free, no relation slots
        tick_positions = [0, n_total//2, n_total-1]
        tick_labels = ['0', f'{n_total//2}', f'{n_total-1}']
        
        ax_main.set_xticks(tick_positions)
        ax_main.set_xticklabels(tick_labels, fontsize=10)
        ax_main.set_xlabel('Feature Index', fontsize=14, fontweight='bold')
        
        # Title
        title_suffix = f'Traditional SAE: {n_total} Hidden Features (No Supervision)'
    else:
        # Supervised 1-to-1 SAE: separate free features and relation slots
        tick_positions = [0, n_free, n_total-1]
        tick_labels = ['0\n(Free Features Start)', f'{n_free}\n(Relation Slots Start)', f'{n_total-1}\n(End)']
        
        ax_main.set_xticks(tick_positions)
        ax_main.set_xticklabels(tick_labels, fontsize=10)
        ax_main.set_xlabel('Feature Index', fontsize=14, fontweight='bold')
        
        # Highlight the relation slots region with a subtle band + divider
        ax_main.axvspan(n_free - 0.5, n_total - 0.5, color='red', alpha=0.05)
        ax_main.axvline(x=n_free - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        ax_main.text(n_free + n_relation / 2 - 0.5, -0.4, 'Relation Slots', color='red',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Title
        title_suffix = f'Supervised 1-to-1 SAE: {n_free} Free + {n_relation} Relation Slots'
    
    ax_main.set_title(f'Layer {layer_idx}: Feature Activation by Queried Relation\n{title_suffix}',
                     fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax_main.set_xticks(np.arange(0, n_total, step=1000), minor=True)
    ax_main.grid(which='minor', color='white', linestyle=':', linewidth=0.3, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'layer{layer_idx:02d}_dense_activation_heatmap.png', 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer {layer_idx} dense heatmap")


def plot_relation_slots_only(layer_idx, activation_matrix, output_dir, n_free=10000, n_relation=6, is_traditional=False):
    """
    Create a zoomed-in heatmap showing only the 6 relation slots (supervised SAE only).
    For traditional SAE, skip this plot since there are no relation slots.
    """
    if is_traditional:
        print(f"  Skipped relation slots plot (traditional SAE has no relation slots)")
        return
    
    rule_names = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']
    
    # Extract only relation slots (last 6 columns)
    relation_slots = activation_matrix[:, n_free:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = sns.heatmap(relation_slots, annot=True, fmt='.3f', cmap='RdYlBu_r',
                     xticklabels=[f'Slot {i}' for i in range(n_relation)],
                     yticklabels=rule_names,
                     ax=ax, cbar_kws={'label': 'Mean Activation'})
    
    ax.set_title(f'Layer {layer_idx}: Relation Slots Only (Supervised 1-to-1 SAE)\n(Rows=Query, Cols=Activated Slot)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Relation Slot', fontsize=12)
    ax.set_ylabel('Queried Relation', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'layer{layer_idx:02d}_relation_slots_only.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer {layer_idx} relation slots heatmap")


def plot_top_free_features(layer_idx, activation_matrix, output_dir, n_free=10000, top_k=50):
    """
    Create heatmap showing only the top-K most active free features.
    """
    rule_names = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']
    
    # Extract free features only
    free_features = activation_matrix[:, :n_free]
    
    # Find top-K features by average activation across all relations
    avg_per_feature = free_features.mean(axis=0)
    top_k_indices = np.argsort(avg_per_feature)[-top_k:][::-1]
    
    top_k_features = free_features[:, top_k_indices]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(top_k_features, aspect='auto', cmap='hot', interpolation='nearest')
    
    plt.colorbar(im, ax=ax, label='Mean Activation')
    
    ax.set_yticks(range(6))
    ax.set_yticklabels(rule_names, fontsize=10)
    ax.set_ylabel('Queried Relation', fontsize=12, fontweight='bold')
    
    # X-axis: show some feature indices
    if top_k <= 50:
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f'F{idx}' for idx in top_k_indices], rotation=90, fontsize=8)
    else:
        step = top_k // 20
        ax.set_xticks(range(0, top_k, step))
        ax.set_xticklabels([f'F{top_k_indices[i]}' for i in range(0, top_k, step)], 
                          rotation=90, fontsize=8)
    
    ax.set_xlabel(f'Top {top_k} Free Features (by avg activation)', fontsize=12, fontweight='bold')
    ax.set_title(f'Layer {layer_idx}: Top {top_k} Most Active Free Features by Relation', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'layer{layer_idx:02d}_top{top_k}_free_features.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer {layer_idx} top-{top_k} free features heatmap")


def create_cross_layer_summary(all_activation_matrices, output_dir):
    """
    Create summary visualizations comparing all 12 layers.
    """
    rule_names = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']
    n_layers = len(all_activation_matrices)
    
    # Extract relation slots only for all layers
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer_idx in range(12):
        ax = axes[layer_idx]
        
        if layer_idx in all_activation_matrices:
            matrix = all_activation_matrices[layer_idx]
            n_free = matrix.shape[1] - 6
            relation_slots = matrix[:, n_free:]
            
            sns.heatmap(relation_slots, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       xticklabels=[f'S{i}' for i in range(6)],
                       yticklabels=rule_names if layer_idx % 4 == 0 else [],
                       ax=ax, cbar=False, vmin=0, vmax=relation_slots.max())
            ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            
            if layer_idx % 4 != 0:
                ax.set_ylabel('')
            if layer_idx < 8:
                ax.set_xlabel('')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer_idx}')
    
    plt.suptitle('Relation Slot Activation Across All Layers\n(Rows=Query, Cols=Slot)', 
                fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_layers_relation_slots_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\nSaved cross-layer summary")


def analyze_feature_statistics(layer_idx, activation_matrix, output_dir, n_free=10000, is_traditional=False):
    """
    Compute and save statistics about feature usage.
    """
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    stats = {}
    
    for rel_idx in range(6):
        rel_activations = activation_matrix[rel_idx, :]
        
        # Free features (or all features for traditional SAE)
        free_act = rel_activations[:n_free]
        
        # Top features
        top_100_indices = np.argsort(rel_activations)[-100:][::-1]
        top_100_values = rel_activations[top_100_indices]
        
        if is_traditional:
            # Traditional SAE: only free features, no relation slots
            stats[rule_names[rel_idx]] = {
                'all_features_mean': float(rel_activations.mean()),
                'all_features_std': float(rel_activations.std()),
                'all_features_max': float(rel_activations.max()),
                'top_100_features': {
                    'indices': top_100_indices.tolist(),
                    'values': top_100_values.tolist()
                }
            }
        else:
            # Supervised SAE: separate free features and relation slots
            slot_act = rel_activations[n_free:]
            
            stats[rule_names[rel_idx]] = {
                'free_features_mean': float(free_act.mean()),
                'free_features_std': float(free_act.std()),
                'free_features_max': float(free_act.max()),
                'relation_slots': slot_act.tolist(),
                'target_slot_activation': float(slot_act[rel_idx]),
                'other_slots_mean': float(np.mean([slot_act[i] for i in range(6) if i != rel_idx])),
                'top_100_features': {
                    'indices': top_100_indices.tolist(),
                    'values': top_100_values.tolist(),
                    'n_free_in_top100': int(np.sum(top_100_indices < n_free)),
                    'n_slots_in_top100': int(np.sum(top_100_indices >= n_free))
                }
            }
    
    with open(output_dir / f'layer{layer_idx:02d}_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def analyze_common_features(layer_idx, all_activations, output_dir, n_free=10000, 
                            activation_threshold=0.1, consistency_threshold=0.8, is_traditional=False):
    """
    Analyze which features CONSISTENTLY activate across samples for each relation.
    These are the "common" features that reliably activate for a given relation type.
    
    Args:
        all_activations: Dict mapping rule_idx -> [n_samples, n_features] array
        activation_threshold: Minimum value to count as "activated"
        consistency_threshold: Fraction of samples where feature must activate (0.8 = 80%)
        is_traditional: True if traditional SAE (no relation slots)
    
    Returns:
        common_features_stats: Statistics about consistent features per relation
    """
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    common_features_stats = {}
    
    for rule_idx in range(6):
        if rule_idx not in all_activations:
            continue
        
        activations = all_activations[rule_idx]  # [n_samples, n_features]
        n_samples = activations.shape[0]
        n_features = activations.shape[1]
        
        # Binary activation matrix: 1 if activated, 0 otherwise
        is_active = (activations > activation_threshold).astype(int)
        
        # Consistency: fraction of samples where each feature activates
        consistency = is_active.mean(axis=0)  # [n_features]
        
        # Find consistently active features
        consistent_features = np.where(consistency >= consistency_threshold)[0]
        
        # Average activation strength for consistent features
        avg_activation_when_active = np.zeros(len(consistent_features))
        for i, feat_idx in enumerate(consistent_features):
            # Get activations only when feature is active
            active_mask = is_active[:, feat_idx] == 1
            if active_mask.sum() > 0:
                avg_activation_when_active[i] = activations[active_mask, feat_idx].mean()
        
        # Separate free features vs relation slots (if supervised SAE)
        if is_traditional:
            # All features are "free" - no relation slots
            consistent_free = consistent_features
            consistent_slots = np.array([])
            
            common_features_stats[rule_names[rule_idx]] = {
                'n_samples': int(n_samples),
                'n_consistent_features': int(len(consistent_features)),
                'n_consistent_free': int(len(consistent_free)),
                'n_consistent_slots': 0,
                'consistent_free_indices': consistent_free.tolist()[:50],  # Top 50
                'consistent_slot_indices': [],
                'consistency_scores': {
                    'free_features_max': float(consistency.max()) if len(consistency) > 0 else 0.0,
                }
            }
        else:
            # Supervised SAE: separate free features and relation slots
            consistent_free = consistent_features[consistent_features < n_free]
            consistent_slots = consistent_features[consistent_features >= n_free] - n_free
            
            common_features_stats[rule_names[rule_idx]] = {
                'n_samples': int(n_samples),
                'n_consistent_features': int(len(consistent_features)),
                'n_consistent_free': int(len(consistent_free)),
                'n_consistent_slots': int(len(consistent_slots)),
                'consistent_free_indices': consistent_free.tolist()[:50],  # Top 50
                'consistent_slot_indices': consistent_slots.tolist(),
                'consistency_scores': {
                    'free_features_max': float(consistency[:n_free].max()) if n_free > 0 else 0.0,
                    'target_slot_consistency': float(consistency[n_free + rule_idx]),
                    'other_slots_consistency': [float(consistency[n_free + i]) for i in range(6) if i != rule_idx]
                }
            }
    
    with open(output_dir / f'layer{layer_idx:02d}_common_features.json', 'w') as f:
        json.dump(common_features_stats, f, indent=2)
    
    return common_features_stats


def plot_consistency_heatmap(layer_idx, all_activations, output_dir, n_free=10000, 
                             activation_threshold=0.1, top_k=100):
    """
    Create heatmap showing feature consistency (how often features activate).
    """
    rule_names = ['Birth Date', 'Birth City', 'University', 'Major', 'Employer', 'Work City']
    
    # Compute consistency for each relation
    consistency_matrix = []
    
    for rule_idx in range(6):
        if rule_idx not in all_activations:
            consistency_matrix.append(np.zeros(n_free + 6))
            continue
        
        activations = all_activations[rule_idx]
        is_active = (activations > activation_threshold).astype(int)
        consistency = is_active.mean(axis=0)  # [n_features]
        consistency_matrix.append(consistency)
    
    consistency_matrix = np.array(consistency_matrix)  # [6, n_features]
    
    # Find top-K most consistent features across all relations
    avg_consistency = consistency_matrix.mean(axis=0)
    top_k_indices = np.argsort(avg_consistency)[-top_k:][::-1]
    
    top_k_consistency = consistency_matrix[:, top_k_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(top_k_consistency, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    plt.colorbar(im, ax=ax, label='Consistency (fraction of samples)')
    
    ax.set_yticks(range(6))
    ax.set_yticklabels(rule_names, fontsize=11)
    ax.set_ylabel('Queried Relation', fontsize=13, fontweight='bold')
    
    # X-axis
    if top_k <= 100:
        step = max(1, top_k // 20)
        ax.set_xticks(range(0, top_k, step))
        ax.set_xticklabels([f'F{top_k_indices[i]}' if top_k_indices[i] < n_free else f'S{top_k_indices[i]-n_free}' 
                           for i in range(0, top_k, step)], rotation=90, fontsize=8)
    
    ax.set_xlabel(f'Top {top_k} Most Consistent Features', fontsize=13, fontweight='bold')
    ax.set_title(f'Layer {layer_idx}: Feature Consistency Across Samples\n'
                 f'(Values = Fraction of samples where feature activates > {activation_threshold})',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'layer{layer_idx:02d}_feature_consistency.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer {layer_idx} consistency heatmap")


def main():
    parser = argparse.ArgumentParser(description='Analyze SAE feature co-activation patterns')
    parser.add_argument('--sae_dir', type=str, default='models/sae_per_layer',
                       help='Directory containing SAE checkpoints (layer0/, layer1/, ...)')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/checkpoint-step-10000',
                       help='Fine-tuned language model checkpoint')
    parser.add_argument('--qa_file', type=str, default='data/generated/qa_test_id.jsonl',
                       help='QA pairs to analyze')
    parser.add_argument('--output_dir', type=str, default='results/feature_coactivation',
                       help='Output directory for results')
    parser.add_argument('--layers', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11',
                       help='Comma-separated layer indices to analyze')
    parser.add_argument('--samples_per_relation', type=int, default=500,
                       help='Number of samples to use per relation')
    parser.add_argument('--top_k_features', type=int, default=50,
                       help='Number of top free features to visualize separately')
    parser.add_argument('--activation_threshold', type=float, default=0.1,
                       help='Threshold for considering a feature as activated (for consistency analysis)')
    parser.add_argument('--summary_only', action='store_true',
                       help='Only generate cross-layer summary (assumes individual layers already processed)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load LM model
    print(f"Loading LM model from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    lm_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load QA data and group by relation
    print(f"\nLoading QA data from {args.qa_file}")
    with open(args.qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    qa_by_relation = defaultdict(list)
    for qa in qa_pairs:
        qa_by_relation[qa['rule_idx']].append(qa)
    
    # Sample per relation
    print(f"\nSampling {args.samples_per_relation} questions per relation:")
    for rel_idx in range(6):
        available = len(qa_by_relation[rel_idx])
        sampled = min(args.samples_per_relation, available)
        qa_by_relation[rel_idx] = qa_by_relation[rel_idx][:sampled]
        print(f"  Relation {rel_idx}: {sampled} questions")
    
    total_questions = sum(len(qa_by_relation[i]) for i in range(6))
    print(f"Total: {total_questions} questions\n")
    
    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]
    
    all_activation_matrices = {}
    
    # If summary_only, load existing matrices and skip processing
    if args.summary_only:
        print("Summary-only mode: Loading existing activation matrices...")
        for layer_idx in layers:
            matrix_file = output_dir / f'layer{layer_idx:02d}_activation_matrix.npy'
            if matrix_file.exists():
                all_activation_matrices[layer_idx] = np.load(matrix_file)
                print(f"  Loaded layer {layer_idx}")
        
        if len(all_activation_matrices) > 1:
            print(f"\nGenerating cross-layer summary...")
            create_cross_layer_summary(all_activation_matrices, output_dir)
            print("Summary complete!")
        return
    
    # Process each layer
    for layer_idx in layers:
        print(f"\n{'='*70}")
        print(f"Processing Layer {layer_idx}")
        print('='*70)
        
        # Load SAE for this layer
        sae_checkpoint = Path(args.sae_dir) / f'layer{layer_idx}' / 'sae_best.pt'
        if not sae_checkpoint.exists():
            sae_checkpoint = Path(args.sae_dir) / f'layer{layer_idx}' / 'sae_final.pt'
        if not sae_checkpoint.exists():
            sae_checkpoint = Path(args.sae_dir) / f'layer{layer_idx}' / 'traditional_sae_final.pt'
        
        if not sae_checkpoint.exists():
            print(f"Warning: No SAE checkpoint found for layer {layer_idx}, skipping")
            continue
        
        print(f"Loading SAE from {sae_checkpoint}")
        sae, sae_args = load_sae(sae_checkpoint, device)
        
        n_free = sae_args.get('n_free', 10000)
        n_relation = sae_args.get('n_relation', 6)
        is_traditional = sae_args.get('is_traditional', False)
        
        if is_traditional:
            print(f"SAE config: Traditional SAE with {n_free} hidden features (no supervision)")
        else:
            print(f"SAE config: Supervised 1-to-1 SAE with {n_free} free features + {n_relation} relation slots")
        
        # Collect activations
        print(f"\nCollecting activations...")
        activation_matrix, all_activations = collect_activations_by_relation(
            sae, lm_model, tokenizer, qa_by_relation, layer_idx, device
        )
        
        all_activation_matrices[layer_idx] = activation_matrix
        
        # Save activation matrix
        np.save(output_dir / f'layer{layer_idx:02d}_activation_matrix.npy', activation_matrix)
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        plot_dense_heatmap(layer_idx, activation_matrix, output_dir, n_free, n_relation, is_traditional)
        plot_relation_slots_only(layer_idx, activation_matrix, output_dir, n_free, n_relation, is_traditional)
        plot_top_free_features(layer_idx, activation_matrix, output_dir, n_free, args.top_k_features)
        
        # NEW: Analyze common/consistent features
        print(f"\nAnalyzing feature consistency (threshold={args.activation_threshold})...")
        if not is_traditional:
            # Only do detailed consistency analysis for supervised SAE
            common_stats = analyze_common_features(layer_idx, all_activations, output_dir, n_free,
                                                   activation_threshold=args.activation_threshold,
                                                   is_traditional=is_traditional)
            plot_consistency_heatmap(layer_idx, all_activations, output_dir, n_free,
                                    activation_threshold=args.activation_threshold)
            
            # Print consistency summary
            print(f"\n  Common Features Summary:")
            for rel_name, cstats in common_stats.items():
                print(f"    {rel_name}: {cstats['n_consistent_features']} consistent features "
                      f"({cstats['n_consistent_free']} free + {cstats['n_consistent_slots']} slots)")
        else:
            print(f"  Skipped consistency analysis (not needed for traditional SAE)")
        
        # Compute statistics
        stats = analyze_feature_statistics(layer_idx, activation_matrix, output_dir, n_free, is_traditional)
        
        # Print summary
        print(f"\nLayer {layer_idx} Summary:")
        print(f"  Activation matrix shape: {activation_matrix.shape}")
        
        if is_traditional:
            print(f"  All features mean activation: {activation_matrix.mean():.4f}")
            print(f"  All features std: {activation_matrix.std():.4f}")
        else:
            print(f"  Free features mean activation: {activation_matrix[:, :n_free].mean():.4f}")
            print(f"  Relation slots mean activation: {activation_matrix[:, n_free:].mean():.4f}")
            print(f"  Diagonal strength (target slots): {np.diag(activation_matrix[:, n_free:]).mean():.4f}")
            print(f"  Off-diagonal strength (other slots): {(activation_matrix[:, n_free:].sum() - np.diag(activation_matrix[:, n_free:]).sum()) / 30:.4f}")
            
            # Print consistency summary (only for supervised SAE)
            print(f"\n  Common Features Summary:")
            for rel_name, cstats in common_stats.items():
                print(f"    {rel_name}: {cstats['n_consistent_features']} consistent features "
                      f"({cstats['n_consistent_free']} free + {cstats['n_consistent_slots']} slots)")
    
    # Generate cross-layer summary
    if len(all_activation_matrices) > 1:
        print(f"\n{'='*70}")
        print("Generating Cross-Layer Summary")
        print('='*70)
        create_cross_layer_summary(all_activation_matrices, output_dir)
    
    # Save overall summary
    # Get n_free and n_relation from last processed layer
    n_free_final = 10000
    n_relation_final = 6
    if len(all_activation_matrices) > 0:
        last_layer = max(all_activation_matrices.keys())
        n_total = all_activation_matrices[last_layer].shape[1]
        n_free_final = n_total - 6
        n_relation_final = 6
    
    summary = {
        'layers_analyzed': sorted(all_activation_matrices.keys()),
        'samples_per_relation': args.samples_per_relation,
        'total_questions': total_questions,
        'n_free': n_free_final,
        'n_relation': n_relation_final
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print('='*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey outputs:")
    print(f"  - layer{{00-11}}_dense_activation_heatmap.png (full feature activation map)")
    print(f"  - layer{{00-11}}_relation_slots_only.png (6x6 dependency matrix, supervised SAE only)")
    print(f"  - layer{{00-11}}_top{args.top_k_features}_free_features.png (top features)")
    print(f"  - layer{{00-11}}_feature_consistency.png (common features across samples)")
    print(f"  - all_layers_relation_slots_summary.png (12-panel overview)")
    print(f"  - layer{{00-11}}_activation_matrix.npy (numerical data)")
    print(f"  - layer{{00-11}}_statistics.json (detailed stats)")
    print(f"  - layer{{00-11}}_common_features.json (consistent features per relation)")
    print(f"\nNOTE: For traditional SAE, relation slots plots are skipped (no supervision)")


if __name__ == '__main__':
    main()
