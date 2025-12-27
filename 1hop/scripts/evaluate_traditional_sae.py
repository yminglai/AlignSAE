#!/usr/bin/env python3
"""
Evaluate Traditional SAE Binding Accuracy on All Layers

Tests:
1. In-Distribution (ID) accuracy: Test templates 0-1, test persons
2. Out-of-Distribution (OOD) accuracy: Test templates 2-3, test persons

For traditional SAE (unsupervised), we evaluate:
- Answer accuracy (does the model generate the correct answer?)
- Per-rule answer accuracy breakdown
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import re
import pandas as pd


def normalize_date(date_str):
    """Normalize date strings to a common format for comparison."""
    date_str = date_str.strip().lower()
    
    match = re.match(r'(\d{1,2}),\s*([a-z]+),\s*(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    match = re.match(r'([a-z]+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    match = re.match(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    match = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
    if match:
        year, month_num, day = match.groups()
        months = ['', 'january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
        month = months[int(month_num)]
        return (int(day), month, int(year))
    
    return None


def compare_answers(gold_answer, gen_answer, rule_name):
    """Compare gold and generated answers with rule-specific handling."""
    gold_answer = gold_answer.strip().lower()
    gen_answer = gen_answer.strip().lower()
    
    if not gen_answer:
        return False
    
    if rule_name == 'birth_date':
        gold_date = normalize_date(gold_answer)
        gen_date = normalize_date(gen_answer)
        
        if gold_date and gen_date:
            return gold_date == gen_date
    
    return (
        gold_answer in gen_answer or 
        gen_answer in gold_answer or
        gold_answer == gen_answer
    )


class TraditionalSAE(torch.nn.Module):
    """Traditional Sparse Autoencoder with only reconstruction and sparsity."""
    def __init__(self, d_model, n_hidden=10000):
        super().__init__()
        self.d_model = d_model
        self.n_hidden = n_hidden
        
        self.encoder = torch.nn.Linear(d_model, n_hidden, bias=True)
        self.decoder = torch.nn.Linear(n_hidden, d_model, bias=True)
    
    def forward(self, h):
        z = F.relu(self.encoder(h))
        h_recon = self.decoder(z)
        return z, h_recon


def load_traditional_sae(checkpoint_path, device):
    """Load trained traditional SAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    d_model = checkpoint['d_model']
    n_hidden = checkpoint['args']['n_hidden']
    
    model = TraditionalSAE(d_model=d_model, n_hidden=n_hidden)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['args']


def evaluate_answer_accuracy_with_sae(sae, lm_model, tokenizer, qa_file, layer_idx, split_name="test"):
    """
    Evaluate answer accuracy for traditional SAE with SAE reconstruction intervention.
    Uses PURE QA format (no biography context) - knowledge retrieval mode.
    
    The SAE reconstruction is applied to the hidden states at the specified layer,
    and we check if the model can still generate the correct answer.
    
    Returns:
        - answer_acc: Does the model generate the correct answer with SAE reconstruction?
        - per_rule_metrics: Breakdown by each rule
    """
    device = next(lm_model.parameters()).device
    
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    results = []
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    qa_to_kg_mapping = {
        'birth_date': 'birth_date',
        'birth_city': 'birth_city',
        'university': 'university',
        'major': 'major',
        'employer': 'employer',
        'company_city': 'work_city',
    }
    
    total = 0
    answer_correct = 0
    
    per_rule = defaultdict(lambda: {'total': 0, 'answer_correct': 0})
    
    lm_model.eval()
    sae.eval()
    
    # Hook storage for intervention
    hook_handles = []
    reconstructed_hidden = [None]
    
    def create_intervention_hook(sae_model, target_layer):
        """Create a hook that replaces hidden states with SAE reconstruction."""
        def hook(module, input, output):
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply SAE reconstruction to all positions
            batch_size, seq_len, d_model = hidden_states.shape
            h_flat = hidden_states.view(-1, d_model)
            
            with torch.no_grad():
                _, h_recon = sae_model(h_flat)
            
            h_recon = h_recon.view(batch_size, seq_len, d_model)
            
            # Store for debugging
            reconstructed_hidden[0] = h_recon
            
            if isinstance(output, tuple):
                return (h_recon,) + output[1:]
            else:
                return h_recon
        return hook
    
    # Get the transformer layers - handle different model architectures
    if hasattr(lm_model, 'transformer'):
        # GPT-2 style
        layers = lm_model.transformer.h
    elif hasattr(lm_model, 'model') and hasattr(lm_model.model, 'layers'):
        # LLaMA style
        layers = lm_model.model.layers
    elif hasattr(lm_model, 'gpt_neox') and hasattr(lm_model.gpt_neox, 'layers'):
        # GPT-NeoX style
        layers = lm_model.gpt_neox.layers
    else:
        raise ValueError(f"Unknown model architecture: {type(lm_model)}")
    
    # Register hook on the target layer
    target_layer = layers[layer_idx]
    hook = target_layer.register_forward_hook(create_intervention_hook(sae, layer_idx))
    hook_handles.append(hook)
    
    try:
        with torch.no_grad():
            for qa in tqdm(qa_pairs, desc=f"Evaluating {split_name}"):
                prompt = f"Q: {qa['question']}\nA:"
                
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                
                # Generate answer with SAE intervention (hook is active)
                generated = lm_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(
                    generated[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                if '.' in generated_text:
                    generated_text = generated_text.split('.')[0].strip()
                if '\n' in generated_text:
                    generated_text = generated_text.split('\n')[0].strip()
                
                answer_is_correct = compare_answers(qa['answer'], generated_text, qa['rule_name'])
                
                total += 1
                if answer_is_correct:
                    answer_correct += 1
                
                kg_rule_name = qa_to_kg_mapping.get(qa['rule_name'], qa['rule_name'])
                per_rule[kg_rule_name]['total'] += 1
                if answer_is_correct:
                    per_rule[kg_rule_name]['answer_correct'] += 1
                
                results.append({
                    'question': qa['question'],
                    'gold_answer': qa['answer'],
                    'generated_answer': generated_text,
                    'true_rule': qa['rule_idx'],
                    'answer_correct': answer_is_correct,
                    'rule_name': qa['rule_name'],
                })
    finally:
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
    
    answer_acc = answer_correct / total if total > 0 else 0
    
    per_rule_metrics = {}
    for rule_name in rule_names:
        if per_rule[rule_name]['total'] > 0:
            count = per_rule[rule_name]['total']
            per_rule_metrics[rule_name] = {
                'answer_acc': per_rule[rule_name]['answer_correct'] / count,
                'count': count
            }
        else:
            per_rule_metrics[rule_name] = {
                'answer_acc': 0.0,
                'count': 0
            }
    
    return {
        'answer_acc': answer_acc,
        'total': total,
        'per_rule_metrics': per_rule_metrics,
        'detailed_results': results
    }


def evaluate_reconstruction_mse(sae, lm_model, tokenizer, qa_file, layer_idx):
    """Evaluate SAE reconstruction MSE on QA activations."""
    device = next(lm_model.parameters()).device
    
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f][:100]  # Sample for efficiency
    
    total_mse = 0.0
    count = 0
    
    sae.eval()
    lm_model.eval()
    
    # hidden_states indexing: 0 = embedding, 1 = layer0, 2 = layer1, etc.
    hidden_layer_idx = layer_idx + 1
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Computing reconstruction MSE"):
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            z, h_recon = sae(h)
            mse = F.mse_loss(h_recon, h).item()
            total_mse += mse
            count += 1
    
    avg_mse = total_mse / count if count > 0 else 0.0
    return avg_mse


def compute_feature_concept_alignment(sae, lm_model, tokenizer, qa_file, layer_idx, top_k=100):
    """
    Compute the alignment between SAE features and concepts (relations).
    
    For each QA, we get the SAE activations and record which features fire strongly.
    Then we build a co-activation matrix: [n_features, n_relations] showing how often
    each feature activates for each relation type.
    
    Finally, we use greedy/Hungarian matching to find the best 1-to-1 mapping
    between top features and relations, and compute diagonal accuracy.
    
    Args:
        sae: Traditional SAE model
        lm_model: Language model
        tokenizer: Tokenizer
        qa_file: Path to QA file
        layer_idx: Layer index (0-indexed)
        top_k: Number of top features to consider for matching
    
    Returns:
        - diagonal_acc: Best 1-to-1 matching accuracy
        - best_matching: Dict mapping relation -> best feature idx
        - coactivation_matrix: [top_k, n_relations] activation counts
    """
    from scipy.optimize import linear_sum_assignment
    
    device = next(lm_model.parameters()).device
    n_relations = 6
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    qa_to_kg_mapping = {
        'birth_date': 'birth_date',
        'birth_city': 'birth_city',
        'university': 'university',
        'major': 'major',
        'employer': 'employer',
        'company_city': 'work_city',
    }
    
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    # hidden_states indexing: 0 = embedding, 1 = layer0, etc.
    hidden_layer_idx = layer_idx + 1
    
    # Collect feature activations for each relation
    n_features = sae.n_hidden
    feature_activations = defaultdict(lambda: defaultdict(list))  # feature_idx -> relation -> [activations]
    
    # Also track total activation per feature to find most active features
    feature_total_activation = np.zeros(n_features)
    
    sae.eval()
    lm_model.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Computing feature-concept alignment"):
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            # Get SAE activations
            z, _ = sae(h)  # z: [1, n_features]
            z = z[0].cpu().numpy()  # [n_features]
            
            # Get relation for this QA
            kg_rule_name = qa_to_kg_mapping.get(qa['rule_name'], qa['rule_name'])
            relation_idx = rule_names.index(kg_rule_name)
            
            # Record activations
            for feat_idx in range(n_features):
                if z[feat_idx] > 0:  # Only count non-zero activations
                    feature_activations[feat_idx][relation_idx].append(z[feat_idx])
                    feature_total_activation[feat_idx] += z[feat_idx]
    
    # Find top-k most active features
    top_feature_indices = np.argsort(feature_total_activation)[-top_k:][::-1]
    
    # Build co-activation matrix for top features: [top_k, n_relations]
    # Each cell = average activation of feature for that relation
    coactivation_matrix = np.zeros((top_k, n_relations))
    activation_counts = np.zeros((top_k, n_relations))
    
    for i, feat_idx in enumerate(top_feature_indices):
        for rel_idx in range(n_relations):
            activations = feature_activations[feat_idx][rel_idx]
            if len(activations) > 0:
                coactivation_matrix[i, rel_idx] = np.mean(activations)
                activation_counts[i, rel_idx] = len(activations)
    
    # Normalize by row (per feature) to get selectivity
    row_sums = coactivation_matrix.sum(axis=1, keepdims=True)
    selectivity_matrix = coactivation_matrix / (row_sums + 1e-9)
    
    # Use Hungarian algorithm to find best 1-to-1 matching
    # We want to maximize, so negate for minimization
    cost_matrix = -selectivity_matrix[:n_relations, :]  # Only use top 6 features for 6 relations
    
    # If we have fewer features than relations, pad
    if cost_matrix.shape[0] < n_relations:
        padding = np.zeros((n_relations - cost_matrix.shape[0], n_relations))
        cost_matrix = np.vstack([cost_matrix, padding])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Compute diagonal accuracy: average selectivity on the diagonal
    diagonal_selectivity = []
    best_matching = {}
    
    for feat_rank, rel_idx in zip(row_ind, col_ind):
        if feat_rank < top_k:
            feat_idx = top_feature_indices[feat_rank]
            diagonal_selectivity.append(selectivity_matrix[feat_rank, rel_idx])
            best_matching[rule_names[rel_idx]] = {
                'feature_idx': int(feat_idx),
                'feature_rank': int(feat_rank),
                'selectivity': float(selectivity_matrix[feat_rank, rel_idx]),
                'avg_activation': float(coactivation_matrix[feat_rank, rel_idx]),
                'count': int(activation_counts[feat_rank, rel_idx])
            }
    
    diagonal_acc = np.mean(diagonal_selectivity) if diagonal_selectivity else 0.0
    
    # Also compute a "purity" metric: for each relation, what fraction of 
    # samples have the matched feature as the top-1 activated feature?
    purity_scores = {}
    
    with torch.no_grad():
        relation_counts = defaultdict(int)
        relation_top1_correct = defaultdict(int)
        
        for qa in tqdm(qa_pairs, desc="Computing purity"):
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            z, _ = sae(h)
            z = z[0].cpu().numpy()
            
            kg_rule_name = qa_to_kg_mapping.get(qa['rule_name'], qa['rule_name'])
            
            # Get the matched feature for this relation
            if kg_rule_name in best_matching:
                matched_feat_idx = best_matching[kg_rule_name]['feature_idx']
                
                # Check if matched feature is in top-1 activated
                top1_feat = np.argmax(z)
                
                relation_counts[kg_rule_name] += 1
                if top1_feat == matched_feat_idx:
                    relation_top1_correct[kg_rule_name] += 1
        
        for rule_name in rule_names:
            if relation_counts[rule_name] > 0:
                purity_scores[rule_name] = relation_top1_correct[rule_name] / relation_counts[rule_name]
            else:
                purity_scores[rule_name] = 0.0
    
    avg_purity = np.mean(list(purity_scores.values())) if purity_scores else 0.0
    
    return {
        'diagonal_acc': float(diagonal_acc),
        'avg_purity': float(avg_purity),
        'best_matching': best_matching,
        'purity_per_relation': purity_scores,
        'top_feature_indices': top_feature_indices.tolist(),
        'selectivity_matrix': selectivity_matrix[:min(top_k, 20), :].tolist(),  # Save top 20 for viz
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_dir', type=str, default='models/traditional_sae',
                       help='Directory containing traditional SAE checkpoints for all layers')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--test_qa_id', type=str, default='data/generated/qa_test_id.jsonl',
                       help='In-distribution test QA')
    parser.add_argument('--test_qa_ood', type=str, default='data/generated/qa_test_ood.jsonl',
                       help='Out-of-distribution test QA')
    parser.add_argument('--output_dir', type=str, default='results/traditional_sae_eval')
    parser.add_argument('--layers', type=str, default='all',
                       help='Layers to evaluate (comma-separated or "all")')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load LM model
    print(f"\nLoading LM model from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine layers to evaluate
    sae_dir = Path(args.sae_dir)
    if args.layers == 'all':
        layer_dirs = sorted([d for d in sae_dir.iterdir() if d.is_dir() and d.name.startswith('layer')],
                           key=lambda x: int(x.name.replace('layer', '')))
        layers = [int(d.name.replace('layer', '')) for d in layer_dirs]
    else:
        layers = [int(l.strip()) for l in args.layers.split(',')]
    
    print(f"Evaluating layers: {layers}")
    
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    all_results = {}
    
    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Load SAE for this layer
        sae_checkpoint = sae_dir / f'layer{layer_idx}' / 'traditional_sae_final.pt'
        if not sae_checkpoint.exists():
            print(f"  WARNING: SAE checkpoint not found at {sae_checkpoint}, skipping layer {layer_idx}")
            continue
        
        print(f"Loading SAE from {sae_checkpoint}")
        sae, sae_args = load_traditional_sae(sae_checkpoint, device)
        
        # ===================================================================
        # IN-DISTRIBUTION EVALUATION
        # ===================================================================
        print(f"\n--- In-Distribution Test (templates 0-1, test persons) ---")
        id_results = evaluate_answer_accuracy_with_sae(
            sae, lm_model, tokenizer, args.test_qa_id, layer_idx, split_name="Test-ID"
        )
        
        print(f"  Answer Accuracy: {id_results['answer_acc']:.3f}")
        print(f"  Per-rule (Test-ID):")
        for rule_name in rule_names:
            metrics = id_results['per_rule_metrics'][rule_name]
            print(f"    {rule_name:20s}: Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
        
        # ===================================================================
        # OUT-OF-DISTRIBUTION EVALUATION
        # ===================================================================
        print(f"\n--- Out-of-Distribution Test (templates 2-3, test persons) ---")
        ood_results = evaluate_answer_accuracy_with_sae(
            sae, lm_model, tokenizer, args.test_qa_ood, layer_idx, split_name="Test-OOD"
        )
        
        print(f"  Answer Accuracy: {ood_results['answer_acc']:.3f}")
        print(f"  Per-rule (Test-OOD):")
        for rule_name in rule_names:
            metrics = ood_results['per_rule_metrics'][rule_name]
            print(f"    {rule_name:20s}: Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
        
        # ===================================================================
        # RECONSTRUCTION MSE
        # ===================================================================
        print(f"\n--- SAE Reconstruction Quality ---")
        recon_mse = evaluate_reconstruction_mse(sae, lm_model, tokenizer, args.test_qa_id, layer_idx)
        print(f"  Reconstruction MSE: {recon_mse:.6f}")
        
        # ===================================================================
        # FEATURE-CONCEPT ALIGNMENT (Diagonal Accuracy)
        # ===================================================================
        print(f"\n--- Feature-Concept Alignment (1-to-1 Mapping) ---")
        alignment_results = compute_feature_concept_alignment(
            sae, lm_model, tokenizer, args.test_qa_id, layer_idx, top_k=100
        )
        
        print(f"  Diagonal Accuracy (best 1-to-1 selectivity): {alignment_results['diagonal_acc']:.3f}")
        print(f"  Average Purity (top-1 feature matches): {alignment_results['avg_purity']:.3f}")
        print(f"  Best Feature Matching:")
        for rule_name in rule_names:
            if rule_name in alignment_results['best_matching']:
                match = alignment_results['best_matching'][rule_name]
                print(f"    {rule_name:20s}: Feature #{match['feature_idx']:5d} "
                      f"(selectivity={match['selectivity']:.3f}, purity={alignment_results['purity_per_relation'].get(rule_name, 0):.3f})")
        
        # Store results
        all_results[f'layer{layer_idx}'] = {
            'layer': layer_idx,
            'test_id': {
                'answer_acc': float(id_results['answer_acc']),
                'per_rule': {r: {k: float(v) if isinstance(v, (int, float)) else v 
                                for k, v in id_results['per_rule_metrics'][r].items()} 
                            for r in rule_names}
            },
            'test_ood': {
                'answer_acc': float(ood_results['answer_acc']),
                'per_rule': {r: {k: float(v) if isinstance(v, (int, float)) else v 
                                for k, v in ood_results['per_rule_metrics'][r].items()} 
                            for r in rule_names}
            },
            'reconstruction_mse': float(recon_mse),
            'diagonal_acc': alignment_results['diagonal_acc'],
            'avg_purity': alignment_results['avg_purity'],
            'best_matching': alignment_results['best_matching'],
            'purity_per_relation': alignment_results['purity_per_relation']
        }
    
    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print("\n" + "="*80)
    print("SUMMARY: Traditional SAE Evaluation Across All Layers")
    print("="*80)
    print(f"\n{'Layer':<10} {'ID Acc':>10} {'OOD Acc':>10} {'Diag Acc':>10} {'Purity':>10} {'Recon MSE':>12}")
    print("-" * 65)
    
    for layer_key in sorted(all_results.keys(), key=lambda x: int(x.replace('layer', ''))):
        layer_data = all_results[layer_key]
        print(f"{layer_key:<10} {layer_data['test_id']['answer_acc']:>10.3f} "
              f"{layer_data['test_ood']['answer_acc']:>10.3f} "
              f"{layer_data.get('diagonal_acc', 0):>10.3f} "
              f"{layer_data.get('avg_purity', 0):>10.3f} "
              f"{layer_data['reconstruction_mse']:>12.6f}")
    
    # Per-rule summary
    print("\n" + "-"*80)
    print("Per-Rule Answer Accuracy (Test-ID)")
    print("-"*80)
    header = f"{'Layer':<10}" + "".join([f"{r[:8]:>10}" for r in rule_names])
    print(header)
    print("-" * (10 + 10 * len(rule_names)))
    
    for layer_key in sorted(all_results.keys(), key=lambda x: int(x.replace('layer', ''))):
        layer_data = all_results[layer_key]
        row = f"{layer_key:<10}"
        for rule_name in rule_names:
            acc = layer_data['test_id']['per_rule'][rule_name]['answer_acc']
            row += f"{acc:>10.3f}"
        print(row)
    
    print("\n" + "-"*80)
    print("Per-Rule Answer Accuracy (Test-OOD)")
    print("-"*80)
    print(header)
    print("-" * (10 + 10 * len(rule_names)))
    
    for layer_key in sorted(all_results.keys(), key=lambda x: int(x.replace('layer', ''))):
        layer_data = all_results[layer_key]
        row = f"{layer_key:<10}"
        for rule_name in rule_names:
            acc = layer_data['test_ood']['per_rule'][rule_name]['answer_acc']
            row += f"{acc:>10.3f}"
        print(row)
    
    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    
    with open(output_dir / 'traditional_sae_accuracy_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'traditional_sae_accuracy_results.json'}")
    
    # Create summary CSV
    summary_data = []
    for layer_key in sorted(all_results.keys(), key=lambda x: int(x.replace('layer', ''))):
        layer_data = all_results[layer_key]
        row = {
            'layer': layer_data['layer'],
            'id_answer_acc': layer_data['test_id']['answer_acc'],
            'ood_answer_acc': layer_data['test_ood']['answer_acc'],
            'diagonal_acc': layer_data.get('diagonal_acc', 0),
            'avg_purity': layer_data.get('avg_purity', 0),
            'reconstruction_mse': layer_data['reconstruction_mse']
        }
        for rule_name in rule_names:
            row[f'id_{rule_name}_acc'] = layer_data['test_id']['per_rule'][rule_name]['answer_acc']
            row[f'ood_{rule_name}_acc'] = layer_data['test_ood']['per_rule'][rule_name]['answer_acc']
            # Add purity per relation
            row[f'purity_{rule_name}'] = layer_data.get('purity_per_relation', {}).get(rule_name, 0)
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'traditional_sae_accuracy_summary.csv', index=False)
    print(f"Summary CSV saved to {output_dir / 'traditional_sae_accuracy_summary.csv'}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
