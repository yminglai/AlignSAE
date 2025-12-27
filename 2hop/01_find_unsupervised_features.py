#!/usr/bin/env python3
"""
Step 1: Find relation-specific features in unsupervised SAE for 2-hop task.
Uses TRAIN data to find top-k features for each relation at each token position.
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


class TraditionalSAE(nn.Module):
    """Traditional Sparse Autoencoder."""
    
    def __init__(self, d_model, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        
    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        x_hat = self.decoder(z)
        return z, x_hat
    
    def encode(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        return z


RELATIONS = [
    'blames', 'reports_to', 'works_with', 'subscribes_to', 'admires',
    'protects', 'mentor_of', 'has_crush_on', 'friend_of', 'forgives',
    'follows', 'accuses', 'warns', 'cousin_of', 'classmate_of',
    'endorsed_by', 'competes_with', 'boss_of', 'neighbor_of', 'owes_debt_to'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, required=True)
    parser.add_argument('--lm_model_path', type=str, required=True)
    parser.add_argument('--train_qa_file', type=str, required=True)
    parser.add_argument('--layer_idx', type=int, default=6)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAE
    print(f"\nLoading unsupervised SAE from {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location=device)
    d_model = checkpoint['d_model']
    n_features = checkpoint['n_features']
    
    sae = TraditionalSAE(d_model=d_model, n_features=n_features)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae = sae.to(device)
    sae.eval()
    
    print(f"SAE: d_model={d_model}, n_features={n_features}")
    
    # Load language model
    print(f"\nLoading language model from {args.lm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load QA pairs
    print(f"\nLoading TRAIN data from {args.train_qa_file}")
    with open(args.train_qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(qa_pairs)} QA pairs")
    print("Using ALL training data for robust feature discovery")
    
    # Collect activations: when generating E2 (use R1), when generating E3 (use R2)
    relation_activations_r1 = {rel: [] for rel in RELATIONS}
    relation_activations_r2 = {rel: [] for rel in RELATIONS}
    
    print(f"\n{'='*80}")
    print("Collecting SAE activations during generation")
    print("  - For R1: activation when generating E2")
    print("  - For R2: activation when generating E3")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Processing QA pairs"):
            e1 = qa['entity_1']
            r1 = qa['relation_1']
            r2 = qa['relation_2']
            e2 = qa['entity_2']
            
            # ===== R1 Feature: When generating E2 =====
            # Prompt: Q: E1 [R1] [R2] \n A:
            # Generate E2, get hidden state at last position before E2 generation
            prompt = f"Q: {e1} [{r1}] [{r2}]\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Get hidden states at last prompt token (before generating E2)
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[args.layer_idx]
            h_before_e2 = hidden_states[0, -1, :]  # Last token of prompt
            
            # Get SAE activation
            z_r1 = sae.encode(h_before_e2.unsqueeze(0))[0]
            relation_activations_r1[r1].append(z_r1.cpu())
            
            # ===== R2 Feature: When generating E3 =====
            # Prompt: Q: E1 [R1] [R2] \n A: E2
            # Generate E3, get hidden state at last position after E2
            prompt_with_e2 = f"Q: {e1} [{r1}] [{r2}]\nA: {e2}"
            inputs_with_e2 = tokenizer(prompt_with_e2, return_tensors='pt').to(device)
            
            # Get hidden states at last token (after E2, before generating E3)
            outputs_with_e2 = lm_model(**inputs_with_e2, output_hidden_states=True)
            hidden_states_e2 = outputs_with_e2.hidden_states[args.layer_idx]
            h_before_e3 = hidden_states_e2[0, -1, :]  # Last token after E2
            
            # Get SAE activation
            z_r2 = sae.encode(h_before_e3.unsqueeze(0))[0]
            relation_activations_r2[r2].append(z_r2.cpu())
    
    # Use differential activation: find features that are UNIQUELY strong for each relation
    # compared to all other relations (like 1-to-1 correspondence)
    print(f"\n{'='*80}")
    print("Finding relation-specific features using differential activation")
    print("(Features that activate strongly for ONE relation vs others)")
    print(f"{'='*80}\n")
    
    # Compute average activation for each feature across all relations
    all_r1_activations = {}  # rel_name -> avg_activation_vector [n_features]
    all_r2_activations = {}
    
    for rel_name in RELATIONS:
        if len(relation_activations_r1[rel_name]) > 0:
            acts_r1 = torch.stack(relation_activations_r1[rel_name])
            all_r1_activations[rel_name] = acts_r1.mean(dim=0)  # [n_features]
        
        if len(relation_activations_r2[rel_name]) > 0:
            acts_r2 = torch.stack(relation_activations_r2[rel_name])
            all_r2_activations[rel_name] = acts_r2.mean(dim=0)  # [n_features]
    
    # For each relation, find features with highest differential:
    # differential_score = activation_for_this_relation - max(activation_for_other_relations)
    relation_to_features = {
        'r1_when_generating_e2': {},
        'r2_when_generating_e3': {}
    }
    
    print(f"{'='*80}")
    print(f"Finding top-{args.top_k} DISCRIMINATIVE features for each relation")
    print(f"{'='*80}")
    
    print(f"\n{'Relation':<20} | {'R1 Features (diff score)':<50} | {'R2 Features (diff score)':<50}")
    print("-" * 130)
    
    for rel_name in RELATIONS:
        # R1 features - find features that are uniquely strong for this relation
        if rel_name in all_r1_activations:
            target_act = all_r1_activations[rel_name]  # [n_features]
            
            # Compute max activation across all OTHER relations
            other_rels = [r for r in RELATIONS if r != rel_name and r in all_r1_activations]
            if len(other_rels) > 0:
                other_acts = torch.stack([all_r1_activations[r] for r in other_rels])  # [n_other_rels, n_features]
                max_other_act = other_acts.max(dim=0)[0]  # [n_features]
                
                # Differential score: how much stronger is this feature for target vs others
                diff_score = target_act - max_other_act  # [n_features]
                
                # Get top-k by differential score
                top_k_diff, top_k_idx = torch.topk(diff_score, k=args.top_k)
                
                relation_to_features['r1_when_generating_e2'][rel_name] = {
                    'feature_indices': top_k_idx.tolist(),
                    'differential_scores': top_k_diff.tolist(),
                    'target_activations': target_act[top_k_idx].tolist(),
                    'num_samples': len(relation_activations_r1[rel_name])
                }
                feat_str_r1 = ', '.join([f"{idx}({diff:.2f})" for idx, diff in zip(top_k_idx.tolist(), top_k_diff.tolist())])
            else:
                feat_str_r1 = "N/A (no other relations)"
        else:
            feat_str_r1 = "N/A"
        
        # R2 features - same approach
        if rel_name in all_r2_activations:
            target_act = all_r2_activations[rel_name]
            
            other_rels = [r for r in RELATIONS if r != rel_name and r in all_r2_activations]
            if len(other_rels) > 0:
                other_acts = torch.stack([all_r2_activations[r] for r in other_rels])
                max_other_act = other_acts.max(dim=0)[0]
                
                diff_score = target_act - max_other_act
                top_k_diff, top_k_idx = torch.topk(diff_score, k=args.top_k)
                
                relation_to_features['r2_when_generating_e3'][rel_name] = {
                    'feature_indices': top_k_idx.tolist(),
                    'differential_scores': top_k_diff.tolist(),
                    'target_activations': target_act[top_k_idx].tolist(),
                    'num_samples': len(relation_activations_r2[rel_name])
                }
                feat_str_r2 = ', '.join([f"{idx}({diff:.2f})" for idx, diff in zip(top_k_idx.tolist(), top_k_diff.tolist())])
            else:
                feat_str_r2 = "N/A (no other relations)"
        else:
            feat_str_r2 = "N/A"
        
        print(f"{rel_name:<20} | {feat_str_r1:<50} | {feat_str_r2:<50}")
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(relation_to_features, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Saved relation-to-features mapping to {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
