#!/usr/bin/env python3
"""
Sanity Checks for Traditional SAE Evaluation

1. Print exact activation extraction position for each layer
2. Compute activation RMS/std per layer and normalized MSE
3. Check layer 11 reconstruction error distribution
4. Compare intervention methods
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


class TraditionalSAE(torch.nn.Module):
    """Traditional Sparse Autoencoder."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_dir', type=str, default='models/traditional_sae')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--test_qa_id', type=str, default='data/generated/qa_test_id.jsonl')
    parser.add_argument('--output_dir', type=str, default='results/traditional_sae_eval')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load LM model
    print(f"\nLoading LM model from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    lm_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check model architecture
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE CHECK")
    print("="*80)
    print(f"Model type: {type(lm_model)}")
    print(f"Model config: {lm_model.config}")
    
    if hasattr(lm_model, 'transformer'):
        print(f"Architecture: GPT-2 style")
        print(f"Number of layers: {len(lm_model.transformer.h)}")
        transformer_layers = lm_model.transformer.h
        # Check layer structure
        layer0 = transformer_layers[0]
        print(f"Layer 0 structure: {layer0}")
        print(f"  - Has ln_1 (pre-attention LN): {hasattr(layer0, 'ln_1')}")
        print(f"  - Has ln_2 (pre-MLP LN): {hasattr(layer0, 'ln_2')}")
        print(f"  - Has attn: {hasattr(layer0, 'attn')}")
        print(f"  - Has mlp: {hasattr(layer0, 'mlp')}")
    
    # Load QA data
    with open(args.test_qa_id, 'r') as f:
        qa_pairs = [json.loads(line) for line in f][:200]  # Sample for efficiency
    
    print("\n" + "="*80)
    print("CHECK 1: ACTIVATION EXTRACTION POSITIONS")
    print("="*80)
    print("\nChecking what hidden_states[i] actually represents...")
    
    # Run a single forward pass to check hidden states
    prompt = f"Q: {qa_pairs[0]['question']}\nA:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = lm_model(**inputs, output_hidden_states=True)
    
    print(f"\nNumber of hidden states returned: {len(outputs.hidden_states)}")
    print(f"  hidden_states[0]: embedding output (before any transformer block)")
    for i in range(1, len(outputs.hidden_states)):
        print(f"  hidden_states[{i}]: after transformer block {i-1} (layer {i-1})")
    
    # Check if there's final layer norm
    if hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'ln_f'):
        print(f"\nFinal LayerNorm (ln_f) exists: {lm_model.transformer.ln_f}")
        print("  NOTE: hidden_states[-1] is BEFORE ln_f, logits are computed AFTER ln_f")
    
    print("\n" + "="*80)
    print("CHECK 2: ACTIVATION STATISTICS PER LAYER")
    print("="*80)
    
    # Collect activation statistics for each layer
    layer_stats = defaultdict(lambda: {'rms': [], 'std': [], 'mean': [], 'max': [], 'min': []})
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs[:100], desc="Collecting activation stats"):
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = lm_model(**inputs, output_hidden_states=True)
            
            last_pos = inputs['input_ids'].shape[1] - 1
            
            for layer_idx in range(12):
                hidden_layer_idx = layer_idx + 1  # 0 = embedding, 1 = layer0, etc.
                h = outputs.hidden_states[hidden_layer_idx][0, last_pos, :]
                
                rms = torch.sqrt(torch.mean(h ** 2)).item()
                std = torch.std(h).item()
                mean = torch.mean(h).item()
                max_val = torch.max(h).item()
                min_val = torch.min(h).item()
                
                layer_stats[layer_idx]['rms'].append(rms)
                layer_stats[layer_idx]['std'].append(std)
                layer_stats[layer_idx]['mean'].append(mean)
                layer_stats[layer_idx]['max'].append(max_val)
                layer_stats[layer_idx]['min'].append(min_val)
    
    print(f"\n{'Layer':<8} {'RMS':>10} {'Std':>10} {'Mean':>10} {'Max':>10} {'Min':>10}")
    print("-" * 60)
    for layer_idx in range(12):
        rms = np.mean(layer_stats[layer_idx]['rms'])
        std = np.mean(layer_stats[layer_idx]['std'])
        mean = np.mean(layer_stats[layer_idx]['mean'])
        max_val = np.mean(layer_stats[layer_idx]['max'])
        min_val = np.mean(layer_stats[layer_idx]['min'])
        print(f"Layer {layer_idx:<3} {rms:>10.4f} {std:>10.4f} {mean:>10.4f} {max_val:>10.4f} {min_val:>10.4f}")
    
    print("\n" + "="*80)
    print("CHECK 3: SAE RECONSTRUCTION ERROR PER LAYER (RAW & NORMALIZED)")
    print("="*80)
    
    sae_dir = Path(args.sae_dir)
    
    print(f"\n{'Layer':<8} {'Raw MSE':>12} {'Var(h)':>12} {'Norm MSE':>12} {'R²':>10}")
    print("-" * 60)
    
    layer_mse_details = {}
    
    for layer_idx in range(12):
        sae_checkpoint = sae_dir / f'layer{layer_idx}' / 'traditional_sae_final.pt'
        if not sae_checkpoint.exists():
            print(f"Layer {layer_idx}: SAE not found")
            continue
        
        sae, sae_args = load_traditional_sae(sae_checkpoint, device)
        hidden_layer_idx = layer_idx + 1
        
        mse_list = []
        var_list = []
        
        with torch.no_grad():
            for qa in qa_pairs[:100]:
                prompt = f"Q: {qa['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = lm_model(**inputs, output_hidden_states=True)
                
                last_pos = inputs['input_ids'].shape[1] - 1
                h = outputs.hidden_states[hidden_layer_idx][0, last_pos, :].unsqueeze(0)
                
                z, h_recon = sae(h)
                mse = F.mse_loss(h_recon, h).item()
                var = torch.var(h).item()
                
                mse_list.append(mse)
                var_list.append(var)
        
        avg_mse = np.mean(mse_list)
        avg_var = np.mean(var_list)
        norm_mse = avg_mse / avg_var if avg_var > 0 else 0
        r_squared = 1 - norm_mse
        
        layer_mse_details[layer_idx] = {
            'raw_mse': avg_mse,
            'var_h': avg_var,
            'norm_mse': norm_mse,
            'r_squared': r_squared
        }
        
        print(f"Layer {layer_idx:<3} {avg_mse:>12.6f} {avg_var:>12.6f} {norm_mse:>12.6f} {r_squared:>10.4f}")
    
    print("\n" + "="*80)
    print("CHECK 4: DETAILED LAYER 11 ANALYSIS")
    print("="*80)
    
    layer_idx = 11
    sae_checkpoint = sae_dir / f'layer{layer_idx}' / 'traditional_sae_final.pt'
    
    if sae_checkpoint.exists():
        sae, sae_args = load_traditional_sae(sae_checkpoint, device)
        hidden_layer_idx = layer_idx + 1
        
        print(f"\nSAE config for layer {layer_idx}:")
        print(f"  d_model: {sae.d_model}")
        print(f"  n_hidden: {sae.n_hidden}")
        print(f"  encoder weight shape: {sae.encoder.weight.shape}")
        print(f"  decoder weight shape: {sae.decoder.weight.shape}")
        
        # Check SAE weight statistics
        print(f"\nSAE weight statistics:")
        print(f"  encoder.weight - mean: {sae.encoder.weight.mean().item():.6f}, std: {sae.encoder.weight.std().item():.6f}")
        print(f"  encoder.bias - mean: {sae.encoder.bias.mean().item():.6f}, std: {sae.encoder.bias.std().item():.6f}")
        print(f"  decoder.weight - mean: {sae.decoder.weight.mean().item():.6f}, std: {sae.decoder.weight.std().item():.6f}")
        print(f"  decoder.bias - mean: {sae.decoder.bias.mean().item():.6f}, std: {sae.decoder.bias.std().item():.6f}")
        
        # Detailed reconstruction error for layer 11
        print(f"\nDetailed reconstruction analysis for layer {layer_idx}:")
        
        mse_list = []
        z_sparsity_list = []
        z_max_list = []
        
        with torch.no_grad():
            for qa in tqdm(qa_pairs[:100], desc=f"Layer {layer_idx} detailed analysis"):
                prompt = f"Q: {qa['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = lm_model(**inputs, output_hidden_states=True)
                
                last_pos = inputs['input_ids'].shape[1] - 1
                h = outputs.hidden_states[hidden_layer_idx][0, last_pos, :].unsqueeze(0)
                
                z, h_recon = sae(h)
                
                mse = F.mse_loss(h_recon, h).item()
                z_nonzero = (z > 0).float().mean().item()
                z_max = z.max().item()
                
                mse_list.append(mse)
                z_sparsity_list.append(z_nonzero)
                z_max_list.append(z_max)
        
        print(f"  MSE - mean: {np.mean(mse_list):.6f}, std: {np.std(mse_list):.6f}, min: {np.min(mse_list):.6f}, max: {np.max(mse_list):.6f}")
        print(f"  z sparsity (% nonzero) - mean: {np.mean(z_sparsity_list)*100:.2f}%")
        print(f"  z max activation - mean: {np.mean(z_max_list):.4f}, max: {np.max(z_max_list):.4f}")
    
    print("\n" + "="*80)
    print("CHECK 5: COMPARE LAYER 9, 10, 11 SIDE BY SIDE")
    print("="*80)
    
    for layer_idx in [9, 10, 11]:
        sae_checkpoint = sae_dir / f'layer{layer_idx}' / 'traditional_sae_final.pt'
        if not sae_checkpoint.exists():
            continue
        
        sae, _ = load_traditional_sae(sae_checkpoint, device)
        hidden_layer_idx = layer_idx + 1
        
        # Check one sample in detail
        prompt = f"Q: {qa_pairs[0]['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = lm_model(**inputs, output_hidden_states=True)
            last_pos = inputs['input_ids'].shape[1] - 1
            h = outputs.hidden_states[hidden_layer_idx][0, last_pos, :].unsqueeze(0)
            
            z, h_recon = sae(h)
            
            print(f"\nLayer {layer_idx}:")
            print(f"  h shape: {h.shape}, h norm: {torch.norm(h).item():.4f}")
            print(f"  h mean: {h.mean().item():.4f}, h std: {h.std().item():.4f}")
            print(f"  z shape: {z.shape}, z nonzero: {(z > 0).sum().item()}/{z.numel()}")
            print(f"  z max: {z.max().item():.4f}, z mean (nonzero): {z[z > 0].mean().item() if (z > 0).any() else 0:.4f}")
            print(f"  h_recon norm: {torch.norm(h_recon).item():.4f}")
            print(f"  Reconstruction MSE: {F.mse_loss(h_recon, h).item():.6f}")
            print(f"  Cosine similarity: {F.cosine_similarity(h, h_recon, dim=-1).item():.4f}")
    
    print("\n" + "="*80)
    print("CHECK 6: TRAINING CONFIG COMPARISON")
    print("="*80)
    
    for layer_idx in [9, 10, 11]:
        config_file = sae_dir / f'layer{layer_idx}' / 'training_args.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"\nLayer {layer_idx} training config:")
            for k, v in config.items():
                print(f"  {k}: {v}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'layer_stats': {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in layer_stats.items()},
        'layer_mse_details': layer_mse_details
    }
    
    with open(output_dir / 'sanity_check_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'sanity_check_results.json'}")


if __name__ == "__main__":
    main()
