#!/usr/bin/env python3
"""
Train Traditional (Unsupervised) SAE - Baseline for Comparison

This script trains a standard Sparse Autoencoder with ONLY:
- Reconstruction loss
- Sparsity penalty (L1)

NO supervised alignment, NO relation binding, NO value prediction.
Used as a baseline to show that supervised SAE achieves better relation binding.

Architecture: 10,000 hidden features (no relation slots)
Loss: reconstruction + L1 sparsity
"""

import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np


class TraditionalSAE(nn.Module):
    """
    Traditional Sparse Autoencoder with only reconstruction and sparsity.
    No supervised relation binding.
    """
    def __init__(self, d_model, n_hidden=10000):
        super().__init__()
        self.d_model = d_model
        self.n_hidden = n_hidden
        
        # Encoder: h -> z
        self.encoder = nn.Linear(d_model, n_hidden, bias=True)
        
        # Decoder: z -> h_recon
        self.decoder = nn.Linear(n_hidden, d_model, bias=True)
        
        # Initialize decoder as transpose of encoder (tied weights approach)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, h):
        """
        Args:
            h: [batch, d_model] - transformer activations
        
        Returns:
            z: [batch, n_hidden] - sparse codes (ReLU activated)
            h_recon: [batch, d_model] - reconstructed activations
        """
        # Encode with ReLU for sparsity
        z = F.relu(self.encoder(h))
        
        # Decode
        h_recon = self.decoder(z)
        
        return z, h_recon


def load_activations(activation_file, max_samples=None):
    """Load activation data."""
    print(f"Loading activations from {activation_file}")
    with open(activation_file, 'rb') as f:
        data = pickle.load(f)
    
    if max_samples is not None:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} activation samples")
    return data


def train_traditional_sae(model, train_data, args, device):
    """
    Train traditional SAE with reconstruction + sparsity loss only.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Extract activations
    h_train = torch.stack([torch.from_numpy(item['h']).float() for item in train_data]).to(device)
    
    n_samples = len(h_train)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    history = {
        'loss': [],
        'recon_loss': [],
        'sparsity_loss': [],
        'l0_norm': []
    }
    
    print("\nTraining Traditional SAE...")
    print(f"  Total samples: {n_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda sparsity: {args.lambda_sparse}")
    
    model.train()
    
    for epoch in range(args.epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        h_shuffled = h_train[perm]
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sparse = 0.0
        epoch_l0 = 0.0
        
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx in pbar:
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, n_samples)
            
            h_batch = h_shuffled[start_idx:end_idx]
            
            # Forward pass
            z, h_recon = model(h_batch)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(h_recon, h_batch)
            
            # Sparsity loss (L1)
            sparsity_loss = z.abs().mean()
            
            # Total loss
            loss = recon_loss + args.lambda_sparse * sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparse += sparsity_loss.item()
            epoch_l0 += (z > 0).float().sum(dim=1).mean().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'L0': f'{(z > 0).float().sum(dim=1).mean().item():.1f}'
            })
        
        # Epoch averages
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_sparse = epoch_sparse / n_batches
        avg_l0 = epoch_l0 / n_batches
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['sparsity_loss'].append(avg_sparse)
        history['l0_norm'].append(avg_l0)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, "
              f"Sparse={avg_sparse:.4f}, L0={avg_l0:.1f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train Traditional (Unsupervised) SAE')
    parser.add_argument('--activation_file', type=str, required=True,
                       help='Pickle file with activation data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save model and results')
    parser.add_argument('--n_hidden', type=int, default=10000,
                       help='Number of hidden features')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_sparse', type=float, default=1e-3,
                       help='L1 sparsity coefficient')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of samples for faster training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load activation data
    train_data = load_activations(args.activation_file, args.max_samples)
    
    # Get d_model from first sample
    d_model = train_data[0]['h'].shape[0]
    print(f"Transformer hidden dimension: {d_model}")
    print(f"SAE hidden features: {args.n_hidden}")
    
    # Create model
    model = TraditionalSAE(d_model=d_model, n_hidden=args.n_hidden)
    model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train_traditional_sae(model, train_data, args, device)
    
    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'd_model': d_model,
        'args': {
            'n_hidden': args.n_hidden,
            'n_free': args.n_hidden,  # For compatibility with analysis script
            'n_relation': 0,
            'is_traditional': True,  # Flag to identify traditional SAE
            **vars(args)
        },
        'history': history
    }
    
    checkpoint_path = output_dir / 'traditional_sae_final.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"\nSaved model to {checkpoint_path}")
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save args
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final reconstruction loss: {history['recon_loss'][-1]:.6f}")
    print(f"Final L0 norm: {history['l0_norm'][-1]:.1f} active features")
    print(f"Final sparsity loss: {history['sparsity_loss'][-1]:.6f}")


if __name__ == '__main__':
    main()
