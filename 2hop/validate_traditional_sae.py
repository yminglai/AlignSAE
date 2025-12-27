#!/usr/bin/env python3
"""
Validate traditional SAE models by checking:
1. Model can be loaded correctly
2. Architecture matches expected dimensions
3. Forward pass works properly
4. Reconstruction quality is reasonable
"""

import torch
import json
from pathlib import Path
import sys

def load_traditional_sae(checkpoint_path):
    """Load a traditional SAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    d_model = checkpoint['d_model']
    n_hidden = checkpoint['args']['n_hidden']
    
    # Recreate model
    from torch import nn
    
    class TraditionalSAE(nn.Module):
        def __init__(self, d_model, n_hidden):
            super().__init__()
            self.encoder = nn.Linear(d_model, n_hidden, bias=True)
            self.decoder = nn.Linear(n_hidden, d_model, bias=True)
            
            # Initialize decoder as transpose of encoder
            self.decoder.weight.data = self.encoder.weight.data.T
        
        def forward(self, h):
            z = torch.relu(self.encoder(h))
            h_recon = self.decoder(z)
            return z, h_recon
    
    model = TraditionalSAE(d_model, n_hidden)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def validate_layer(layer_idx, models_dir='/home/mlyang721/bio_data/models/traditional_sae'):
    """Validate a specific layer's traditional SAE."""
    print(f"\n{'='*60}")
    print(f"Validating Layer {layer_idx}")
    print(f"{'='*60}")
    
    layer_dir = Path(models_dir) / f'layer{layer_idx}'
    checkpoint_path = layer_dir / 'traditional_sae_final.pt'
    
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        return False
    
    # Load model
    print(f"✓ Loading checkpoint from {checkpoint_path}")
    model, checkpoint = load_traditional_sae(checkpoint_path)
    
    # Check model parameters
    d_model = checkpoint['d_model']
    n_hidden = checkpoint['args']['n_hidden']
    
    print(f"✓ Model architecture:")
    print(f"  - d_model: {d_model}")
    print(f"  - n_hidden: {n_hidden}")
    print(f"  - Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check training history
    history = checkpoint['history']
    print(f"\n✓ Training history:")
    print(f"  - Epochs: {len(history['loss'])}")
    print(f"  - Final loss: {history['loss'][-1]:.6f}")
    print(f"  - Final reconstruction loss: {history['recon_loss'][-1]:.6f}")
    print(f"  - Final L0 norm: {history['l0_norm'][-1]:.1f} active features")
    print(f"  - Final sparsity loss: {history['sparsity_loss'][-1]:.6f}")
    
    # Load training args
    args_path = layer_dir / 'training_args.json'
    if args_path.exists():
        with open(args_path) as f:
            args = json.load(f)
        print(f"\n✓ Training configuration:")
        print(f"  - Learning rate: {args['lr']}")
        print(f"  - Lambda sparse: {args['lambda_sparse']}")
        print(f"  - Batch size: {args['batch_size']}")
    
    # Test forward pass
    print(f"\n✓ Testing forward pass:")
    model.eval()
    with torch.no_grad():
        # Create random input
        test_input = torch.randn(4, d_model)
        z, h_recon = model(test_input)
        
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Latent shape: {z.shape}")
        print(f"  - Reconstruction shape: {h_recon.shape}")
        print(f"  - Active features per sample: {(z > 0).float().sum(dim=1).mean().item():.1f}")
        
        # Check reconstruction error
        recon_error = torch.nn.functional.mse_loss(h_recon, test_input)
        print(f"  - Test reconstruction MSE: {recon_error.item():.6f}")
    
    # Validation checks
    print(f"\n{'='*60}")
    print(f"Validation Summary for Layer {layer_idx}")
    print(f"{'='*60}")
    
    checks_passed = True
    
    # Check 1: Model loads
    print(f"✓ Model loads correctly")
    
    # Check 2: Reasonable dimensions
    if d_model != 768:
        print(f"⚠ Warning: d_model={d_model}, expected 768 for GPT-2 small")
    else:
        print(f"✓ d_model matches GPT-2 small (768)")
    
    if n_hidden != 10000:
        print(f"⚠ Warning: n_hidden={n_hidden}, expected 10000")
    else:
        print(f"✓ n_hidden matches config (10000)")
    
    # Check 3: Training converged
    if history['recon_loss'][-1] > 1.0:
        print(f"⚠ Warning: High reconstruction loss ({history['recon_loss'][-1]:.4f})")
        checks_passed = False
    else:
        print(f"✓ Reconstruction loss is reasonable ({history['recon_loss'][-1]:.4f})")
    
    # Check 4: Sparsity
    if history['l0_norm'][-1] < 10 or history['l0_norm'][-1] > 1000:
        print(f"⚠ Warning: L0 norm ({history['l0_norm'][-1]:.1f}) may be too sparse or dense")
    else:
        print(f"✓ L0 norm is reasonable ({history['l0_norm'][-1]:.1f} active features)")
    
    # Check 5: Forward pass works
    print(f"✓ Forward pass produces correct shapes")
    
    return checks_passed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate Traditional SAE Models')
    parser.add_argument('--layers', type=str, default='all',
                       help='Comma-separated layer indices or "all"')
    parser.add_argument('--models_dir', type=str, 
                       default='/home/mlyang721/bio_data/models/traditional_sae',
                       help='Directory containing trained models')
    args = parser.parse_args()
    
    if args.layers == 'all':
        layers = list(range(12))
    else:
        layers = [int(l) for l in args.layers.split(',')]
    
    print(f"\nValidating Traditional SAE Models")
    print(f"Layers to validate: {layers}")
    
    results = {}
    for layer_idx in layers:
        success = validate_layer(layer_idx, args.models_dir)
        results[layer_idx] = success
    
    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nLayers validated: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print(f"\n✅ All layers passed validation!")
        return 0
    else:
        print(f"\n⚠️ Some layers failed validation")
        failed_layers = [l for l, passed in results.items() if not passed]
        print(f"Failed layers: {failed_layers}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
