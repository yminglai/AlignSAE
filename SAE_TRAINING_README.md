# SAE Training: Large Supervised Sparse Autoencoder

This document describes the training procedure for the Large Supervised Sparse Autoencoder (SAE) with 100,006 slots (100,000 free + 6 relation slots) designed for perfect 1-to-1 feature binding in biography question-answering tasks.

## Architecture Overview

The SAE consists of:
- **Encoder**: Maps 768-dimensional activations to 100,006 sparse features
- **Decoder**: Reconstructs original activations from sparse features
- **Value Heads**: Predict answers using relation slot activations

## Training Modes

Three training modes are supported:

1. **`joint`**: Train encoder, decoder, and value heads simultaneously
2. **`separate_activation`**: Train only encoder and decoder (reconstruction + binding losses)
3. **`separate_value`**: Train only value heads (answer prediction)

## Multi-Stage Training

### Stage 1: Reconstruction Only (50 epochs)
Focuses on learning basic reconstruction before introducing binding constraints.

### Stage 2: Full Training (100 epochs)
Introduces all loss components for complete binding and prediction.

## Loss Components

The total loss combines six components:

### 1. Reconstruction Loss
Ensures the SAE can reconstruct original activations:
$$L_{recon} = \frac{1}{N} \sum_{i=1}^N \|h_i - \hat{h}_i\|^2$$

### 2. Sparsity Loss
Encourages sparse activation patterns:
$$L_{sparse} = \frac{1}{N \cdot D} \sum_{i=1}^N \sum_{j=1}^D |z_{ij}|$$

### 3. Alignment Loss
Supervises relation slots to match one-hot rule encodings:
$$L_{align} = \frac{1}{N} \sum_{i=1}^N \|z_{rel}^{(i)} - y_{rule}^{(i)}\|^2$$

### 4. Independence Loss (Free Slots)
Decorrelates free slots within themselves (skipped for >10k slots to avoid OOM):
$$L_{indep} = \sum_{j \neq k} (\rho_{jk})^2 \quad \text{where} \quad \rho_{jk} = \frac{\sum_i (z_{ij} - \bar{z}_j)(z_{ik} - \bar{z}_k)}{\sqrt{\sum_i (z_{ij} - \bar{z}_j)^2} \sqrt{\sum_i (z_{ik} - \bar{z}_k)^2}}$$

### 5. Orthogonality Loss
Ensures relation slots remain decorrelated from free slots:
$$L_{ortho} = \sum_{j,k} (\rho_{jk}^{rel,free})^2$$

### 6. Value Prediction Loss
Trains answer prediction using relation slot activations:
$$L_{value} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(s_{y_i})}{\sum_c \exp(s_c)}$$

## Total Loss

The total loss depends on training mode:

**Joint Mode:**
$$L_{total} = \lambda_{recon} L_{recon} + \lambda_{sparse} L_{sparse} + \lambda_{align} L_{align} + \lambda_{indep} L_{indep} + \lambda_{ortho} L_{ortho} + \lambda_{value} L_{value}$$

**Separate Activation Mode:**
$$L_{total} = \lambda_{recon} L_{recon} + \lambda_{sparse} L_{sparse} + \lambda_{align} L_{align} + \lambda_{indep} L_{indep} + \lambda_{ortho} L_{ortho}$$

**Separate Value Mode:**
$$L_{total} = \lambda_{value} L_{value}$$

## Default Hyperparameters

- **Learning Rate**: 1e-3 (AdamW optimizer)
- **Batch Size**: 64
- **Stage 1 Epochs**: 50
- **Stage 2 Epochs**: 100
- **Loss Weights**:
  - $\lambda_{recon}$: 1.0
  - $\lambda_{sparse}$: 1e-3
  - $\lambda_{align}$: 1.0
  - $\lambda_{indep}$: 1e-2 (when applicable)
  - $\lambda_{ortho}$: 1e-2
  - $\lambda_{value}$: 0.5

## Parallel Training

The training supports parallel execution across multiple GPUs with automatic layer distribution:

- **GPU 1**: Layers 0-3
- **GPU 2**: Layers 4-7
- **GPU 3**: Layers 8-11

## Output Structure

Training creates organized subdirectories based on mode:
```
models/
├── feature_activation/     # separate_activation mode
├── next_token_prediction/  # separate_value mode
└── joint/                  # joint training mode
    ├── sae_final.pt
    ├── training_history.json
    └── checkpoints/
```

## Monitoring and Evaluation

During training, the following metrics are tracked:
- **Slot Accuracy**: Fraction of correct relation slot activations
- **Value Accuracy**: Fraction of correct answer predictions
- **All Loss Components**: Individual loss values for analysis

Training history is saved to `training_history.json` and can be visualized using `plot_sae_training.py`.

## Key Innovations

1. **Orthogonality Constraint**: Ensures clean separation between relation and free slots
2. **Memory-Efficient Independence**: Skips expensive independence loss for large slot counts
3. **Staged Training**: Prevents suboptimal initialization of binding constraints
4. **Multi-Mode Training**: Supports different training objectives for analysis</content>
<parameter name="filePath">/home/mlyang721/bio_data/SAE_TRAINING_README.md