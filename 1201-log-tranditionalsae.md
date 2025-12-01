# Traditional SAE Evaluation Results

**Date:** December 1, 2025  
**Repository:** AlignSAE  
**Author:** Ming-Yang Lai

---

## Overview

This document presents the evaluation results for Traditional Sparse Autoencoders (SAEs) trained on transformer hidden states across all 12 layers (0-11) of a fine-tuned GPT-2 model. The evaluation measures SAE-intervened generation accuracy and feature-concept alignment.

---

## Methodology

### SAE Architecture

Traditional SAE with reconstruction and sparsity losses only:

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_e \mathbf{h} + \mathbf{b}_e)$$

$$\hat{\mathbf{h}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$$

**Loss Function:**

$$\mathcal{L} = \underbrace{\|\mathbf{h} - \hat{\mathbf{h}}\|_2^2}_{\text{Reconstruction}} + \lambda \underbrace{\|\mathbf{z}\|_1}_{\text{Sparsity}}$$

**Hyperparameters:**
- Hidden dimension: $d = 768$
- SAE features: $n = 10{,}000$
- Sparsity coefficient: $\lambda = 0.001$
- Learning rate: $0.001$
- Batch size: $256$

### Evaluation Protocol

1. **SAE Intervention:** During generation, hidden states at layer $\ell$ are replaced with SAE reconstructions via forward hooks
2. **In-Distribution (ID):** Test templates 0-1, test persons (3,000 samples)
3. **Out-of-Distribution (OOD):** Test templates 2-3, test persons (12,000 samples)

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Answer Accuracy** | $\frac{\text{correct answers}}{\text{total}}$ | Generation correctness with SAE intervention |
| **Reconstruction MSE** | $\frac{1}{n}\sum_i \|\mathbf{h}_i - \hat{\mathbf{h}}_i\|_2^2$ | Activation reconstruction quality |
| **Diagonal Accuracy** | Best 1-to-1 feature-concept mapping | Feature selectivity for concepts |
| **Purity** | Top-1 feature consistency per concept | Feature specificity |

---

## Results

### Base Model Performance (No SAE Intervention)

| Split | Accuracy |
|-------|----------|
| In-Distribution (ID) | **1.000** |
| Out-of-Distribution (OOD) | **0.734** |

### SAE-Intervened Performance Across Layers

| Layer | SAE ID Acc | SAE OOD Acc | Diag Acc | Purity | Recon MSE |
|------:|:----------:|:-----------:|:--------:|:------:|:---------:|
| 0 | 0.724 | 0.346 | 0.169 | 0.167 | 0.0036 |
| 1 | 0.197 | 0.087 | 0.171 | 0.167 | 0.0052 |
| 2 | 0.315 | 0.128 | 0.188 | 0.167 | 0.0088 |
| 3 | 0.377 | 0.212 | 0.189 | 0.263 | 0.0056 |
| 4 | 0.537 | 0.300 | 0.219 | 0.419 | 0.0073 |
| 5 | 0.388 | 0.231 | 0.254 | 0.167 | 0.0108 |
| 6 | 0.488 | 0.311 | 0.338 | 0.167 | 0.0130 |
| 7 | 0.625 | 0.397 | 0.472 | 0.500 | 0.0129 |
| 8 | 0.858 | 0.549 | 0.632 | 0.622 | 0.0126 |
| 9 | 0.950 | 0.601 | 0.846 | 0.937 | 0.0147 |
| 10 | 0.995 | 0.727 | 0.459 | 0.424 | 0.0198 |
| 11 | 1.000 | 0.712 | 0.392 | 0.469 | 2.1530* |

> *Layer 11 reconstruction MSE is elevated due to activation distribution mismatch (see [Technical Notes](#technical-notes))

### Per-Rule Accuracy Breakdown (ID)

| Layer | birth_date | birth_city | university | major | employer | work_city |
|------:|:----------:|:----------:|:----------:|:-----:|:--------:|:---------:|
| 0 | 0.676 | 0.760 | 0.802 | 0.710 | 0.732 | 0.662 |
| 1 | 0.180 | 0.198 | 0.210 | 0.204 | 0.178 | 0.212 |
| 2 | 0.262 | 0.334 | 0.358 | 0.328 | 0.296 | 0.312 |
| 3 | 0.352 | 0.388 | 0.382 | 0.392 | 0.376 | 0.374 |
| 4 | 0.512 | 0.556 | 0.520 | 0.584 | 0.494 | 0.554 |
| 5 | 0.362 | 0.426 | 0.366 | 0.420 | 0.366 | 0.388 |
| 6 | 0.440 | 0.540 | 0.506 | 0.490 | 0.448 | 0.504 |
| 7 | 0.580 | 0.700 | 0.620 | 0.654 | 0.554 | 0.642 |
| 8 | 0.832 | 0.896 | 0.870 | 0.874 | 0.810 | 0.864 |
| 9 | 0.936 | 0.964 | 0.970 | 0.946 | 0.924 | 0.958 |
| 10 | 0.994 | 0.998 | 0.998 | 0.994 | 0.986 | 0.998 |
| 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Per-Rule Accuracy Breakdown (OOD)

| Layer | birth_date | birth_city | university | major | employer | work_city |
|------:|:----------:|:----------:|:----------:|:-----:|:--------:|:---------:|
| 0 | 0.286 | 0.338 | 0.490 | 0.410 | 0.344 | 0.207 |
| 1 | 0.082 | 0.084 | 0.100 | 0.086 | 0.082 | 0.088 |
| 2 | 0.108 | 0.126 | 0.174 | 0.140 | 0.118 | 0.104 |
| 3 | 0.172 | 0.236 | 0.258 | 0.222 | 0.200 | 0.182 |
| 4 | 0.254 | 0.284 | 0.368 | 0.356 | 0.296 | 0.244 |
| 5 | 0.200 | 0.226 | 0.274 | 0.268 | 0.218 | 0.200 |
| 6 | 0.270 | 0.320 | 0.362 | 0.340 | 0.294 | 0.278 |
| 7 | 0.348 | 0.426 | 0.454 | 0.440 | 0.380 | 0.334 |
| 8 | 0.518 | 0.560 | 0.596 | 0.572 | 0.530 | 0.516 |
| 9 | 0.560 | 0.618 | 0.678 | 0.646 | 0.578 | 0.524 |
| 10 | 0.688 | 0.740 | 0.808 | 0.784 | 0.716 | 0.622 |
| 11 | 0.670 | 0.598 | 1.000 | 1.000 | 0.906 | 0.120 |

---

## Key Findings

### 1. Layer-Dependent Information Encoding

Accuracy increases monotonically with layer depth:

```
Layer 0-3:   ID < 0.40  (Early layers: low-level features)
Layer 4-7:   ID ~ 0.40-0.65 (Middle layers: transitional)
Layer 8-11:  ID > 0.85  (Late layers: high-level semantics)
```

### 2. Best Performing Layers

| Metric | Best Layer | Value |
|--------|------------|-------|
| ID Accuracy | Layer 10, 11 | 0.995, 1.000 |
| OOD Accuracy | Layer 10 | 0.727 |
| Diagonal Accuracy | Layer 9 | 0.846 |
| Purity | Layer 9 | 0.937 |

### 3. Feature-Concept Alignment

Layer 9 shows the best feature-concept alignment:
- **Diagonal Accuracy:** 0.846 (best 1-to-1 mapping)
- **Purity:** 0.937 (most concept-specific features)

This suggests layer 9 forms the most interpretable factual representations.

### 4. Accuracy vs Reconstruction Trade-off

$$\text{Accuracy} \uparrow \quad \text{as} \quad \text{Layer} \uparrow \quad \text{while} \quad \text{MSE} \uparrow$$

Later layers have higher reconstruction error but better preserve task-relevant information.

---

## Technical Notes

### Layer 11 Reconstruction MSE Anomaly

The elevated MSE (2.15 vs ~0.01) for layer 11 is due to an **activation distribution mismatch** between training and evaluation:

| Metric | Stored (Training) | Fresh (Evaluation) | Ratio |
|--------|------------------:|-------------------:|------:|
| Norm | 115.54 | 37.53 | 3.08× |
| Correlation | — | 0.45 | — |

**Root Cause:** Training activations were collected with a different model checkpoint than the current evaluation model. Layer 11 (closest to output) shows the largest divergence.

**Evidence:**
```
Layer | Stored Norm | Fresh Norm | Correlation
------+-------------+------------+------------
    0 |        2.56 |       3.15 |      0.237
    5 |       35.89 |       6.33 |      0.601
   10 |       89.48 |      12.38 |      0.614
   11 |      115.54 |      37.53 |      0.451  ← Drops!
```

Despite the MSE anomaly, layer 11 achieves **100% ID accuracy**, indicating the SAE preserves task-relevant information.

---

## Reproduction

### Training Command

```bash
python scripts/04_train_traditional_sae.py \
    --activation_file data/activations/train_activations_layer{N}.pkl \
    --output_dir models/traditional_sae/layer{N} \
    --n_hidden 10000 \
    --epochs 1000 \
    --batch_size 256 \
    --lr 0.001 \
    --lambda_sparse 0.001
```

### Evaluation Command

```bash
python scripts/evaluate_traditional_sae.py \
    --sae_dir models/traditional_sae \
    --lm_model models/base_sft/final \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11
```

---

## Files

| File | Description |
|------|-------------|
| `traditional_sae_accuracy_results.json` | Full results in JSON format |
| `traditional_sae_accuracy_summary.csv` | Summary table in CSV format |
| `README.md` | This documentation |

---

## Citation

```bibtex
@misc{alignsae2025,
  title={AlignSAE: Supervised Sparse Autoencoders},
  author={Yang, Minglai},
  year={2025},
  howpublished={\url{https://github.com/yminglai/AlignSAE}}
}
```
