# AlignSAE: Supervised Sparse Autoencoder with Perfect Feature Binding

This repository implements a **supervised sparse autoencoder (SAE)** that achieves **perfect 1-to-1 mapping** between semantic concepts and latent features for biography question-answering.

## Goal

Train an SAE where each latent feature corresponds to exactly one semantic rule:

| Feature | Semantic Rule |
|---------|---------------|
| f₁ | Birth Date |
| f₂ | Birth City |
| f₃ | University |
| f₄ | Major/Field |
| f₅ | Employer |
| f₆ | Work City |

Each feature must:
- Activate **only** for its corresponding question type
- Contain sufficient information to answer the question
- Generalize to unseen question phrasings (OOD templates)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python scripts/01_generate_dataset.py      # Generate data
python scripts/02_sft_base_model.py        # Fine-tune LLM
python scripts/03_collect_activations.py   # Collect activations
python scripts/04_train_sae.py             # Train SAE
python scripts/05_evaluate_sae.py          # Evaluate binding accuracy
```

## Key Metrics

### Binding Accuracy
Measures whether questions activate the correct feature and generate correct answers:

| Metric | Target | Description |
|--------|--------|-------------|
| Train Slot Binding | ≥ 0.95 | Sanity check |
| Test-ID Slot Binding | ≥ 0.85 | Generalization to new persons |
| Test-OOD Slot Binding | ≥ 0.75 | Generalization to new phrasings |
| Diagonal Accuracy | ≥ 0.85 | 1-to-1 mapping quality |

### Swap Controllability
Tests the SAE's ability to control outputs by amplifying specific features:
- Amplify feature B when asking about attribute A
- Success if model outputs attribute B's value instead

## Results Summary

| Layer | Train Slot Acc | Test OOD Slot Acc | Diagonal Acc | Swap Success |
|-------|----------------|-------------------|--------------|--------------|
| 0 | 0.232 | 0.165 | 0.238 | 0.08 |
| 5 | **1.000** | 0.887 | **1.000** | 0.80 |
| 6 | **1.000** | 0.912 | **1.000** | **1.00** |
| 7 | **1.000** | 0.877 | **1.000** | 0.84 |

**Key Finding**: Semantic concept binding emerges in middle transformer layers (5-8). Layer 6 achieves perfect binding accuracy and swap controllability.

## Project Structure

```
scripts/
├── 01_generate_dataset.py     # Generate synthetic biography data
├── 02_sft_base_model.py       # Fine-tune base language model
├── 03_collect_activations.py  # Extract hidden state activations
├── 04_train_sae.py            # Train supervised SAE
├── 05_evaluate_sae.py         # Evaluate binding accuracy
└── 06_swap_evaluate.py        # Evaluate swap controllability

data/
├── entities/                  # Entity lists (names, cities, etc.)
├── templates/                 # Biography templates
├── qa_templates/              # Question templates
└── generated/                 # Generated datasets

models/
├── base_sft/                  # Fine-tuned LLM checkpoints
└── sae_per_layer/             # Trained SAE models per layer

results/
└── sae_per_layer/             # Evaluation results per layer
```

## License

MIT License

