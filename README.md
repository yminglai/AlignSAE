# AlignSAE: Concept-Aligned Sparse Autoencoders

**Idea.** Train a sparse autoencoder that dedicates one slot per concept, so features are easy to find, read, and steer. We first pretrain an SAE for reconstruction/sparsity, then post-train with a binding loss that maps each ontology concept to its own slot while leaving a free bank for everything else.

![AlignSAE overview](main.png)
<br><sub>An overview of our approach. Left: An unsupervised SAE trained post hoc on frozen LLM activations optimizes only reconstruction and sparsity, so each concept tends to be spread across multiple features, making interventions unreliable. Right: Our Concept-Aligned SAE adds a supervised binding loss that maps each concept to a dedicated feature, yielding clean, isolated activations that are easy to find, interpret, and steer.</sub>

**Paper (reference wording):** AlignSAE: Concept-Aligned Sparse Autoencoders — Minglai Yang, Xinyu Guo, Mihai Surdeanu, Liangming Pan.

**What you get**
- 1-hop (bio QA) pipeline with concept-aligned slots (BIRTH_DATE, BIRTH_CITY, UNIVERSITY, MAJOR, EMPLOYER, WORK_CITY).
- 2-hop reasoning extension with step-wise slot binding and swap interventions.
- Ready-to-run scripts for data gen, training, SAE, evaluation, and swaps.

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

### 2-Hop (relations) quick start

For the relation-chained task (2-hop):
- Data gen/convert: `python 2hop/_gen_data/generate_two_hop.py --path 2hop/_dataset/_org/{train,val}.jsonl`
- Optional downsample: `python 2hop/split_dataset.py` (writes *_4k.jsonl)
- Train 2-hop LM: `bash 2hop/run_train_two_hop.sh`
- Full SAE pipeline: `bash 2hop/run_full_pipeline.sh` (activations → SAE → eval → swap → grokking)
- Swap demo: `bash 2hop/run_swap_layer6.sh`

Details: see `RUN.md` (top-level) and `2hop/README.md`.

## Key Takeaways
- Mid-layer slots (e.g., layer 6) bind concepts one-to-one and enable reliable swaps.
- Post-training with a binding loss converts SAEs from diagnostic probes to operational knobs.
- 2-hop extension shows step-wise relation binding and controllable swaps along a path.

## Key Metrics

### Binding Accuracy (Bio QA)
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

**2-Hop note:** For relation swaps (R1 → R2), intervention strength α is swept logarithmically (e.g., 0.5 → 1000). Supervised slots respond; unsupervised features generally do not.

## Results Snapshot

![Main results figure](main.png)

**Key Finding (Bio QA):** Semantic concept binding emerges in middle transformer layers (5–8). Layer 6 achieves perfect binding accuracy and strong swap controllability.

**2-Hop Reasoning:** Swapping relation slots shows supervised SAEs outperform unsupervised baselines (see `2hop/swap_results/`).

**Grokking:** Binding selectivity jumps mid-training; see `2hop/grokking_analysis/` for curves and matrices.

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

2hop/
├── run_full_pipeline.sh       # End-to-end 2-hop
├── run_train_two_hop.sh       # Train 2-hop LM
├── run_swap_layer6.sh         # Swap (supervised SAE)
├── run_unsupervised_swap.sh   # Swap (unsupervised SAE)
├── 01_collect_activations_2hop.py
├── 01_find_unsupervised_features.py
├── 02_train_sae_2hop.py
├── 02_swap_unsupervised_single_alpha.py
├── 03_evaluate_sae_2hop.py
├── 04_swap_intervention.py
├── 05_merge_and_plot_swap.py
├── 06_analyze_grokking.py
├── grokking_analysis/         # Curves, matrices
└── swap_results/              # Supervised swap JSONs per α

results/
└── sae_per_layer/             # Evaluation results per layer
├── unsupervised_sae_swap/     # Unsupervised 2-hop swap JSONs

plots/
├── pretraining_vs_posttraining_acl_style.pdf
├── pretraining_vs_posttraining_acl_style.png
└── swap_comparison_acl.pdf
```

## License

MIT License

