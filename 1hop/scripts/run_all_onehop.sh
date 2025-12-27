#!/bin/bash
set -e

# One-hop (bio QA) end-to-end pipeline
# Steps: generate -> sft -> collect activations -> train SAE -> evaluate -> swap

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "[1] Generate dataset"
python scripts/01_generate_dataset.py

echo "[2] Fine-tune base model"
python scripts/02_sft_base_model.py

echo "[3] Collect activations"
python scripts/03_collect_activations.py

echo "[4] Train supervised SAE"
python scripts/04_train_sae.py

echo "[5] Evaluate binding"
python scripts/05_evaluate_sae.py

echo "[6] Swap evaluate"
python scripts/06_swap_evaluate.py

echo "Done. Artifacts stored under models/, results/, data/generated/ (ignored by git)."
