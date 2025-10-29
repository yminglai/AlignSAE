#!/usr/bin/env bash
set -eu

OUTROOT=results/sae_eval_swap_extended
mkdir -p "$OUTROOT"

ALPHAS="0.1,0.5,1,2,5,10,20,50,100,200,500,1000"
NUM_SAMPLES=200
MAX_TOKENS=200
LM_MODEL="models/base_sft/final"

for L in $(seq 0 11); do
  outdir="$OUTROOT/layer${L}"
  mkdir -p "$outdir"
  session="swap_layer${L}"

  # If session exists, skip
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists, skipping"
    continue
  fi

  SAE_CP="models/sae_per_layer/layer${L}/sae_final.pt"
  cmd="export CUDA_VISIBLE_DEVICES=1; python3 scripts/06_swap_evaluate.py --sae_checkpoint ${SAE_CP} --lm_model ${LM_MODEL} --qa_file data/generated/qa_test_id.jsonl --kg_file data/generated/test_kg.json --layer ${L} --alphas ${ALPHAS} --num_samples ${NUM_SAMPLES} --max_new_tokens ${MAX_TOKENS} --output_dir ${outdir}"

  echo "Starting tmux session $session -> $outdir/run.log"
  # Start in detached tmux session and tee output to run.log
  tmux new-session -d -s "$session" "bash -lc '${cmd} |& tee ${outdir}/run.log'"
done

echo "Launched tmux sessions swap_layer0..swap_layer11 (detached)."
echo "To attach to a session: tmux attach -t swap_layer0"
echo "To tail logs: tail -f results/sae_eval_swap_extended/layer0/run.log"
