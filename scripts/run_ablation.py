#!/usr/bin/env python3
"""
Run ablation study over SAE loss components.

This script runs a set of ablation configs (turning off individual loss terms by
setting their lambda to 0) for specified transformer layers. For each config and
layer it will:
  1. Train SAE (calls scripts/04_train_sae.py)
  2. Evaluate binding accuracy (calls scripts/05_evaluate_sae.py)
  3. Run swap controllability (calls scripts/06_swap_evaluate.py)
  4. Collect results into results/ablation/<config>/layer<layer>/

The training and evaluation scripts already use tqdm; this script uses
subprocess.run without capturing stdout so the tqdm progress bars are shown live.

Use --dry_run to print planned commands without executing them.
"""
import argparse
import os
import subprocess
import json
from pathlib import Path


DEFAULT_ABLATIONS = {
    'baseline': {},
    'no_sparse': {'--lambda_sparse': '0.0'},
    'no_align': {'--lambda_align': '0.0'},
    'no_indep': {'--lambda_indep': '0.0'},
    'no_ortho': {'--lambda_ortho': '0.0'},
    'no_value': {'--lambda_value': '0.0'},
    'no_indep_no_ortho': {'--lambda_indep': '0.0', '--lambda_ortho': '0.0'},
}


def run_cmd(cmd, env=None, dry_run=False):
    print("\n> ", " ".join(cmd))
    if dry_run:
        return True
    try:
        # Don't capture stdout/stderr so child process progress bars show live
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Command failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='0',
                        help='Comma-separated layer indices to run (e.g. 0,6)')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--epochs_stage1', type=int, default=2,
                        help='Short defaults for quick tests; override for full runs')
    parser.add_argument('--epochs_stage2', type=int, default=4,
                        help='Short defaults for quick tests; override for full runs')
    parser.add_argument('--n_free', type=int, default=10000)
    parser.add_argument('--ablation_list', type=str, default='',
                        help='Comma-separated ablation keys from DEFAULT_ABLATIONS; empty=all')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training step (useful if checkpoints already exist)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    results_base = base_dir / 'results' / 'ablation'
    models_base = base_dir / 'models' / 'ablation'
    results_base.mkdir(parents=True, exist_ok=True)
    models_base.mkdir(parents=True, exist_ok=True)

    if args.ablation_list:
        keys = [k.strip() for k in args.ablation_list.split(',') if k.strip()]
    else:
        keys = list(DEFAULT_ABLATIONS.keys())

    layers = [int(x) for x in args.layers.split(',') if x.strip()]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    summary = {}

    for key in keys:
        if key not in DEFAULT_ABLATIONS:
            print(f"Unknown ablation key: {key} - skipping")
            continue

        cfg = DEFAULT_ABLATIONS[key]
        print(f"\n=== Running ablation: {key} ===")
        summary[key] = {}

        for layer in layers:
            print(f"\n--- Layer {layer} ({key}) ---")
            output_dir = models_base / key / f'layer{layer}'
            results_dir = results_base / key / f'layer{layer}'
            output_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            activation_file = base_dir / f"data/activations/train_activations_layer{layer}.pkl"

            # TRAIN
            trained_ok = True
            if not args.skip_train:
                train_cmd = [
                    'python', str(base_dir / 'scripts' / '04_train_sae.py'),
                    '--activation_file', str(activation_file),
                    '--output_dir', str(output_dir),
                    '--mode', 'joint',
                    '--n_free', str(args.n_free),
                    '--epochs_stage1', str(args.epochs_stage1),
                    '--epochs_stage2', str(args.epochs_stage2),
                ]

                # append ablation-specific overrides
                for k, v in cfg.items():
                    train_cmd.extend([k, v])

                trained_ok = run_cmd(train_cmd, env=env, dry_run=args.dry_run)

            # EVAL
            # Prefer best checkpoint if available, else final
            best_ckpt = output_dir / 'sae_best.pt'
            final_ckpt = output_dir / 'sae_final.pt'
            sae_ckpt = best_ckpt if best_ckpt.exists() else final_ckpt

            if args.dry_run:
                sae_ckpt = output_dir / 'sae_best.pt'

            eval_ok = False
            if sae_ckpt.exists() or args.dry_run:
                eval_cmd = [
                    'python', str(base_dir / 'scripts' / '05_evaluate_sae.py'),
                    '--sae_checkpoint', str(sae_ckpt),
                    '--lm_model', str(base_dir / 'models' / 'base_sft' / 'checkpoint-step-10000'),
                    '--activation_file', str(activation_file),
                    '--layer', str(layer),
                    '--output_dir', str(results_dir)
                ]
                eval_ok = run_cmd(eval_cmd, env=env, dry_run=args.dry_run)
            else:
                print(f"No checkpoint found for layer {layer} in {output_dir}; skipping evaluation")

            # SWAP EVAL
            swap_ok = False
            if sae_ckpt.exists() or args.dry_run:
                swap_cmd = [
                    'python', str(base_dir / 'scripts' / '06_swap_evaluate.py'),
                    '--sae_checkpoint', str(sae_ckpt),
                    '--lm_model', str(base_dir / 'models' / 'base_sft' / 'checkpoint-step-10000'),
                    '--qa_file', str(base_dir / 'data' / 'generated' / 'qa_test_id.jsonl'),
                    '--kg_file', str(base_dir / 'data' / 'generated' / 'test_kg.json'),
                    '--layer', str(layer),
                    '--num_samples', '50',
                    '--output_dir', str(results_dir / 'swap')
                ]
                swap_ok = run_cmd(swap_cmd, env=env, dry_run=args.dry_run)
            
            # Collect main metrics if evaluation produced results
            metrics = {
                'train_success': bool(trained_ok),
                'eval_success': bool(eval_ok),
                'swap_success': bool(swap_ok),
            }

            # Try to read binding_accuracy_results.json
            binding_file = results_dir / 'binding_accuracy_results.json'
            if binding_file.exists():
                try:
                    with open(binding_file, 'r') as f:
                        data = json.load(f)
                    metrics['train_slot_acc'] = data.get('train', {}).get('slot_binding_acc', None)
                    metrics['test_ood_slot_acc'] = data.get('test_ood', {}).get('slot_binding_acc', None)
                    metrics['diagonal_acc'] = data.get('diagonal_accuracy', None)
                    metrics['swap_best_alpha'] = data.get('swap_controllability', {}).get('best_alpha')
                    metrics['swap_best_success'] = data.get('swap_controllability', {}).get('best_success_rate')
                except Exception as e:
                    print(f"Failed to read {binding_file}: {e}")

            summary[key][str(layer)] = metrics

    # Save summary
    summary_path = results_base / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary saved to {summary_path}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Ablation study runner for SAE losses.

This script orchestrates multiple training runs of `04_train_sae.py` with
different loss components zeroed out, then evaluates each trained model using
`05_evaluate_sae.py` and (optionally) extended swap controllability via
`06_swap_evaluate.py`. Finally it aggregates metrics into a summary JSON/CSV.

Designed for lightweight comparative experiments: you can reduce epochs and
number of free slots to make runs tractable.

Outputs (per configuration name `cfg_name`):
  models/ablation/{cfg_name}/        -> checkpoints & history
  results/ablation/{cfg_name}/       -> evaluation artifacts
  results/ablation/ablation_summary.json
  results/ablation/ablation_summary.csv

Example:
  python scripts/run_ablation.py \
    --activation_file data/activations/train_activations_layer0.pkl \
    --lm_model models/base_sft/final \
    --epochs_stage1 5 --epochs_stage2 10 --n_free 2048 --device cuda

Configuration set (by default):
  baseline          : all losses enabled
  no_sparse         : lambda_sparse=0
  no_align          : lambda_align=0
  no_indep          : lambda_indep=0
  no_ortho          : lambda_ortho=0
  no_value          : lambda_value=0
  no_regularization : sparse+indep+ortho=0 (align & value kept)

You can extend/modify the CONFIGS list below.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import csv

DEFAULT_CONFIGS = [
    {
        "name": "baseline",
        "lambda_recon": 1.0,
        "lambda_sparse": 1e-3,
        "lambda_align": 1.0,
        "lambda_indep": 1e-2,
        "lambda_ortho": 1e-2,
        "lambda_value": 0.5,
    },
    {
        "name": "no_sparse",
        "lambda_recon": 1.0,
        "lambda_sparse": 0.0,
        "lambda_align": 1.0,
        "lambda_indep": 1e-2,
        "lambda_ortho": 1e-2,
        "lambda_value": 0.5,
    },
    {
        "name": "no_align",
        "lambda_recon": 1.0,
        "lambda_sparse": 1e-3,
        "lambda_align": 0.0,
        "lambda_indep": 1e-2,
        "lambda_ortho": 1e-2,
        "lambda_value": 0.5,
    },
    {
        "name": "no_indep",
        "lambda_recon": 1.0,
        "lambda_sparse": 1e-3,
        "lambda_align": 1.0,
        "lambda_indep": 0.0,
        "lambda_ortho": 1e-2,
        "lambda_value": 0.5,
    },
    {
        "name": "no_ortho",
        "lambda_recon": 1.0,
        "lambda_sparse": 1e-3,
        "lambda_align": 1.0,
        "lambda_indep": 1e-2,
        "lambda_ortho": 0.0,
        "lambda_value": 0.5,
    },
    {
        "name": "no_value",
        "lambda_recon": 1.0,
        "lambda_sparse": 1e-3,
        "lambda_align": 1.0,
        "lambda_indep": 1e-2,
        "lambda_ortho": 1e-2,
        "lambda_value": 0.0,
    },
    {
        "name": "no_regularization",
        "lambda_recon": 1.0,
        "lambda_sparse": 0.0,
        "lambda_align": 1.0,
        "lambda_indep": 0.0,
        "lambda_ortho": 0.0,
        "lambda_value": 0.5,
    },
]


def run_cmd(cmd, cwd=None):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def train_config(cfg, args):
    model_dir = Path(args.model_base) / cfg['name']
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'scripts/04_train_sae.py',
           '--activation_file', args.activation_file,
           '--output_dir', str(model_dir),
           '--epochs_stage1', str(args.epochs_stage1),
           '--epochs_stage2', str(args.epochs_stage2),
           '--batch_size', str(args.batch_size),
           '--lr', str(args.lr),
           '--n_free', str(args.n_free),
           '--mode', args.mode,
           '--lambda_recon', str(cfg['lambda_recon']),
           '--lambda_sparse', str(cfg['lambda_sparse']),
           '--lambda_align', str(cfg['lambda_align']),
           '--lambda_indep', str(cfg['lambda_indep']),
           '--lambda_ortho', str(cfg['lambda_ortho']),
           '--lambda_value', str(cfg['lambda_value'])]
    run_cmd(cmd)
    return model_dir / 'sae_final.pt'


def evaluate_config(cfg, checkpoint_path, args):
    eval_dir = Path(args.results_base) / cfg['name']
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'scripts/05_evaluate_sae.py',
           '--sae_checkpoint', str(checkpoint_path),
           '--lm_model', args.lm_model,
           '--activation_file', args.recon_activation_file,
           '--train_qa', args.train_qa,
           '--train_kg', args.train_kg,
           '--test_qa_id', args.test_qa_id,
           '--test_qa_ood', args.test_qa_ood,
           '--test_kg', args.test_kg,
           '--output_dir', str(eval_dir),
           '--layer', str(args.layer)]
    run_cmd(cmd)
    result_file = eval_dir / 'binding_accuracy_results.json'
    if not result_file.exists():
        raise FileNotFoundError(f"Expected evaluation results at {result_file}")
    with open(result_file) as f:
        metrics = json.load(f)
    return metrics


def swap_config(cfg, checkpoint_path, args):
    # Optional extended swap evaluation (can be skipped to save time)
    if not args.run_swap:
        return None
    swap_dir = Path(args.results_base) / cfg['name'] / 'swap_extended'
    swap_dir.mkdir(parents=True, exist_ok=True)
    alphas_str = ','.join([str(a) for a in args.swap_alphas])
    cmd = [sys.executable, 'scripts/06_swap_evaluate.py',
           '--sae_checkpoint', str(checkpoint_path),
           '--lm_model', args.lm_model,
           '--qa_file', args.test_qa_id,
           '--kg_file', args.test_kg,
           '--layer', str(args.layer),
           '--alphas', alphas_str,
           '--num_samples', str(args.swap_samples),
           '--max_new_tokens', str(args.swap_max_new_tokens),
           '--output_dir', str(swap_dir)]
    run_cmd(cmd)
    summary_file = swap_dir / 'swap_controllability_all_alphas.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def aggregate(summary_path, rows):
    # JSON
    with open(summary_path, 'w') as f:
        json.dump({r['config']: r for r in rows}, f, indent=2)
    # CSV
    csv_path = summary_path.with_suffix('.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved ablation summary to {summary_path} and {csv_path}")


def extract_primary_metrics(cfg_name, eval_metrics, swap_metrics):
    train = eval_metrics['train']
    test_ood = eval_metrics['test_ood']
    row = {
        'config': cfg_name,
        'train_slot_acc': train['slot_binding_acc'],
        'train_answer_acc': train['answer_acc'],
        'ood_slot_acc': test_ood['slot_binding_acc'],
        'ood_answer_acc': test_ood['answer_acc'],
        'diagonal_acc': eval_metrics['diagonal_accuracy'],
        'recon_mse': eval_metrics['reconstruction_mse'],
    }
    if swap_metrics:
        row.update({
            'swap_best_alpha': swap_metrics.get('best_alpha'),
            'swap_best_success_rate': swap_metrics.get('best_success_rate'),
        })
    else:
        row.update({'swap_best_alpha': None, 'swap_best_success_rate': None})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_file', type=str, required=True, help='Training activations pickle for SAE stage 1/2')
    parser.add_argument('--recon_activation_file', type=str, default='data/activations/train_activations_layer0.pkl', help='File used for reconstruction MSE evaluation')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--train_qa', type=str, default='data/generated/qa_train.jsonl')
    parser.add_argument('--train_kg', type=str, default='data/generated/train_kg.json')
    parser.add_argument('--test_qa_id', type=str, default='data/generated/qa_test_id.jsonl')
    parser.add_argument('--test_qa_ood', type=str, default='data/generated/qa_test_ood.jsonl')
    parser.add_argument('--test_kg', type=str, default='data/generated/test_kg.json')
    parser.add_argument('--model_base', type=str, default='models/ablation')
    parser.add_argument('--results_base', type=str, default='results/ablation')
    parser.add_argument('--epochs_stage1', type=int, default=5)
    parser.add_argument('--epochs_stage2', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_free', type=int, default=2048, help='Reduced number of free slots for faster ablation')
    parser.add_argument('--mode', type=str, default='joint', choices=['joint','separate_activation','separate_value'])
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--run_swap', action='store_true', help='Also run extended swap evaluation')
    parser.add_argument('--swap_alphas', type=float, nargs='*', default=[0.5,1,2,5,10,20,50])
    parser.add_argument('--swap_samples', type=int, default=100)
    parser.add_argument('--swap_max_new_tokens', type=int, default=60)
    parser.add_argument('--configs', type=str, default='', help='Optional comma-separated subset of config names to run')
    args = parser.parse_args()

    Path(args.model_base).mkdir(parents=True, exist_ok=True)
    Path(args.results_base).mkdir(parents=True, exist_ok=True)

    # Filter configs if requested
    if args.configs:
        wanted = set([x.strip() for x in args.configs.split(',') if x.strip()])
        configs = [c for c in DEFAULT_CONFIGS if c['name'] in wanted]
        if not configs:
            raise ValueError(f"No matching configs for: {wanted}")
    else:
        configs = DEFAULT_CONFIGS

    summary_rows = []
    start_time = datetime.utcnow().isoformat()
    print(f"Starting ablation study at {start_time} UTC with {len(configs)} configurations")

    for cfg in configs:
        print("\n" + "="*80)
        print(f"Configuration: {cfg['name']}")
        print("="*80)
        checkpoint_path = train_config(cfg, args)
        eval_metrics = evaluate_config(cfg, checkpoint_path, args)
        swap_metrics = swap_config(cfg, checkpoint_path, args)
        row = extract_primary_metrics(cfg['name'], eval_metrics, swap_metrics)
        summary_rows.append(row)

    # Aggregate
    summary_path = Path(args.results_base) / 'ablation_summary.json'
    aggregate(summary_path, summary_rows)

    print("\nAblation complete.")
    print(f"Summary stored at: {summary_path}")


if __name__ == '__main__':
    main()
