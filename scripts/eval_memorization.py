"""
Evaluate SFT model memorization: For each checkpoint, test on all QA pairs (ID and OOD) and report accuracy.
"""
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# List of SFT checkpoints to evaluate
CHECKPOINTS = [
    f"models/new_id_split/checkpoint-step-{step}" for step in range(10000, 90000, 10000)
] + ["models/new_id_split/final"]

QA_SPLITS = {
    "ID": "data/generated/qa_train_id.jsonl",
    "OOD": "data/generated/qa_test_ood.jsonl"
}

RESULTS_DIR = Path("results/memorization_eval")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def load_qa(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def get_answer_from_output(output, prompt):
    # Remove prompt from output, strip whitespace, take first line
    return output[len(prompt):].strip().split("\n")[0]

def evaluate_checkpoint(ckpt_path, tokenizer, model, qa_list, device):
    correct = 0
    for qa in tqdm(qa_list, desc=f"Evaluating {ckpt_path}"):
        prompt = qa["question"]
        gold = qa["answer"].strip()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = get_answer_from_output(output, prompt)
        if pred == gold:
            correct += 1
    return correct, len(qa_list)


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_checkpoint', type=str, default=None, help='If set, only evaluate this checkpoint (name, e.g. checkpoint-step-10000 or final)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    checkpoints = CHECKPOINTS
    if args.only_checkpoint:
        # Accept either full path or just name
        if args.only_checkpoint == 'final':
            checkpoints = ["models/new_id_split/final"]
        else:
            checkpoints = [f"models/new_id_split/{args.only_checkpoint}"]
    for ckpt in checkpoints:
        tag = Path(ckpt).name.replace("checkpoint-step-", "step") if "checkpoint" in ckpt else "final"
        print(f"\n=== Evaluating SFT checkpoint: {ckpt} ===")
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
        model.eval()
        results[tag] = {}
        for split, qa_path in QA_SPLITS.items():
            qa_list = load_qa(qa_path)
            correct, total = evaluate_checkpoint(ckpt, tokenizer, model, qa_list, device)
            acc = correct / total
            results[tag][split] = {"correct": correct, "total": total, "accuracy": acc}
            print(f"{split}: {correct}/{total} = {acc:.4f}")
        # Free memory
        del model
        torch.cuda.empty_cache()
    # Save results
    outname = f"memorization_results_{tag}.json" if args.only_checkpoint else "memorization_results.json"
    with open(RESULTS_DIR / outname, "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll checkpoints evaluated. Results saved.")

if __name__ == "__main__":
    main()
