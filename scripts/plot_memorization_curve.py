import matplotlib.pyplot as plt
import json
from pathlib import Path

# Checkpoint steps and corresponding result files
steps = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
result_dir = Path("results/memorization_eval")

id_acc = []
ood_acc = []

for step in steps:
    fname = result_dir / f"memorization_results_step{step}.json"
    with open(fname, "r") as f:
        res = json.load(f)
    tag = f"step{step}"
    id_acc.append(res[tag]["ID"]["accuracy"])
    ood_acc.append(res[tag]["OOD"]["accuracy"])

plt.figure(figsize=(8,5))
plt.plot(steps, id_acc, marker='o', label='ID (train templates)')
plt.plot(steps, ood_acc, marker='s', label='OOD (OOD templates)')
plt.xlabel("SFT Checkpoint Step")
plt.ylabel("Accuracy")
plt.title("Memorization Accuracy vs. SFT Checkpoint")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/memorization_eval/memorization_curve_newID.png")
plt.show()
