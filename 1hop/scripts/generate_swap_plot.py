import json
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Directory containing the layer folders
base_dir = '/home/mlyang721/bio_data/results/sae_per_layer'

# List of layers
layers = [f'layer{i}' for i in range(12)]

# Alphas to plot
alphas = ['1.0', '10.0', '100.0', '1000.0']

# Collect data
data = {}
for layer in layers:
    layer_data = {}
    file_path = os.path.join(base_dir, layer, 'swap_controllability_detailed.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            swap_data = json.load(f)
        for alpha in alphas:
            if alpha in swap_data:
                layer_data[float(alpha)] = swap_data[alpha]['overall_success_rate']
            else:
                layer_data[float(alpha)] = 0.0  # or None
    data[int(layer[5:])] = layer_data  # layer number

# Sort layers
sorted_layers = sorted(data.keys())

# Prepare data for heatmap
heatmap_data = np.zeros((len(sorted_layers), len(alphas)))
layer_labels = [f'Layer {l}' for l in sorted_layers]
alpha_labels = alphas

for i, layer in enumerate(sorted_layers):
    for j, alpha in enumerate(alphas):
        heatmap_data[i, j] = data[layer].get(float(alpha), 0.0)

# Plot heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', 
            xticklabels=alpha_labels, yticklabels=layer_labels, ax=ax,
            cbar_kws={'label': 'Swap Success Rate'})

ax.set_title('Swap Controllability Heatmap: Layers vs Alpha Values', fontsize=14)
ax.set_xlabel('Alpha (Feature Amplification)', fontsize=12)
ax.set_ylabel('Transformer Layer', fontsize=12)

plt.tight_layout()
plt.savefig('swap_controllability_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("Swap controllability heatmap saved.")
