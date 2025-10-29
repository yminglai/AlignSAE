import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import cm
import numpy as np

# Directory containing the layer folders
base_dir = '/home/mlyang721/bio_data/models/sae_per_layer'

# List of layers
layers = [f'layer{i}' for i in range(12)]

# Function to load data from a layer
def load_layer_data(layer):
    file_path = os.path.join(base_dir, layer, 'training_history.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Combined plot for all layers
# Load all data
all_data = {}
for layer in layers:
    all_data[layer] = load_layer_data(layer)

# Metrics to plot
metric_names = ['loss', 'L_recon', 'L_sparse', 'L_align', 'L_indep', 'L_ortho', 'L_value', 'slot_acc', 'value_acc']

# Colormap for layers
colors = cm.tab20.colors[:len(layers)]  # Use tab20 for distinct colors

# Create 3x3 subplot
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Combined Training History for All Layers (Outliers Handled)', fontsize=16)
axes = axes.flatten()

for i, metric in enumerate(metric_names):
    ax = axes[i]
    all_values = []
    for j, layer in enumerate(layers):
        data = all_data[layer]
        steps = list(range(len(data)))
        values = [entry[metric] for entry in data]
        all_values.extend(values)
        
        # Smooth using rolling mean
        df = pd.DataFrame({'steps': steps, 'values': values})
        smoothed = df['values'].rolling(window=10, center=True).mean()
        
        # Plot
        ax.plot(steps, smoothed, label=layer, color=colors[j], linewidth=1.5)
    
    ax.set_title(metric)
    ax.grid(True)
    if i >= 6:
        ax.set_xlabel('Training Steps')
    if metric in ['slot_acc', 'value_acc']:
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)  # Clip accuracies to 0-1
    else:
        ax.set_ylabel('Loss')
        # Handle outliers: use log scale for wide ranges, and clip to 95th percentile
        if metric in ['L_indep', 'L_value', 'loss']:
            ax.set_yscale('log')
            # Clip to 95th percentile to remove extreme outliers
            percentile_95 = np.percentile([v for v in all_values if v > 0], 95)
            ax.set_ylim(bottom=1e-10, top=percentile_95 * 10)  # Adjust as needed
        else:
            # For other losses, clip to 95th percentile
            percentile_95 = np.percentile(all_values, 95)
            ax.set_ylim(0, percentile_95 * 1.1)
    
    # Only show legend for first subplot to avoid clutter
    if i == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('combined_training_history.png', dpi=150, bbox_inches='tight')
plt.close()

print("Combined plot saved.")

# Plot final results from pipeline_summary.json
# Load the summary
with open('/home/mlyang721/bio_data/results/sae_per_layer/pipeline_summary.json', 'r') as f:
    summary = json.load(f)

layer_results = summary['layer_results']
layers_num = [int(layer) for layer in layer_results.keys()]  # Convert to int for sorting
layers_num.sort()
layers_str = [str(l) for l in layers_num]
metrics = ['train_slot_acc', 'test_ood_slot_acc', 'diagonal_acc', 'swap_success', 'reconstruction_mse']

# Prepare data
data = {metric: [layer_results[str(layer)][metric] for layer in layers_num] for metric in metrics}

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Final SAE Evaluation Metrics Across Layers (Outliers Handled)', fontsize=16)
axes = axes.flatten()

colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(layers_num, data[metric], marker='o', color=colors[i], linewidth=2, markersize=6)
    ax.set_title(metric)
    ax.set_xlabel('Layer')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.grid(True)
    ax.set_xticks(layers_num)
    ax.set_xticklabels(layers_str)
    
    # Handle outliers for reconstruction_mse
    if metric == 'reconstruction_mse':
        ax.set_yscale('log')

# Hide the last subplot if not used
if len(metrics) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig('final_evaluation_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

print("Final results plot saved.")

# Sample plot: Layer 6 Training History
layer = 'layer6'
data_layer = load_layer_data(layer)

# Extract metrics
steps = list(range(len(data_layer)))
metrics_dict = {
    'loss': [entry['loss'] for entry in data_layer],
    'L_recon': [entry['L_recon'] for entry in data_layer],
    'L_sparse': [entry['L_sparse'] for entry in data_layer],
    'L_align': [entry['L_align'] for entry in data_layer],
    'L_indep': [entry['L_indep'] for entry in data_layer],
    'L_ortho': [entry['L_ortho'] for entry in data_layer],
    'L_value': [entry['L_value'] for entry in data_layer],
    'slot_acc': [entry['slot_acc'] for entry in data_layer],
    'value_acc': [entry['value_acc'] for entry in data_layer]
}

# Create 3x3 subplot
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f'Training History for {layer} (Sample Plot)', fontsize=16)

# Flatten axes for easy iteration
axes = axes.flatten()

for i, (metric_name, values) in enumerate(metrics_dict.items()):
    ax = axes[i]
    ax.plot(steps, values, label=metric_name)
    ax.set_title(metric_name)
    ax.grid(True)
    if i >= 6:  # Bottom row
        ax.set_xlabel('Training Steps')
    if metric_name in ['slot_acc', 'value_acc']:
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel('Value')
        # Handle outliers
        if metric_name in ['L_indep', 'L_value', 'loss']:
            ax.set_yscale('log')
            # Clip to 95th percentile
            percentile_95 = np.percentile([v for v in values if v > 0], 95)
            ax.set_ylim(bottom=1e-10, top=percentile_95 * 10)
        else:
            percentile_95 = np.percentile(values, 95)
            ax.set_ylim(0, percentile_95 * 1.1)

plt.tight_layout()
plt.savefig('layer6_training_history.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Sample plot for {layer} saved.")