# Feature Co-Activation Analysis - What You'll Get

## Overview
This analysis creates **12 dense heatmaps** (one per layer) showing which of the 10,006 SAE features activate when asking questions about different relations.

## The Heatmaps

### For Each Layer (0-11):

#### 1. **Dense Activation Heatmap** (`layer{00-11}_dense_activation_heatmap.png`)
- **Size**: Very wide (24" x 8")
- **Y-axis (6 rows)**: The 6 relation types
  - Row 0: Birth Date questions
  - Row 1: Birth City questions  
  - Row 2: University questions
  - Row 3: Major questions
  - Row 4: Employer questions
  - Row 5: Work City questions
- **X-axis (10,006 columns)**: All SAE features
  - Columns 0-9,999: Free features
  - Columns 10,000-10,005: The 6 relation slots
- **Color**: Hot colormap (black=0, yellow/white=high activation)
- **Vertical cyan line**: Marks where relation slots start (at 10,000)

**What it shows**: When you ask about relation Y, which features light up?

#### 2. **Relation Slots Only** (`layer{00-11}_relation_slots_only.png`)
- Zoomed-in view showing only the 6 relation slots
- 6x6 matrix with actual numbers
- **Diagonal**: Should be bright (asking about relation X activates slot X)
- **Off-diagonal**: Should be dim (clean separation)

#### 3. **Top 50 Free Features** (`layer{00-11}_top50_free_features.png`)
- Shows only the 50 most active free features
- Identifies which free features are most important for each relation
- Each row shows a different relation's top features

### Cross-Layer Summary (`all_layers_relation_slots_summary.png`)
- 12-panel view showing all layers at once
- Quick comparison of relation slot separation across depth
- Helps identify which layers have cleanest 1-to-1 mappings

## Data Files

### Per Layer:
- `layer{00-11}_activation_matrix.npy`: Raw numpy array [6, 10006]
- `layer{00-11}_statistics.json`: Detailed stats including:
  - Target slot activation strength
  - Other slots mean activation
  - Top 100 features for each relation
  - Free features statistics

### Overall:
- `analysis_summary.json`: Configuration and summary statistics

## What to Look For

### Good Signs (Clean Disentanglement):
1. **Dense heatmap**: 
   - Each row has a distinct pattern
   - Clear vertical bright lines at specific feature positions
   - Relation slots (rightmost 6 columns) show diagonal pattern

2. **Relation slots only**:
   - Strong diagonal (>1.0)
   - Weak off-diagonal (<0.1)

3. **Top features**:
   - Different relations activate different top features
   - Little overlap in top-50 lists

### Warning Signs (Entangled Representations):
1. **Dense heatmap**:
   - All rows look similar (same features activate for all relations)
   - Diffuse activation (many features mildly active, no clear peaks)
   - Relation slots don't show diagonal pattern

2. **Relation slots only**:
   - Weak diagonal (<0.5)
   - Strong off-diagonal (>0.2)
   - Uniform activation (all slots ~equally active)

3. **Top features**:
   - Same features in top-50 for all relations
   - Very few features dominate all relations

## Expected Patterns Across Layers

### Early Layers (0-3):
- More distributed activation (many features active)
- Weaker relation slot separation
- Similar patterns across different relations

### Middle Layers (4-7):
- Gradual specialization
- Relation slots begin to separate
- Some distinct feature patterns emerge

### Late Layers (8-11):
- Cleanest separation (if training succeeded)
- Strong diagonal in relation slots
- Distinct feature patterns per relation
- Sparse activation (few strong features per relation)

## Example Analysis Workflow

```bash
# 1. Check overall summary
less results/feature_coactivation_analysis/all_layers_relation_slots_summary.png

# 2. Identify best layer (strongest diagonal)
# Look for layer with highest diagonal, lowest off-diagonal

# 3. Examine that layer's dense heatmap
# e.g., if layer 6 looks best:
less results/feature_coactivation_analysis/layer06_dense_activation_heatmap.png

# 4. Check top features for each relation
less results/feature_coactivation_analysis/layer06_top50_free_features.png

# 5. Load numerical data for analysis
python
>>> import numpy as np
>>> import json
>>> matrix = np.load('results/feature_coactivation_analysis/layer06_activation_matrix.npy')
>>> print(matrix.shape)  # (6, 10006)
>>> print(f"Diagonal: {np.diag(matrix[:, 10000:]).mean():.3f}")
>>> print(f"Off-diagonal: {(matrix[:, 10000:].sum() - np.diag(matrix[:, 10000:]).sum()) / 30:.3f}")
>>>
>>> with open('results/feature_coactivation_analysis/layer06_statistics.json') as f:
>>>     stats = json.load(f)
>>> for rel in stats:
>>>     print(f"{rel}: target={stats[rel]['target_slot_activation']:.3f}, "
>>>           f"others={stats[rel]['other_slots_mean']:.3f}")
```

## Key Questions Answered

1. **Which features encode each relation?**
   - Look at the vertical bright stripes in dense heatmap
   - Each relation's row shows its associated features

2. **Are features shared across relations?**
   - If vertical stripes appear in multiple rows, features are shared
   - Distinct rows = distinct feature sets = good disentanglement

3. **How many features are needed per relation?**
   - Count bright pixels in each row
   - Top-50 heatmap shows if few features dominate or many contribute

4. **Do the 6 relation slots work as intended?**
   - Check the rightmost 6 columns in dense heatmap
   - Should show diagonal pattern (6 bright pixels total)

5. **Which layer is best for interventions?**
   - Layer with strongest diagonal, distinct row patterns
   - Use for swap experiments and causal interventions

## Runtime
- **500 samples per relation** = 3,000 total questions
- **12 layers** to process
- **Estimated time**: 20-40 minutes on GPU
  - ~2-3 minutes per layer
  - Depends on GPU speed and batch processing

## Memory Requirements
- **SAE checkpoint**: ~500 MB per layer
- **Activation matrix**: ~250 KB per layer (6 × 10,006 × 4 bytes)
- **LM model**: ~500 MB (stays loaded across layers)
- **Total RAM**: ~2-3 GB
- **GPU VRAM**: ~2-4 GB

## Troubleshooting

### "No SAE checkpoint found"
→ Check `models/sae_per_layer/layer{X}/` contains `sae_best.pt` or `sae_final.pt`

### "Not enough QA samples"
→ Need at least 500 questions per relation in `qa_test_id.jsonl`
→ Use `--samples_per_relation 100` for smaller dataset

### Heatmaps too large
→ Reduce DPI: Edit script and change `dpi=200` to `dpi=100`

### Analysis too slow
→ Reduce samples: `--samples_per_relation 100`
→ Process fewer layers: `--layers 0,3,6,9,11`

## Next Steps After Analysis

1. **Identify best layer** from cross-layer summary
2. **Extract top features** for each relation
3. **Validate with interventions**: 
   - Ablate top features and check answer changes
   - Boost features and check controllability
4. **Paper figures**: Use dense heatmaps to show:
   - Learned structure is interpretable
   - Relations are separable (or not)
   - Evolution across layers
