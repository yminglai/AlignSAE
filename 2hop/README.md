# 2-Hop Reasoning Training Pipeline

This directory contains the training pipeline for a 2-hop reasoning task with 1-to-1 SAE interface support.

## Overview

The model learns to predict two entity tokens (entity_2, entity_3) given either:
1. A question (QA mode)
2. Relation sentences (Paragraph mode)
3. Both paragraph and questions (Mixed mode - **recommended**)

## Data Usage

### Training Set
- Uses `train_two_hop_qa_data.jsonl`
- **Mixed mode** (default): 100% paragraph samples + 50% QA samples
- Paragraphs: All relation sentences → standard language modeling
- QA: Questions → predict exactly 2 entity tokens

### Validation Set
- Uses `val_two_hop_qa_data.jsonl`
- **During training**: Only sentences participate as paragraph samples (language modeling)
- **During evaluation**: Questions used for evaluation (QA mode)
- This ensures validation sentences contribute to learning while questions remain for testing

## Training Modes

### 1. Mixed Mode (`--mode mixed`) **[RECOMMENDED]**

**Training data (train set):**
- 100% of examples as paragraph samples (all relation sentences) → **Language modeling**
- 50% of examples also as QA samples (configurable via `--qa_ratio`) → **Predict 2 entity tokens**

**Training data (validation set):**
- 100% of examples as paragraph samples (sentences only) → **Language modeling**
- 0% QA samples (questions not used for training)

**Evaluation data (validation set):**
- 100% QA samples for consistent evaluation

This creates a rich training signal where the model learns:
- From paragraphs: standard language modeling over relation sentences (predict all tokens)
- From QA: how to answer questions with exactly 2 entity tokens
- Validation sentences participate in training but questions are held out for testing

**Example training sample (paragraph - language modeling):**
```
Dominic needs to send regular updates to Avery. Dominic works under the supervision of Avery. Dominic is accountable to Avery on all tasks. Dominic attends one-on-one meetings with Avery to discuss progress. Dominic escalates issues to Avery when problems arise. Dominic consults Avery before making major decisions. Dominic files weekly performance reports with Avery. Dominic has Avery as the main decision-maker on the team. Gerald and Dominic are best pals. Everyone knows that Gerald is a close friend of Dominic. Gerald hangs out a lot with Dominic. Gerald confides in Dominic about personal matters. Gerald has maintained a strong friendship with Dominic. People often see Gerald and Dominic spending weekends together. Gerald has known Dominic since childhood and remains very close. Gerald trusts Dominic deeply as a friend.
```
→ Standard next-token prediction for all tokens

**Example training sample (QA - 2 token prediction):**
```
Question: Who is the friend of the report of Avery?
Answer:DominicGerald
```
→ Only supervise the 2 answer tokens### 2. QA Mode (`--mode qa`)

**Input format:**
```
Question: Who is the friend of the report of Avery?
Answer:
```

**Output:** Exactly two entity tokens (no space)
```
DominicGerald
```

The model learns direct question-answering. Only the 2 answer tokens are supervised.

**Run training:**
```bash
# Modify run_train_two_hop.sh to set MODE="qa"
```

### 3. Paragraph Mode (`--mode paragraph`)

**Input format:**
```
Dominic needs to send regular updates to Avery. Dominic works under the supervision of Avery. 
Gerald and Dominic are best pals. Everyone knows that Gerald is a close friend of Dominic.
... (all 16 sentences)
```

**Output:** Standard language modeling (predict all tokens)

The model learns from relation sentences with standard next-token prediction. All tokens in the sequence are supervised.

**Run training:**
```bash
bash run_train_paragraph.sh
```

## Data Format

Input JSONL files (`train_two_hop_qa_data.jsonl`, `val_two_hop_qa_data.jsonl`) contain:

```json
{
  "question_template": "Who is the [RELATION_2] of the [RELATION_1] of [ENTITY_1]",
  "entity_1": "Avery",
  "relation_1": "reports_to",
  "relation_2": "friend_of",
  "entity_2": "Dominic",
  "entity_3": "Gerald",
  "question": "Who is the friend of the report of Avery?",
  "output": "Dominic Gerald",
  "relation_1_sentences": [...],
  "relation_2_sentences": [...]
}
```

## Minimal Commands

- Generate / convert data
```bash
python 2hop/_gen_data/generate_two_hop.py --path 2hop/_dataset/_org/train.jsonl
python 2hop/_gen_data/generate_two_hop.py --path 2hop/_dataset/_org/val.jsonl
python 2hop/split_dataset.py   # optional 4k downsample
```

- Train 2-hop LM (mixed)
```bash
bash 2hop/run_train_two_hop.sh
```

- End-to-end SAE pipeline
```bash
bash 2hop/run_full_pipeline.sh
```

- Swap demo (supervised SAE)
```bash
bash 2hop/run_swap_layer6.sh
```

Other helper shell scripts are ignored by git to keep the repo clean.

## Model Architecture

- Base: GPT-2 architecture
- Modified with:
  - 6 transformer layers (reduced from 12 for faster training)
  - 8 attention heads
  - 512 hidden dimensions
  - Custom vocabulary with entity tokens

## Output

Training produces:

1. **Checkpoints** (saved every epoch):
   - `checkpoint_epoch_N/` - Model weights, tokenizer, optimizer state
   - `best_model/` - Best model based on validation loss

2. **Logs**:
   - `training_log.csv` - Step-by-step training metrics
   - Columns: step, epoch, split, loss, acc_1hop, acc_2hop, lr

3. **Plots** (auto-generated):
   - `plots/loss_curve.png` - Training/validation loss over steps
   - `plots/val_accuracy_curve.png` - 1-hop and 2-hop accuracy
   - `plots/combined_curves.png` - Combined loss and accuracy

## Evaluation Metrics

- **loss**: Cross-entropy loss on target tokens
- **acc_1hop**: Accuracy of predicting entity_2 (first token)
- **acc_2hop**: Accuracy of predicting entity_3 (second token)

## SAE Interface

The model includes hooks for 1-to-1 SAE integration:

```python
# Extract activations at a specific layer
activations = get_layer_activations(model, input_ids, attention_mask, layer_idx=3)
# Returns: [batch_size, seq_len, hidden_dim]
```

This allows you to:
- Extract hidden states at any transformer layer
- Attach SAE models that map relations to specific feature slots
- Analyze learned representations

## Visualization

After training, generate plots:

```bash
python plot_curves.py \
    --log_file _trained_model/training_log.csv \
    --output_dir _trained_model/plots
```

## Notes

- Entity tokens must be single tokens in the tokenizer (already configured in `_base_model/`)
- **QA mode**: The model predicts exactly 2 tokens (entity_2, entity_3) with no space between them
- **Paragraph mode**: Standard language modeling - all tokens are supervised and predicted
- **Mixed mode**: Combines both approaches - paragraph for general understanding, QA for targeted 2-token prediction
- Validation always uses QA mode to test the model's ability to answer questions with 2 entity tokens

## Example Training Flow

1. **Mixed Mode Training (Recommended):**
```bash
bash run_train_two_hop.sh
# Training: 100% paragraph + 50% QA
# Validation: 100% QA
```

2. **Paragraph-Only Training:**
```bash
bash run_train_paragraph.sh
# Model learns: Relation sentences -> Entity tokens
```

3. **Compare Results:**
Check `training_log.csv` in both output directories to compare learning curves.

## Training Strategy

The **mixed mode** is recommended because:
- Model learns relational structure from paragraphs (100% coverage)
- Model learns question-answering from QA samples (50% coverage)
- Validation uses pure QA to test generalization
- This mimics a realistic scenario where the model sees both contexts and queries
