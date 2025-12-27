# OOD Question Generation with OpenRouter API

This script generates out-of-distribution (OOD) questions using the OpenRouter API to create diverse paraphrases that differ from in-distribution training templates.

## Setup

1. **Install dependencies:**
```bash
pip install requests tqdm
```

2. **Set your OpenRouter API key:**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

You can get an API key from: https://openrouter.ai/

## Usage

Run the script to generate OOD questions:

```bash
python scripts/generate_ood_questions_api.py
```

## What it does

1. **Loads knowledge graph** with all persons and their attributes
2. **For each person and each relation (6 rules)**:
   - Shows the LLM the 2 ID templates as examples to AVOID
   - Asks it to generate 2 completely different questions
   - Uses high temperature (0.9) for diversity
3. **Saves to** `data/generated/qa_test_ood.jsonl`

## Expected Output

- **Total questions**: ~12,000 (1,000 persons × 6 rules × 2 questions)
- **Cost estimate**: Varies by model, approximately:
  - Using `anthropic/claude-3.5-sonnet`: ~$3-5 for full dataset
  - Using cheaper models like `meta-llama/llama-3.1-8b-instruct:free`: Free but less diverse

## Customization

### Use a different model

Edit line 16 in the script:
```python
def call_openrouter(prompt, model="meta-llama/llama-3.1-8b-instruct:free", ...):
```

Available models at: https://openrouter.ai/models

### Adjust creativity

Change `temperature` parameter (line 17):
- Higher (0.9-1.0): More diverse/creative
- Lower (0.5-0.7): More conservative

### Rate limiting

Adjust sleep time on line 192:
```python
time.sleep(0.5)  # seconds between API calls
```

## Output Format

Each line in `qa_test_ood.jsonl`:
```json
{
  "person_id": "person_0000",
  "full_name": "John Doe",
  "rule_idx": 0,
  "rule_name": "birth_date",
  "question": "Could you inform me of the date John Doe came into this world?",
  "answer": "24,March,1964",
  "template_idx": 2,
  "split": "test_ood",
  "is_ood": true
}
```

## Troubleshooting

**Error: "Please set OPENROUTER_API_KEY environment variable"**
- Make sure you've exported your API key

**Rate limiting errors**
- Increase sleep time between requests
- Some models have higher rate limits

**Low quality questions**
- Try a more capable model (e.g., Claude or GPT-4)
- Adjust the prompt in `create_few_shot_prompt()`
