"""
Generate OOD questions using OpenRouter API with few-shot prompting.
Creates diverse paraphrases that differ from ID templates for distribution shift testing.
"""
import json
import os
import time
from pathlib import Path
import requests
from tqdm import tqdm
from multiprocessing import Process, Queue
import sys

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(prompt, model="anthropic/claude-3.5-sonnet", max_tokens=150, temperature=0.9):
    """Call OpenRouter API with the given prompt."""
    if not OPENROUTER_API_KEY:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yminglai/AlignSAE",
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    output = response.json()["choices"][0]["message"]["content"].strip()
    return output

def create_few_shot_prompt(rule_name, full_name, id_examples):
    """Create a few-shot prompt for generating OOD questions."""
    
    rule_descriptions = {
        "birth_date": "the person's birth date",
        "birth_city": "the city where the person was born",
        "university": "the university the person attended",
        "major": "the person's academic major/field of study",
        "employer": "the company the person works for",
        "company_city": "the city where the person works"
    }
    
    prompt = f"""You are helping generate simple question variations for a dataset. Your task is to create 2 NEW questions that ask about {rule_descriptions[rule_name]} for the person "{full_name}".

IMPORTANT REQUIREMENTS:
1. Keep questions SIMPLE and DIRECT - similar to these training examples but with SLIGHT variations:
{chr(10).join(f'   - "{ex}"' for ex in id_examples)}

2. Make SMALL changes only - change just 1-2 words or reorder slightly
3. Questions must be answerable with a direct factual answer (date, city, university name, etc.)
4. DO NOT ask questions that require explanation or opinion
5. Use the exact name "{full_name}" in each question
6. Keep questions SHORT (under 15 words)

Generate exactly 2 simple question variations, one per line. Do NOT number them or add any other text.
"""
    
    return prompt

def validate_question(question, full_name):
    """Validate that the question contains the person's full name."""
    # Check if the full name appears in the question
    if full_name not in question:
        return False
    
    # Check it's actually a question (ends with ? or is reasonable length)
    if len(question) < 10 or len(question) > 200:
        return False
    
    return True

def generate_ood_questions_for_person(person_data, rule_name, id_templates, max_retries=5):
    """Generate 2 OOD questions for a specific person and rule.
    
    This generates UNIQUE questions for this specific person-relation pair,
    NOT template-based questions that would be the same for all persons.
    """
    
    # Create few-shot examples from ID templates
    id_examples = [tmpl.format(FULL_NAME=person_data["full_name"]) for tmpl in id_templates[:2]]
    full_name = person_data["full_name"]
    
    for attempt in range(max_retries):
        try:
            prompt = create_few_shot_prompt(rule_name, full_name, id_examples)
            response = call_openrouter(prompt, temperature=0.9)
            
            # Parse response - should be 2 lines
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Filter out any numbered or prefixed questions
            questions = [q.lstrip('0123456789.-) ') for q in questions]
            
            # Validate each question contains the person's name
            valid_questions = [q for q in questions if validate_question(q, full_name)]
            
            # We need exactly 2 valid questions
            if len(valid_questions) >= 2:
                return valid_questions[:2]
            
            # If we didn't get enough valid questions, retry
            print(f"  Warning: Got {len(valid_questions)}/{len(questions)} valid questions (attempt {attempt+1}), retrying...")
            
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                pass
            else:
                raise
    
    raise ValueError(f"Failed to generate valid questions for {full_name} - {rule_name} after {max_retries} attempts")

def process_rule(rule_idx, rule_name, all_persons, qa_templates, rule_to_person_key, data_dir, completed_pairs_set):
    """Process a single rule for all persons in a separate process."""
    checkpoint_file = data_dir / f"qa_test_ood_light_checkpoint_{rule_name}.jsonl"
    
    # Filter to only persons not completed for this rule
    persons_to_process = [p for p in all_persons if (p['person_id'], rule_name) not in completed_pairs_set]
    
    if not persons_to_process:
        print(f"[{rule_name}] All persons already completed, skipping...")
        return
    
    failed_count = 0
    success_count = 0
    
    # Create progress bar for this rule with specific position
    pbar = tqdm(
        total=len(persons_to_process),
        desc=f"{rule_name:15}",
        position=rule_idx,
        leave=True
    )
    
    for person in persons_to_process:
        try:
            # Generate 2 UNIQUE diverse OOD questions for THIS specific person-relation pair
            ood_questions = generate_ood_questions_for_person(
                person, 
                rule_name, 
                qa_templates[rule_name]
            )
            
            person_key = rule_to_person_key[rule_name]
            answer = person[person_key]
            
            # Verify questions contain the person's name (double-check)
            for question in ood_questions:
                if person["full_name"] not in question:
                    raise ValueError(f"Generated question missing person name: {question}")
            
            # Save to checkpoint immediately
            with open(checkpoint_file, "a") as f:
                for template_idx, question in enumerate(ood_questions, start=2):
                    qa_entry = {
                        "person_id": person["person_id"],
                        "full_name": person["full_name"],
                        "rule_idx": rule_idx,
                        "rule_name": rule_name,
                        "question": question,
                        "answer": answer,
                        "template_idx": template_idx,
                        "split": "test_ood",
                        "is_ood": True
                    }
                    f.write(json.dumps(qa_entry) + "\n")
            
            success_count += 1
            pbar.update(1)
            
        except Exception as e:
            failed_count += 1
            pbar.write(f"[{rule_name}] Error for {person['person_id']}: {str(e)[:50]}")
            continue
    
    pbar.close()
    print(f"[{rule_name}] COMPLETED: {success_count} succeeded, {failed_count} failed")

def main():
    # Paths
    data_dir = Path("data/generated")
    qa_templates_dir = Path("data/qa_templates")
    output_file = data_dir / "qa_test_ood_light.jsonl"
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    with open(data_dir / "train_kg.json") as f:
        train_kg = json.load(f)
    
    with open(data_dir / "test_kg.json") as f:
        test_kg = json.load(f)
    
    all_persons = train_kg + test_kg
    print(f"Loaded {len(all_persons)} persons")
    
    # Load ID templates (templates 0-1 are used for training)
    qa_templates = {
        "birth_date": [line.strip() for line in open(qa_templates_dir / "birth_date_questions.txt") if line.strip()],
        "birth_city": [line.strip() for line in open(qa_templates_dir / "birth_city_questions.txt") if line.strip()],
        "university": [line.strip() for line in open(qa_templates_dir / "university_questions.txt") if line.strip()],
        "major": [line.strip() for line in open(qa_templates_dir / "major_questions.txt") if line.strip()],
        "employer": [line.strip() for line in open(qa_templates_dir / "employer_questions.txt") if line.strip()],
        "company_city": [line.strip() for line in open(qa_templates_dir / "company_city_questions.txt") if line.strip()],
    }
    
    rule_names = ["birth_date", "birth_city", "university", "major", "employer", "company_city"]
    rule_to_person_key = {
        "birth_date": "birth_date",
        "birth_city": "birth_city",
        "university": "university",
        "major": "major",
        "employer": "employer",
        "company_city": "work_city"
    }
    
    # Check for existing checkpoints from all rules and resume
    qa_test_ood = []
    completed_pairs = set()
    
    print(f"\nChecking for existing checkpoints...")
    for rule_name in rule_names:
        rule_checkpoint = data_dir / f"qa_test_ood_light_checkpoint_{rule_name}.jsonl"
        if rule_checkpoint.exists():
            with open(rule_checkpoint) as f:
                for line in f:
                    qa = json.loads(line)
                    qa_test_ood.append(qa)
                    completed_pairs.add((qa['person_id'], qa['rule_name']))
            print(f"  Loaded {rule_name}: {sum(1 for p in completed_pairs if p[1] == rule_name)} person-rule pairs")
    
    if completed_pairs:
        print(f"Total loaded: {len(qa_test_ood)} questions from {len(completed_pairs)} completed pairs\n")
    
    # Generate OOD questions - UNIQUE per person per relation - PARALLEL by rule
    total_to_generate = len(all_persons) * len(rule_names)
    
    print(f"\n{'='*70}")
    print(f"Generating UNIQUE OOD questions with PARALLEL processing")
    print(f"{'='*70}")
    print(f"Total: {len(all_persons)} persons × {len(rule_names)} rules = {total_to_generate} batches")
    print(f"Each batch generates 2 unique questions → {total_to_generate * 2} total questions")
    print(f"Running {len(rule_names)} parallel processes (one per rule)!\n")
    
    # Create a process for each rule
    processes = []
    for rule_idx, rule_name in enumerate(rule_names):
        p = Process(
            target=process_rule,
            args=(rule_idx, rule_name, all_persons, qa_templates, rule_to_person_key, data_dir, completed_pairs)
        )
        p.start()
        processes.append(p)
        print(f"Started process for rule: {rule_name}")
    
    # Wait for all processes to complete
    print(f"\nWaiting for all {len(processes)} processes to complete...")
    for p in processes:
        p.join()
    
    print(f"\n{'='*70}")
    print(f"All parallel processes completed!")
    print(f"{'='*70}\n")
    
    # Collect all results from checkpoint files
    qa_test_ood = []
    failed_generations = []
    
    print("Collecting results from all rules...")
    for rule_name in rule_names:
        rule_checkpoint = data_dir / f"qa_test_ood_light_checkpoint_{rule_name}.jsonl"
        if rule_checkpoint.exists():
            count = 0
            with open(rule_checkpoint) as f:
                for line in f:
                    qa = json.loads(line)
                    qa_test_ood.append(qa)
                    count += 1
            print(f"  {rule_name}: {count} questions")
    
    # Save final OOD questions
    print(f"\n{'='*70}")
    print(f"Saving {len(qa_test_ood)} OOD questions to {output_file}")
    with open(output_file, "w") as f:
        for qa in qa_test_ood:
            f.write(json.dumps(qa) + "\n")
    
    # Remove all checkpoint files on successful completion
    for rule_name in rule_names:
        rule_checkpoint = data_dir / f"qa_test_ood_light_checkpoint_{rule_name}.jsonl"
        if rule_checkpoint.exists():
            rule_checkpoint.unlink()
    print("All checkpoint files removed")
    
    print(f"\n{'='*70}")
    print(f"OOD question generation complete!")
    print(f"{'='*70}")
    print(f"Generated: {len(qa_test_ood)} questions")
    print(f"Expected:  {total_to_generate * 2} questions")
    print(f"Failed:    {len(failed_generations)} person-rule pairs")
    print(f"Success rate: {len(qa_test_ood)/(total_to_generate * 2)*100:.1f}%")
    
    # Validate complete coverage
    print(f"\n{'='*70}")
    print("Validating coverage...")
    rule_counts = {rule: 0 for rule in rule_names}
    person_counts = {}
    for qa in qa_test_ood:
        rule_counts[qa['rule_name']] += 1
        person_counts[qa['person_id']] = person_counts.get(qa['person_id'], 0) + 1
    
    print(f"Questions per rule:")
    for rule, count in rule_counts.items():
        expected = len(all_persons) * 2
        print(f"  {rule}: {count}/{expected} ({count/expected*100:.1f}%)")
    
    incomplete_persons = [p_id for p_id, count in person_counts.items() if count < len(rule_names) * 2]
    if incomplete_persons:
        print(f"\nWARNING: {len(incomplete_persons)} persons have incomplete questions!")
        print(f"First 5: {incomplete_persons[:5]}")
    else:
        print(f"\n✓ All {len(all_persons)} persons have complete question sets!")
    
    if failed_generations:
        print(f"\nFailed generations saved to: {data_dir / 'ood_generation_failures.json'}")
        with open(data_dir / "ood_generation_failures.json", "w") as f:
            json.dump(failed_generations, f, indent=2)
    
    # Validate uniqueness
    print(f"\n{'='*70}")
    print("Validating uniqueness...")
    unique_questions = set(qa['question'] for qa in qa_test_ood)
    print(f"Total questions: {len(qa_test_ood)}")
    print(f"Unique questions: {len(unique_questions)}")
    print(f"Uniqueness rate: {len(unique_questions)/len(qa_test_ood)*100:.1f}%")
    
    # Validate all questions contain the person's name
    missing_names = [qa for qa in qa_test_ood if qa['full_name'] not in qa['question']]
    print(f"Questions with missing names: {len(missing_names)}")
    if missing_names:
        print("WARNING: Some questions don't contain the person's name!")
        for qa in missing_names[:5]:
            print(f"  - {qa['question']} (should have {qa['full_name']})")
    
    # Show a few examples
    print(f"\n{'='*70}")
    print("Sample OOD questions:")
    print(f"{'='*70}")
    for qa in qa_test_ood[:5]:
        print(f"  {qa['rule_name']}: {qa['question']}")

if __name__ == "__main__":
    main()
