import csv
import json
import re
import time
import random
from pathlib import Path
from typing import Any, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Updated for Fine-Tuned Model ===
def mmlu_parser(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    if not isinstance(model_output, str):
        return None
    model_output = model_output.strip()

    match = re.search(r'the correct answer is\s+([A-D])\b', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.fullmatch(r'[A-D]', model_output.strip(), re.IGNORECASE)
    if match:
        return match.group(0).upper()

    match = re.search(r'\b([A-D])\b', model_output[-10:], re.IGNORECASE)
    if match:
        return match.group(1).upper()
  
    return None

def load_mmlu_val_csvs(val_dir: Path) -> List[Dict]:
    examples = []
    for csv_file in sorted(val_dir.glob("*.csv")):
        subject = csv_file.stem.replace("_", " ")
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue
                question, a, b, c, d, answer = row
                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": [a, b, c, d],
                    "answer": answer.strip().upper(),
                })
    return examples

def format_prompt(example: dict) -> str:
    return (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"Answer the following multiple choice question about {example['subject']}.\n"
        f"Respond with a single sentence of the form \"The correct answer is _\" using A, B, C, or D.\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['options'][0]}\n"
        f"B. {example['options'][1]}\n"
        f"C. {example['options'][2]}\n"
        f"D. {example['options'][3]}\n"
        f"### Response:"
    )

def generate_responses(model, tokenizer, prompts: List[str], device: str, batch_size: int = 1) -> List[str]:
    all_outputs = []
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded)
        print(f"Generated {i}/{len(prompts)} prompts")

    elapsed = time.time() - start_time
    throughput = len(prompts) / elapsed
    print(f"Generation took {elapsed:.2f} seconds for {len(prompts)} examples")
    print(f"Throughput: {throughput:.2f} examples/second")

    return all_outputs

def evaluate(examples: List[dict], generations: List[str]) -> float:
    correct = 0
    for ex, gen in zip(examples, generations):
        pred = mmlu_parser(ex, gen)
        ex["model_output"] = gen
        ex["predicted"] = pred
        ex["correct"] = pred == ex["answer"]
        correct += ex["correct"]
    return correct / len(examples)

def main():
    val_dir = Path("data/mmlu/val")
    output_path = Path("results/mmlu_sft_qwen_eval.jsonl")
    model_dir = "/workspace/cs336_alignment/checkpoints/checkpoints-files/checkpoints"

    print("Loading examples")
    examples = load_mmlu_val_csvs(val_dir)
    prompts = [format_prompt(e) for e in examples]

    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda")
    model.eval()

    print("Generating responses")
    generations = generate_responses(model, tokenizer, prompts, device="cuda")

    print("Evaluating")
    acc = evaluate(examples, generations)
    print(f"Validation Accuracy: {acc:.2%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()
