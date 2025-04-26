import csv
import json
import re
import time
import random
from pathlib import Path
from typing import Any, List, Dict
from vllm import LLM, SamplingParams

# from cs336_alignment.functions import mmlu_parser


#Section 3.2 MMLU Baseline script for part B
def mmlu_parser(mmlu_example: dict[str, Any], model_output: str)-> str | None:
    if not isinstance(model_output, str):
        return None

    model_output = model_output.strip()

    # Try to match "The correct answer is X"
    match = re.search(r'the correct answer is\s+([A-D])\b', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: exact single letter A–D
    match = re.fullmatch(r'[A-D]', model_output.strip(), re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # Bonus fallback: any standalone A–D in the last few tokens
    match = re.search(r'\b([A-D])\b', model_output[-10:], re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

# 1 Load our MMLU examples from the val folder 
def load_mmlu_val_csvs(val_dir: Path) -> List[Dict]:
    examples = []
    for csv_file in sorted(val_dir.glob("*.csv")):
        subject = csv_file.stem.replace("_", " ")
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue  # skip malformed lines
                question, a, b, c, d, answer = row
                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": [a, b, c, d],
                    "answer": answer.strip().upper(),
                })
    return examples

# 2 Format the examples into string prompts for the language model 
def format_prompt(example: dict) -> str:
    return (
        f"Answer the following multiple choice question about {example['subject']}\n"
        f"Respond with a single sentence of the form \"The correct answer is _\", \n"
        f"filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['options'][0]}\n"
        f"B. {example['options'][1]}\n"
        f"C. {example['options'][2]}\n"
        f"D. {example['options'][3]}\n"
        f"Answer:"
    )
# 3, Generate outputs for each example
def generate_responses(llm: LLM, prompts: List[str], batch_size: int = 8) -> List[str]:
    all_outputs = []
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, SamplingParams(temperature=0.0, max_tokens=32, stop=["\n"]))
        all_outputs.extend([out.outputs[0].text.strip() for out in outputs])

    elapsed = time.time() - start_time
    throughput = len(prompts) / elapsed
    print(f"Generation took {elapsed:.2f} seconds for {len(prompts)} examples")
    print(f"Throughput: {throughput:.2f} examples/second")

    return all_outputs

# 4, Evaluation 
def evaluate(examples: List[dict], generations: List[str]) -> float:
    correct = 0
    for ex, gen in zip(examples, generations):
        pred = mmlu_parser(ex, gen)
        ex["model_output"] = gen
        ex["predicted"] = pred
        ex["correct"] = pred == ex["answer"]
        correct += ex["correct"]
    return correct / len(examples)

# 5 Running everything, storing outputs printing status updates
def main():
    val_dir = Path("data/mmlu/val")
    output_path = Path("results/mmlu_qwen_eval.jsonl")
    model_path = "/workspace/cs336_alignment/models/Qwen/Qwen2.5-0.5B"

    print("Loading MMLU validation set...")
    examples = load_mmlu_val_csvs(val_dir)
    prompts = [format_prompt(e) for e in examples]

    print(f"Loaded {len(examples)} examples. Initializing model...")
    llm = LLM(model=model_path, dtype="float32")
    generations = generate_responses(llm, prompts)

    print("Evaluating...")
    acc = evaluate(examples, generations)
    print(f"Validation Accuracy: {acc:.2%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def count_unparsed_generations(path: str):
    unparsed = []
    total = 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            ex = json.loads(line)
            if ex.get("correct") is False:
                unparsed.append(ex)

    print(f"Total examples: {total}")
    print(f"Unparsed generations: {len(unparsed)}")
    print(f"Percentage unparsed: {len(unparsed) / total:.2%}")
    return unparsed


 
if __name__ == "__main__":
    main()
    unparsed = count_unparsed_generations(Path("results/mmlu_qwen_eval.jsonl"))
    for i in range(10):
        print(unparsed[random.randint(0,len(unparsed))])
        print("\n")
    