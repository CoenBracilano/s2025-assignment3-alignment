import json
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams

def gms8k_parser(model_output: str)-> str | None:
    if not isinstance(model_output, str):
        return None
    
    matches = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if matches:
        return matches[-1]
    return None

# 1 Load examples
def load_gsm8k(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    

#2 Format as strings
def format_prompt(example: dict) -> str:
    return f"{example['question']}\nAnswer:"

#3 Generate outputs
def generate_responses(llm: LLM, prompts: List[str], batch_size: int = 8) -> List[str]:
    outputs: List[str] = []
    start = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(
            batch,
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, stop=["\n"]),
        )
        outputs.extend([r.outputs[0].text.strip() for r in results])

    elapsed = time.time() - start
    print(f"Generated {len(prompts)} examples in {elapsed:.2f}s {len(prompts)/elapsed:.2f} ex/s")
    return outputs


#4 Evaluate predictions
def evaluate(examples: List[dict], gens: List[str]) -> float:
    correct = 0
    for ex, gen in zip(examples, gens):
        gold_num: Optional[str] = gms8k_parser(ex["answer"])
        pred_num: Optional[str] = gms8k_parser(gen)
        ex["model_output"] = gen
        ex["gold_number"] = gold_num
        ex["predicted"] = pred_num
        ex["correct"] = (pred_num == gold_num)
        correct += ex["correct"]
    return correct / len(examples)

#5 searlize manage, ect
def main():
    data_path = Path("data/gsm8k/test.jsonl")
    output_path = Path("results/gsm8k_qwen_eval.jsonl")
    model_path = "/workspace/cs336_alignment/models/Qwen/Qwen2.5-0.5B"

    print("Loading GSM8K examples")
    examples = load_gsm8k(data_path)
    prompts = [format_prompt(e) for e in examples]

    print(f"Loaded {len(examples)} examples. Initializing model")
    llm = LLM(model=model_path, dtype="float32")
    generations = generate_responses(llm, prompts)

    print("Evaluating predictions")
    acc = evaluate(examples, generations)
    print(f"Accuracy: {acc:.2%}")

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
            if ex.get("correct") is False and ex.get("predicted") is not None:
                unparsed.append(ex)

    print(f"Total examples: {total}")
    print(f"Unparsed generations: {len(unparsed)}")
    print(f"Percentage unparsed: {len(unparsed) / total:.2%}")
    return unparsed

if __name__ == "__main__":
    # main()
    unparsed = count_unparsed_generations(Path("results/gsm8k_qwen_eval.jsonl"))
    for i in range(10):
        print(unparsed[random.randint(0,len(unparsed))])
        print("\n")
