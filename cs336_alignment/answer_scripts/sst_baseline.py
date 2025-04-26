import csv
import json
import time
from pathlib import Path
from typing import List, Dict
from vllm import LLM, SamplingParams


# Load Prompts, avoid the first line since its not an example just a header
def load_safety_prompts(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("prompts_final")]
    
# Generate responses using qwen
def generate_responses(llm: LLM, prompts: List[str], batch_size: int = 4) -> List[str]:
    outputs = []
    start_time = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        results = llm.generate(batch, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, stop=["\n"]))
        outputs.extend([r.outputs[0].text.strip() for r in results])
    elapsed = time.time() - start_time
    print(f"Generation took {elapsed:.2f}s for {len(prompts)} examples → {len(prompts)/elapsed:.2f} examples/s")
    return outputs

# Main controller
def main():
    input_path = Path("data/simple_safety_tests/simple_safety_tests.csv")
    output_path = Path("results/safety_qwen_outputs.jsonl")
    model_path = "/workspace/cs336_alignment/models/Qwen/Qwen2.5-0.5B"

    print("Loading prompts from SimpleSafetyTests CSV")
    examples = load_safety_prompts(input_path)
    prompts = [ex["prompts_final"] for ex in examples]

    print(f"Loaded {len(prompts)} prompts. Initializing model")
    llm = LLM(model=model_path, dtype="float32", enforce_eager=True)

    print("Generating responses")
    generations = generate_responses(llm, prompts, batch_size=1)

    print("Saving responses in JSONL format")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prompt, output in zip(prompts, generations):
            json.dump({"prompts_final": prompt, "output": output}, f)
            f.write("\n")

    print("Complete")


if __name__ == "__main__":
    main()
