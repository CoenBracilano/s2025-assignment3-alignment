import json
from pathlib import Path
import time
from typing import List, Dict
from vllm import LLM, SamplingParams




# 1 Load examples
def load_alpacaeval(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    
# 2 Format
def format_prompt(example: dict) -> str:
    return example["instruction"].strip()

# 3 Generate ouptuts
def generate_responses(llm: LLM, prompts: List[str], batch_size: int = 8) -> List[str]:
    outputs = []
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        results = llm.generate(
            batch,
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, stop=["\n"])
        )
        outputs.extend([r.outputs[0].text.strip() for r in results])

    elapsed = time.time() - start_time
    throughput = len(prompts) / elapsed
    print(f"Generation took {elapsed:.2f} seconds for {len(prompts)} examples")
    print(f"Throughput: {throughput:.2f} examples/second")

    return outputs


# Controller 
def main():
    input_path = Path("data/alpaca_eval/alpaca_eval.jsonl")
    output_path = Path("results/alpacaeval_qwen_output.json")
    model_path = "/workspace/cs336_alignment/models/Qwen/Qwen2.5-0.5B"
    generator_id = "qwen2.5-0.5b"

    print("Loading AlpacaEval set...")
    eval_set = load_alpacaeval(input_path)
    prompts = [format_prompt(ex) for ex in eval_set]

    print("Generating outputs...")
    llm = LLM(model=model_path, dtype="float32")
    generations = generate_responses(llm, prompts)

    print("Annotating examples...")
    for ex, gen in zip(eval_set, generations):
        ex["output"] = gen
        ex["generator"] = generator_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Saved {len(eval_set)} outputs to {output_path}")

########################################################################################################################

def load_combined_outputs(test_model_path: Path, reference_model_path: Path) -> List[Dict]:
    with test_model_path.open("r", encoding="utf-8") as f:
        test_outputs = {ex["instruction"]: ex["output"] for ex in json.load(f)}

    with reference_model_path.open("r", encoding="utf-8") as f:
        reference_outputs = [json.loads(line) for line in f if line.strip()]

    examples = []
    for ref in reference_outputs:
        instruction = ref["instruction"]
        if instruction not in test_outputs:
            continue
        examples.append({
            "instruction": instruction,
            "test_output": test_outputs[instruction],
            "reference_output": ref["output"],
            "dataset": ref["dataset"],
        })
    return examples

def format_preference_prompt(template: str, instruction: str, output_a: str, output_b: str) -> str:
    return template.replace("{{instruction}}", instruction).replace("{{output_1}}", output_a).replace("{{output_2}}", output_b)

def run_pairwise_annotation(llm: LLM, examples: List[Dict], template: str, batch_size: int = 1) -> List[Dict]:
    annotated = []
    prompts = [format_preference_prompt(template, ex["instruction"], ex["test_output"], ex["reference_output"]) for ex in examples]

    print(f"Evaluating {len(prompts)} comparisons with batch_size={batch_size}...")
    start_time = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        results = llm.generate(batch, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16, stop=["\n"]))
        for j, result in enumerate(results):
            vote = result.outputs[0].text.strip().upper()
            ex = examples[i + j]
            ex["annotator_vote"] = vote if vote in {"A", "B"} else None
            ex["len_test_output"] = len(ex["test_output"]) if ex["test_output"] else 0
            ex["len_ref_output"] = len(ex["reference_output"]) if ex["reference_output"] else 0
            annotated.append(ex)
    elapsed = time.time() - start_time
    print(f"Annotation took {elapsed:.2f} seconds → {len(prompts)/elapsed:.2f} examples/second")
    return annotated

def evaluator():
    test_outputs_path = Path("results/alpacaeval_qwen_output.json")
    reference_outputs_path = Path("data/alpaca_eval/alpaca_eval.jsonl")
    output_path = Path("results/alpacaeval_pairwise_results.json")
    template_path = Path("scripts/alpaca_eval_vllm_llama3_70b_fn/alpaca_eval_fn.txt")
    annotator_model_path = "/workspace/cs336_alignment/models/Qwen/Qwen2.5-3B-Instruct"

    print("Loading and pairing outputs")
    paired_examples = load_combined_outputs(test_outputs_path, reference_outputs_path)

    print("Loading preference prompt template")
    template = template_path.read_text(encoding="utf-8")

    print("Initializing Qwen2.5-3B-Instruct as annotator")
    llm = LLM(model=annotator_model_path, dtype="float32", enforce_eager=True)

    annotated = run_pairwise_annotation(llm, paired_examples, template, batch_size=1)

    print("Computing winrates")
    wins = sum(1 for ex in annotated if ex["annotator_vote"] == "A")
    total = sum(1 for ex in annotated if ex["annotator_vote"] in {"A", "B"})
    winrate = wins / total if total > 0 else 0.0
    print(f"Winrate: {wins}/{total} = {winrate:.2%}")


    length_filtered = [ex for ex in annotated if ex["len_test_output"] and ex["len_ref_output"] and abs(ex["len_test_output"] - ex["len_ref_output"]) / max(ex["len_test_output"], ex["len_ref_output"]) <= 0.15]
    length_filtered_total = sum(1 for ex in length_filtered if ex["annotator_vote"] in {"A", "B"})
    length_filtered_wins = sum(1 for ex in length_filtered if ex["annotator_vote"] == "A")
    length_winrate = length_filtered_wins / length_filtered_total if length_filtered_total > 0 else 0.0
    print(f"Length-controlled winrate: {length_filtered_wins}/{length_filtered_total} = {length_winrate:.2%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2)


if __name__ == "__main__":
    #main()
    evaluator()

