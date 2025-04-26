import gzip
import json
from pathlib import Path
from typing import List, Dict

def load_anthropic_hh_dataset(data_dir: Path) -> List[Dict]:
    dataset = []
    files = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]

    for file_name in files:
        path = data_dir / file_name
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                chosen = ex.get("chosen")
                rejected = ex.get("rejected")

                if not chosen or not rejected:
                    continue
                if len(chosen) != 2 or len(rejected) != 2:
                    continue
                if chosen[0]["role"] != "human" or chosen[1]["role"] != "assistant":
                    continue
                if rejected[0]["role"] != "human" or rejected[1]["role"] != "assistant":
                    continue

                dataset.append({
                    "instruction": chosen[0]["content"],
                    "chosen": chosen[1]["content"],
                    "rejected": rejected[1]["content"],
                    "source": file_name.replace(".jsonl.gz", "")
                })

    return dataset