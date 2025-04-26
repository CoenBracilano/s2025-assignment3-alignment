# This file contains any functions or scripts that have a test in adapters
# In essence it is a companion to the adapters.py file

import gzip
import json
import random
import re
from typing import Any
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch


# Section 3.2 question a) 
def mmlu_parser(mmlu_example: dict[str, Any], model_output: str)-> str | None:
    if not isinstance(model_output, str):
        return None

    model_output = model_output.strip() # remove whitespaces

    # Use regex to find the correct letter given the output 
    match = re.search(r'the correct answer is\s+([A-D])\b', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

def gms8k_parser(model_output: str)-> str | None:
    if not isinstance(model_output, str):
        return None


    matches = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if matches:
        return matches[-1]
    return None


# Section 4.2.1 
class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle=False, pack_by_example=False):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.input_ids = []

        def _maybe_open(path):
            if str(path).endswith(".gz"):
                return gzip.open(path, "rt", encoding="utf-8")
            return open(path, "r", encoding="utf-8")

        with _maybe_open(dataset_path) as f:
            data = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            random.shuffle(data)
        # Build flat token stream
        documents = []
        for i, ex in enumerate(data):
            prompt = ex["prompt"].strip()
            response = ex["response"].strip()
            full_text = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            )
            encoded = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if i > 0:
                documents.append(tokenizer.eos_token_id)
            documents.append(tokenizer.bos_token_id)
            documents.extend(encoded)
        documents.append(tokenizer.eos_token_id)
        self.flat_tokens = documents
        # Chunk into sequences
        for i in range(0, len(documents) - seq_length + 1, seq_length):
            chunk = documents[i:i + seq_length]
            self.input_ids.append(torch.tensor(chunk, dtype=torch.long))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids = self.input_ids[i]

        # Default next token: eos_token if we're truly at the end of the dataset
        if i + 1 < len(self.input_ids):
            next_token = self.input_ids[i + 1][0].unsqueeze(0)
        else:
            # If we are at the end, try to extract the next token directly from the raw token stream
            raw_flat_index = (i + 1) * self.seq_length
            if raw_flat_index < len(self.flat_tokens):
                next_token = torch.tensor([self.flat_tokens[raw_flat_index]], dtype=torch.long)
            else:
                next_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        labels = torch.cat([ids[1:], next_token])
        return {
            "input_ids": ids,
            "labels": labels
        }





#4.2.1 data_loading part b
def get_batch_loader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

