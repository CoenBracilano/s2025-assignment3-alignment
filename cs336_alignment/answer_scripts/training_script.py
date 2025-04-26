import math
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import torch.nn.functional as F
import wandb
from cs336_alignment.functions import InstructionDataset


def run_training():
    # Static hyperparameters for easy tuning
    model_name_or_path = "Qwen/Qwen2.5-0.5B"
    train_data_path = "data/finetune/train.jsonl.gz"
    eval_data_path = "data/finetune/test.jsonl.gz"
    output_dir = "qwen_finetuned"
    context_length = 512
    batch_size = 1  # sequences per device
    gradient_accumulation_steps = 32  # to get total 32
    learning_rate = 2e-5
    warmup_ratio = 0.03
    num_epochs = 1
    log_interval = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="qwen_finetuning")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    ).to(device)

    train_dataset = InstructionDataset(tokenizer, train_data_path, context_length, shuffle=True)
    eval_dataset = InstructionDataset(tokenizer, eval_data_path, context_length, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    total_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_interval == 0:
                    print(f"[Epoch {epoch}, Step {global_step}] Train Loss: {loss.item():.4f}")
                    wandb.log({"train/loss": loss.item(), "step": global_step})

        # Evaluation after each epoch
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)
                total_loss += loss.item()
            avg_eval_loss = total_loss / len(eval_dataloader)
            print(f"[Epoch {epoch}] Eval Loss: {avg_eval_loss:.4f}")
            wandb.log({"eval/loss": avg_eval_loss, "epoch": epoch})
        model.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()


if __name__ == "__main__":
    run_training()