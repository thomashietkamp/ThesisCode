#!/usr/bin/env python3
"""
Fine-tune one specialised Stage-1 agent (Gemma-3-4B QLoRA).

Example
-------
python train_stage1.py                         \
    --train_jsonl  data/train_competition_exclusivity.jsonl \
    --val_jsonl    data/val_competition_exclusivity.jsonl   \
    --agent_name   Competition_Exclusivity                  \
    --output_dir   checkpoints/competition_exclusivity

The script automatically:
  • loads Gemma-3-4B in 4-bit
  • applies your prompt template
  • trains with QLoRA + gradient checkpointing
  • logs metrics to Weights-&-Biases (optional)
"""

import argparse
import json
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ----------------------- constants & defaults ---------------------------- #

CONTEXT_TEMPLATE = (
    "<AGENT={agent}>\n"
    "<CONTRACT_TEXT>\n"
    "{contract}\n"
)
TARGET_KEY = "target"

MAX_LENGTH = 131_072           # <<< adjust if your checkpoint differs
BATCH_SIZE = 4                 # micro-batch
GRAD_ACC = 8                 # => effective 32
EPOCHS = 3
LEARNING_RATE = 5e-5

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------- helper functions -------------------------------- #


def load_jsonl(path: str) -> Dataset:
    return load_dataset(
        "json",
        data_files=path,
        split="train",   # ‘train’ even for val/test files – we control the split
        cache_dir=".cache",
    )


def format_example(ex: Dict[str, Any], agent_name: str) -> str:
    prompt = CONTEXT_TEMPLATE.format(agent=agent_name,
                                     contract=ex["input"].split(
                                         "<CONTRACT_TEXT>\n", 1)[1])
    # We keep the full prompt from the jsonl to preserve your ‘enhanced’ parts,
    # but make absolutely sure it ends with the contract text – no duplicate agent tags
    return prompt + "\n" + json.dumps(ex[TARGET_KEY], ensure_ascii=False)


def preprocess(tokenizer, agent_name: str):
    def _fn(batch):
        texts = [format_example(ex, agent_name) for ex in batch]
        batch_enc = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch_enc["labels"] = batch_enc["input_ids"].clone()
        return batch_enc
    return _fn


def compute_metrics(eval_pred):
    # simple clause-level F1 ignoring boundary IoU for speed –
    # use your dedicated evaluation script for the paper
    preds, labels = eval_pred
    # decode first, then count “label”: occurrences
    # (placeholder – real metric should parse JSON and compute IoU)
    return {"dummy_F1": 0.0}

# ----------------------- main ------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--agent_name",  required=True,
                    help="e.g. Competition_Exclusivity")
    ap.add_argument("--output_dir",  required=True, type=pathlib.Path)
    ap.add_argument("--model_name",  default="google/gemma-3-4b")
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token  # Gemma has no pad-token

    # 4-bit config
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_conf,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA – adapt all attention projections + MLP gating
    lora_conf = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_conf)
    model.gradient_checkpointing_enable()

    # Datasets
    train_ds = load_jsonl(args.train_jsonl)
    val_ds = load_jsonl(args.val_jsonl)

    preprocess_fn = preprocess(tokenizer, args.agent_name)
    train_ds = train_ds.map(preprocess_fn, batched=True,
                            remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_fn,   batched=True,
                        remove_columns=val_ds.column_names)

    # Trainer
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        bf16=True,
        report_to=["wandb"] if args.wandb_project else [],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    print("✅ Finished:", args.agent_name, "→", args.output_dir / "final")


if __name__ == "__main__":
    main()
