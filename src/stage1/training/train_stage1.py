#!/usr/bin/env python3
"""
Gemma-3-4B QLoRA fine-tuner for Stage-1 clause-extraction agents.

Key differences vs. previous draft
----------------------------------
1. Pre-processing no longer tries to `split("<CONTRACT_TEXT>")`; it uses
   *whatever string you stored in "input"* verbatim.
2. Dataset.map is **not batched** (fixes the TypeError you saw).
3. A --dry_run_steps flag lets you exit after N optimiser steps.
4. All hyper-params kept identical to the defaults we agreed on.

Test run
--------
python train_stage1.py                            \
    --train_jsonl  ./train_small.jsonl            \
    --val_jsonl    ./train_small.jsonl            \
    --agent_name   Competition_Exclusivity        \
    --output_dir   ./dry_ckpt                     \
    --dry_run_steps 10
"""

from functools import lru_cache
import orjson                          # 2‑3× faster than json
import argparse
import json
import os
import pathlib
import random
import sys
from typing import Dict, Any, List
from functools import partial

import numpy as np
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


login(token="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")

# -------------------- constants you rarely touch -------------------- #
MAX_LENGTH = 5000           # Gemma-3-4B-IT context window
MICRO_BATCH = 4               # fits on an A100-80 GB with 4-bit weights
GRAD_ACC = 16            # 4 × 8 = 32 effective batch
EPOCHS = 3
LR = 1e-4

random.seed(42)
torch.manual_seed(42)

# -------------------- helper: load a JSON-Lines file ---------------- #


def load_jsonl(path: str):
    return load_dataset("json", data_files=path, split="train", cache_dir=".cache")


class MaskedTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# -------------------- helper: per-example tokeniser ----------------- #


def make_preprocess(tokenizer):
    def _pp(example):
        # Tokenize the input normally
        p = tokenizer(example["input"], add_special_tokens=False)

        # example["target"] is always a list of strings (e.g., [] or ["ans1", "ans2"]).
        # We format it as a JSON string, so the model learns to output valid JSON.
        # This ensures the output is parsable by metrics functions (like _safe_json)
        # and aligns with the prompt's expectation of an "empty list" or items
        # "in double quotes, separated by commas".
        target_str = json.dumps(example["target"], ensure_ascii=False)

        # Tokenize the target
        t = tokenizer(target_str, add_special_tokens=False)

        # Append EOS token so model learns end of sequence
        input_ids = p.input_ids + t.input_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(p.input_ids) + t.input_ids + \
            [tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    return _pp

# -------------------- custom metric: order-independent clause F1 --- #


@lru_cache(maxsize=1024)
def _safe_json(text: str):
    """
    Parse the first JSON object or array in *text*. Returns a dict or list.
    If nothing valid is found, returns an empty dict (so _extract_clauses
    will return [] and you'll see F1=0 when the model truly predicts nothing).
    """
    # 1) Try to parse the whole string
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        pass

    # 2) Try to find a JSON object {...}
    obj_start, obj_end = text.find('{'), text.rfind('}') + 1
    if obj_start != -1 and obj_end > obj_start:
        try:
            return orjson.loads(text[obj_start:obj_end])
        except orjson.JSONDecodeError:
            pass

    # 3) Try to find a JSON array [...]
    arr_start, arr_end = text.find('['), text.rfind(']') + 1
    if arr_start != -1 and arr_end > arr_start:
        try:
            return orjson.loads(text[arr_start:arr_end])
        except orjson.JSONDecodeError:
            pass

    # 4) Give up
    return {}


def _extract_clauses(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return list(raw.keys())
    return []


def compute_metrics(eval_pred, tokenizer):
    raw_preds, raw_labels = eval_pred
    # handle both logits and generated‐IDs
    # if raw_preds.ndim == 3:
    #     preds = np.argmax(raw_preds, axis=-1)
    # else:
    preds = raw_preds
    labels = np.where(raw_labels == -100,
                      tokenizer.pad_token_id,
                      raw_labels)

    dec_preds = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # debug print
    if len(dec_preds) and True:
        print(">>> PRED:", repr(dec_preds[0]))
        print(">>> LABL:", repr(dec_labels[0]))

    f1_sum = 0.0
    for p_str, l_str in zip(dec_preds, dec_labels):
        raw_p = _safe_json(p_str)
        raw_l = _safe_json(l_str)
        p_list = _extract_clauses(raw_p)
        l_list = _extract_clauses(raw_l)
        p_set, l_set = set(p_list), set(l_list)

        if not l_set and not p_set:
            f1 = 1.0
        elif not l_set or not p_set:
            f1 = 0.0
        else:
            inter = len(p_set & l_set)
            prec = inter / len(p_set)
            rec = inter / len(l_set)
            f1 = 2*prec*rec/(prec+rec)
        f1_sum += f1

    return {"clause_f1": f1_sum/len(dec_preds)}


# -------------------- metric stub (replace later) ------------------- #


def dummy_metrics(_):
    return {"dummy_F1": 0.0}

# -------------------- main entry-point ------------------------------ #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--agent_name",  required=True)
    ap.add_argument("--output_dir",  required=True, type=pathlib.Path)
    # change if needed
    ap.add_argument("--model_name",  default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dry_run_steps", type=int, default=0,
                    help=">0 = stop after N optimiser steps (smoke-test)")

    # if get_ipython() is not None:
    #     # means we are running in a notebook
    #     print("Running in a notebook, using default arguments for train_jsonl, val_jsonl, agent_name, and output_dir")
    #     args = ap.parse_args([
    #         '--train_jsonl', 'train_small.jsonl',
    #         '--val_jsonl', 'val_small.jsonl',
    #         '--agent_name', 'Competition_Exclusivity',
    #                         '--output_dir', 'checkpoints/competition_exclusivity',
    #                         '--dry_run_steps', '10'
    #     ])
    # else:

    args = ap.parse_args()

    if args.dry_run_steps:
        print(
            f"🔎 DRY-RUN enabled: will stop after {args.dry_run_steps} updates.")

    tok = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=MAX_LENGTH, padding_side="left", use_fast=True, trust_remote_code=True
    )
    tok.pad_token = tok.eos_token

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=qconf,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager',
        offload_folder='hf_offload',
        offload_state_dict=True
    )
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    # model.enable_xformers_memory_efficient_attention()
    model = prepare_model_for_kbit_training(model)

    wandb.init(project=args.agent_name, config={
        "model_name": args.model_name,
        "max_length": MAX_LENGTH,
        "micro_batch": MICRO_BATCH,
        "grad_acc": GRAD_ACC,
        "epochs": EPOCHS,
        "lr": LR,
    })
    wandb.watch(model, log="all", log_freq=25)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model = model.to(torch.bfloat16)

    train_ds = load_jsonl(args.train_jsonl).map(make_preprocess(tok),
                                                remove_columns=["id", "input", "target"])
    # Select only the first sample
    # train_ds = train_ds.select([0])
    val_ds = load_jsonl(args.val_jsonl).map(make_preprocess(tok),
                                            remove_columns=["id", "input", "target"])
    # val_ds = val_ds.select([0])

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=MICRO_BATCH,
        per_device_eval_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        bf16=True,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=args.dry_run_steps if args.dry_run_steps > 0 else -1,
        run_name=args.agent_name,
        eval_accumulation_steps=1,
        torch_empty_cache_steps=1,
        log_level="debug",
        use_cpu=False,
        bf16_full_eval=True,
        predict_with_generate=False,
        generation_max_length=MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tok,
        padding=True,
        return_tensors="pt"
    )

    # Use partial to pass tokenizer to compute_metrics
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tok)

    trainer = MaskedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        # compute_metrics=compute_metrics_with_tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tok.save_pretrained(str(args.output_dir / "final"))
    print("✅ finished — model saved to", args.output_dir / "final")


if __name__ == "__main__":
    main()
