#!/usr/bin/env python
"""
eval_bio.py — Token-level evaluation against ground truth
=========================================================
Loads:
  1) Original CUAD_v1.json
  2) JSON mapping file (fine→broad)
  3) Predicted BIO JSONL file

Reconstructs **gold** BIO tags per record (contract × bucket) and compares
token-level labels to predicted, computing precision, recall, F1 for each bucket
and overall.

Usage:
    python eval_bio.py \
      --cuad_json   CUAD_v1.json \
      --label_json  fine_to_agent.json \
      --preds       cuad_bio_train.jsonl \
      --tokenizer   google/gemma-3-4b-it

Outputs a classification report to stdout.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import classification_report
from transformers import AutoTokenizer

# Agent buckets (must match processing)
BUCKETS = [
    "Metadata",
    "Intellectual Property & Licensing Agent",
    "Competition & Exclusivity Agent",
    "Termination & Control Rights Agent",
    "Financial & Commercial Terms Agent",
    "Legal Protections & Liability Agent",
]

# Helpers


def load_fine2broad(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_gold_bio(context: str,
                   spans: list[tuple[int, str]],
                   tokenizer: AutoTokenizer,
                   bucket: str) -> list[str]:
    # Tokenize and clean tokens exactly as in processing
    enc = tokenizer(context, return_offsets_mapping=True,
                    add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    raw_offsets = enc['offset_mapping']
    tokens, offsets = [], []
    for t, off in zip(raw_tokens, raw_offsets):
        t = t.replace("▁", "")
        if t:
            tokens.append(t)
            offsets.append(off)
    # Initialize gold labels
    gold_labels = ['O'] * len(tokens)
    # Mark spans
    for start, text in spans:
        end = start + len(text)
        started = False
        for i, (s, e) in enumerate(offsets):
            if e <= start or s >= end:
                continue
            if not started:
                gold_labels[i] = f"B-{bucket}"
                started = True
            elif gold_labels[i] == 'O':
                gold_labels[i] = f"I-{bucket}"
    return gold_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuad_json',   type=Path, required=True)
    parser.add_argument('--label_json',  type=Path, required=True)
    parser.add_argument('--preds',       type=Path, required=True)
    parser.add_argument('--tokenizer',   type=str,
                        default='google/gemma-3-4b-it')
    args = parser.parse_args()

    # Load data
    with args.cuad_json.open(encoding='utf-8') as f:
        data = json.load(f)['data']
    fine2broad = load_fine2broad(args.label_json)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Build gold spans per record id
    gold_spans = {}  # id -> spans per bucket
    id_to_context_para = {}  # id to (context, para)
    camel_pat = re.compile(r"(?<!^)(?=[A-Z])")

    for doc in data:
        cid = doc.get('title', '')
        for para in doc['paragraphs']:
            ctx = para['context']
            for qa in para['qas']:
                suffix = qa['id'].split('__')[-1]
                # try variants
                broad = None
                for var in (suffix, suffix.replace('_', ' '), camel_pat.sub(' ', suffix)):
                    broad = fine2broad.get(var)
                    if broad:
                        break
                if not broad:
                    continue
                rec_id = f"{cid}__{broad.replace(' ', '_')}"
                spans = [(ans['answer_start'], ans['text'])
                         for ans in qa['answers']]
                gold_spans.setdefault(rec_id, []).extend(spans)
                # store context for this rec
                id_to_context_para[rec_id] = ctx

    # Load predictions
    pred_labels = {}  # id -> predicted labels
    with args.preds.open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            pred_labels[rec['id']] = rec['labels']

    # Prepare evaluation lists
    y_true, y_pred = [], []

    for rec_id, spans in gold_spans.items():
        if rec_id not in pred_labels:
            print(f"Missing prediction for {rec_id}")
            continue
        context = id_to_context_para[rec_id]
        # build gold labels
        bucket = rec_id.split('__')[-1].replace('_', ' ')
        gold = build_gold_bio(context, spans, tokenizer, bucket)
        pred = pred_labels[rec_id]
        # ensure same length
        if len(gold) != len(pred):
            print(
                f"Length mismatch for {rec_id}: gold {len(gold)} vs pred {len(pred)}")
            continue
        # collect
        y_true.extend(gold)
        y_pred.extend(pred)

    # Classification report (token-level)
    labels = []
    for b in BUCKETS:
        labels.append(f"B-{b}")
        labels.append(f"I-{b}")
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0)
    print("=== Token-level classification report ===")
    print(report)


if __name__ == '__main__':
    print("Starting evaluation...")
    main()
