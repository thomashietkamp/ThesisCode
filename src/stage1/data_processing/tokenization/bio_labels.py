#!/usr/bin/env python
"""CUAD v1 → BIO converter — *project‑specific 6‑agent buckets*
================================================================
This edition matches the **exact six agents** you supplied:

1. Metadata
2. Intellectual Property & Licensing Agent
3. Competition & Exclusivity Agent
4. Termination & Control Rights Agent
5. Financial & Commercial Terms Agent
6. Legal Protections & Liability Agent

Provide a JSON file whose keys are fine‑grained labels and whose values
are *one of the six* bucket names (example shown in your last message).
All SentencePiece `▁` markers are removed; alignment is preserved.

Typical call:
```bash
python cuad_bio_processing.py \
  --cuad_json   ./CUAD_v1.json \
  --label_json  ./fine_to_agent.json \
  --out         cuad_bio_train.jsonl
```
Tokenizer defaults to *google/gemma-3-4b-it*. 
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKETS = [
    "Metadata",
    "Intellectual Property & Licensing Agent",
    "Competition & Exclusivity Agent",
    "Termination & Control Rights Agent",
    "Financial & Commercial Terms Agent",
    "Legal Protections & Liability Agent",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_mapping(json_path: Path) -> dict[str, str]:
    """Load fine→agent dictionary from JSON."""
    with json_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def mark_span(labels: List[str], offsets: List[Tuple[int, int]], start: int, text: str, agent: str):
    """In‑place BIO tagging for one character span."""
    end = start + len(text)
    started = False
    for i, (s, e) in enumerate(offsets):
        if e <= start or s >= end:
            continue
        if not started:
            labels[i] = f"B-{agent}"
            started = True
        elif labels[i] == "O":
            labels[i] = f"I-{agent}"

# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert(cuad_path: Path, map_path: Path, tokenizer_name: str, out_path: Path):
    fine2agent = load_mapping(map_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    camel_pat = re.compile(r"(?<!^)(?=[A-Z])")

    with cuad_path.open(encoding="utf-8") as fh:
        dataset = json.load(fh)["data"]

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for doc in tqdm(dataset, desc="contracts"):
            cid = doc.get("title", "")
            for para in doc["paragraphs"]:
                ctx = para["context"]
                enc = tok(ctx, return_offsets_mapping=True,
                          add_special_tokens=False)
                raw_tokens = tok.convert_ids_to_tokens(enc["input_ids"])
                raw_offsets = enc["offset_mapping"]

                # Clean SentencePiece markers
                tokens, offsets = [], []
                for t, off in zip(raw_tokens, raw_offsets):
                    t = t.replace("▁", "")
                    if t:
                        tokens.append(t)
                        offsets.append(off)

                # Map spans to agents
                ag_spans: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
                for qa in para["qas"]:
                    suffix = qa["id"].split("__")[-1]
                    # try raw, underscores to spaces, CamelCase to spaces
                    for variant in (
                        suffix,
                        suffix.replace("_", " "),
                        camel_pat.sub(" ", suffix),
                    ):
                        agent = fine2agent.get(variant)
                        if agent:
                            break
                    else:
                        continue  # fine label not in mapping
                    for ans in qa["answers"]:
                        ag_spans[agent].append(
                            (ans["answer_start"], ans["text"]))

                # Emit one record per agent bucket
                for agent in BUCKETS:
                    labels = ["O"] * len(tokens)
                    for st, txt in ag_spans.get(agent, []):
                        mark_span(labels, offsets, st, txt, agent)
                    record = {
                        "id": f"{cid}__{agent.replace(' ', '_')}",
                        "contract_id": cid,
                        "broad_label": agent,
                        "tokens": tokens,
                        "labels": labels,
                    }
                    out.write(json.dumps(record) + "\n")
                    written += 1
    print(f"✅ wrote {written:,} records to {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli(argv: List[str] | None = None):
    ap = argparse.ArgumentParser("CUAD BIO converter (6-agent edition)")
    ap.add_argument("--cuad_json", type=Path, required=True)
    ap.add_argument("--label_json", type=Path, required=True)
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-it")
    ap.add_argument("--out", type=Path, default=Path("cuad_bio_train.jsonl"))
    args = ap.parse_args(argv)

    convert(args.cuad_json, args.label_json, args.tokenizer, args.out)


if __name__ == "__main__":
    cli()
