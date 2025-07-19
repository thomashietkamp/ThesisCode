#!/usr/bin/env python3
"""
package_cuad_stage1.py — CUAD Stage‑1 data packer (token‑length aware, negatives kept)

Builds instruction‑tuning JSON‑Lines files that never exceed 16 000 tokens for
Gemma‑3‑4B‑IT fine‑tuning **and now retains slices whose `target` list is empty**, so
the model sees negative examples. Changes vs. the previous revision:

* No longer skips windows (or entire contracts) that contain zero labelled
  clauses — they are written with `"target": []`.
* Everything else (15 600‑token window, 512‑token stride, robust clause
  flattening) remains the same.

USAGE
-----
python package_cuad_stage1.py                       \
       --labels_dir    ./json_stage1/spans          \
       --contracts_dir ./contracts_txt              \
       --out_dir       ./stage1_jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import textwrap
from typing import Dict, List, Any, Tuple

import yaml
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------

_prompts_path = pathlib.Path(__file__).parent.parent.parent / "config.yml"
with _prompts_path.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

AGENT_PROMPT_TMPL: str = _cfg["AGENT_PROMPT_TMPL"]

# ---- 16k‑context slicing parameters ---------------------------------------

CTX_WINDOW = 16_000          # Gemma‑3‑4B‑IT context window
PROMPT_HEADROOM = 400        # reserve for system/user prompt + reply
SLICE_TOKENS = CTX_WINDOW - PROMPT_HEADROOM   # 15 600
STRIDE = 512                 # overlap to avoid boundary losses

_tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen3-4b")

# --------------------------  MAP SOURCE → PROMPT NAME  -------------------- #
category_map: Dict[str, str] = {
    "Competition_ExclusivityAgent":        "Competition_Exclusivity",
    "Financial_CommercialTermsAgent":      "Financial_Commercial_Terms",
    "IntellectualProperty_LicensingAgent": "Intellectual_Property",
    "LegalProtections_LiabilityAgent":     "Legal_Protections_Liability",
    "Termination_ControlRightsAgent":      "Termination_Control_Rights",
    "Metadata":                            "Metadata",
}

# Mapping from agent_prompt value (from category_map) to MAPPING key in config.yml
_agent_to_mapping_key: Dict[str, str] = {
    "Competition_Exclusivity": "Competition & Exclusivity",
    "Financial_Commercial_Terms": "Financial & Commercial Terms",
    "Intellectual_Property": "Intellectual Property & Licensing",
    "Legal_Protections_Liability": "Legal Protections & Liability",
    "Termination_Control_Rights": "Termination & Control Rights",
    # "Metadata" is not in the MAPPING, will be handled in formatting.
}
_mapping_data = _cfg.get('MAPPING', {})

# --------------------------  HELPER FUNCTIONS  ---------------------------- #


def load_source(path: pathlib.Path) -> Dict[str, Any]:
    """Load JSON / JSONL flexibly and return a dict keyed by record id."""
    with path.open("r", encoding="utf-8") as f:
        data = f.read().strip()

    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {rec["id"]: rec for rec in obj if "id" in rec}
    except json.JSONDecodeError:
        pass

    records: Dict[str, Any] = {}
    for ln, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"❌  {path.name}:{ln} – bad JSON: {e}", file=sys.stderr)
            continue
        if "id" not in rec:
            print(
                f"⚠️  {path.name}:{ln} – record missing 'id'; skipping", file=sys.stderr)
            continue
        records[rec["id"]] = rec
    return records


def flatten_text(obj: Any) -> str:
    """Recursively join arbitrary nested lists/strings into a single string."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten_text(x) for x in obj)
    return str(obj)


def slices_from_tokens(total: int, length: int = SLICE_TOKENS, stride: int = STRIDE) -> List[Tuple[int, int]]:
    """Yield (start, end) token‑offset pairs for a sliding window."""
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + length, total)
        spans.append((start, end))
        if end == total:
            break
        start += length - stride
    return spans

# --------------------------  CORE PACKAGING  ------------------------------ #


def package_file(
    split_raw: str,
    cat_key: str,
    in_path: pathlib.Path,
    contracts_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Transform one annotation file into fine‑tuning JSONL."""
    records = load_source(in_path)
    agent_prompt = category_map[cat_key]

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for cid, rec in records.items():
            # -------- Load contract body --------
            full_text = rec.get("text")
            if full_text is None:
                txt_path = contracts_dir / f"{cid}.txt"
                if not txt_path.exists():
                    print(
                        f"⚠️  Missing text for {cid}; skipping", file=sys.stderr)
                    continue
                full_text = txt_path.read_text(encoding="utf-8")

            contract_ids = _tokenizer(
                full_text, add_special_tokens=False).input_ids
            if not contract_ids:
                continue  # empty file

            # -------- Gather clause texts and labels --------
            raw_clauses = rec.get("clauses", [])
            # Store tuples of (label, text, token_set)
            clause_info: List[Tuple[str, str, set[int]]] = []
            for cl in raw_clauses:
                if not isinstance(cl, dict):
                    print(
                        f"⚠️  {cid}: Skipping non-dict clause: {cl}", file=sys.stderr)
                    continue

                label = cl.get("fine_label")
                text_field = cl.get("text")

                if not label or not text_field:
                    print(
                        f"⚠️  {cid}: Clause missing 'label' or 'text'; skipping: {cl}", file=sys.stderr)
                    continue

                text = flatten_text(text_field).strip()
                if not text:
                    continue  # Skip empty clauses

                tokset = set(_tokenizer(
                    text, add_special_tokens=False).input_ids)
                clause_info.append((label, text, tokset))

            # -------- Slice contract into ≤15 600‑token windows --------
            for s_idx, e_idx in slices_from_tokens(len(contract_ids)):
                window_ids = contract_ids[s_idx:e_idx]
                win_set = set(window_ids)

                # Build target dictionary {label: text} for clauses in this window
                targets: Dict[str, str] = {}
                if clause_info:
                    for label, txt, tset in clause_info:
                        if tset & win_set:
                            # If a label appears multiple times within the same window slice,
                            # the last occurrence's text will be stored.
                            targets[label] = txt
                # Note: targets can still be {} if no clauses intersect the window

                contract_slice = _tokenizer.decode(
                    window_ids, skip_special_tokens=True)

                # -------- Format prompt with specific clauses --------
                mapping_key = _agent_to_mapping_key.get(agent_prompt)
                clauses_list = _mapping_data.get(
                    mapping_key, []) if mapping_key else []
                if clauses_list:
                    clauses_str = "\n".join(
                        f"- {cl.lower()}" for cl in clauses_list)
                else:
                    # Handle cases like 'Metadata' or if mapping is missing
                    clauses_str = "(No specific clauses listed for this agent)"

                prompt_text = AGENT_PROMPT_TMPL.format(
                    agent=agent_prompt, clauses=clauses_str, contract=contract_slice)

                fout.write(json.dumps(
                    {"input": prompt_text, "target": targets}, ensure_ascii=False) + "\n")
                written += 1

    print(f"✓ {in_path.name:60s} → {out_path.name:45s}  ({written} slices)")

# --------------------------  ENTRY POINT  --------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Pack CUAD Stage‑1 span labels into Gemma‑3‑4B‑IT fine‑tuning JSONL files.
            Contracts are chunked so no record exceeds 16 k tokens, and slices
            with *no* labelled clauses are now *retained* (target = []).
            """,
        ),
    )
    ap.add_argument("--labels_dir", required=True, type=pathlib.Path,
                    help="Directory with spans_*_filenames_*.jsonl files")
    ap.add_argument("--contracts_dir", required=True, type=pathlib.Path,
                    help="Directory containing original contract .txt files")
    ap.add_argument("--out_dir", required=True, type=pathlib.Path,
                    help="Output directory for packed *.jsonl")
    args = ap.parse_args()

    pattern = re.compile(
        r"^spans_(train|validation|test)_filenames_(.+?)\.jsonl$", re.IGNORECASE)
    files = sorted(args.labels_dir.glob("*.jsonl"))
    if not files:
        sys.exit("No .jsonl files found in --labels_dir")

    for path in files:
        m = pattern.match(path.name)
        if not m:
            print(f"· skipping {path.name}")
            continue

        split_raw, cat_key = m.groups()
        if cat_key not in category_map:
            print(
                f"❌  Unknown category '{cat_key}' in {path.name}", file=sys.stderr)
            continue

        out_file = f"{'val' if split_raw=='validation' else split_raw}_{category_map[cat_key].lower()}.jsonl"
        package_file(split_raw, cat_key, path,
                     contracts_dir=args.contracts_dir, out_path=args.out_dir / out_file)


if __name__ == "__main__":
    main()
