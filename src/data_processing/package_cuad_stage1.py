#!/usr/bin/env python3
"""
package_cuad_stage1.py  –  CUAD Stage-1 data packer  (logging version)

Converts your 18 source files

    spans_{train|validation|test}_filenames_{CategoryAgent}.jsonl

into *.jsonl instruction-tuning files with the frozen schema required for
Gemma-3 fine-tuning.

USAGE
-----
python package_cuad_stage1.py                       \
       --labels_dir    ./json_stage1/spans          \
       --contracts_dir ./contracts_txt              \
       --out_dir       ./stage1_jsonl

Author: 2025-04-24
"""

import argparse
import json
import pathlib
import re
import sys
import textwrap
from typing import Dict, List, Any
import yaml
import pathlib

_prompts_path = pathlib.Path(__file__).parent.parent.parent / "config.yml"
with _prompts_path.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

MAX_SNIPPET_CHARS = 200
AGENT_PROMPT_TMPL = _cfg["AGENT_PROMPT_TMPL"]

# --------------------------  MAP SOURCE → PROMPT NAME  -------------------- #
category_map: Dict[str, str] = {
    "Competition_ExclusivityAgent":        "Competition_Exclusivity",
    "Financial_CommercialTermsAgent":      "Financial_Commercial_Terms",
    "IntellectualProperty_LicensingAgent": "Intellectual_Property",
    "LegalProtections_LiabilityAgent":     "Legal_Protections_Liability",
    "Termination_ControlRightsAgent":      "Termination_Control_Rights",
    "Metadata":                            "Metadata",
}

# --------------------------  HELPER FUNCTIONS  ---------------------------- #


def pascal(label: str) -> str:
    """non-compete -> Non-Compete"""
    return label[:1].upper() + label[1:]


def make_snippet(txt: str) -> str:
    if len(txt) <= MAX_SNIPPET_CHARS:
        return txt
    return txt[:MAX_SNIPPET_CHARS].rstrip() + " …"


def load_source(path: pathlib.Path) -> Dict[str, Any]:
    """
    Flexibly load either:
        • single JSON object (dict of id -> record)
        • JSON array (list[record])
        • JSON-Lines (one record per line)
    Return dict[id] = record
    """
    with path.open("r", encoding="utf-8") as f:
        data = f.read().strip()

    # Try dict
    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {rec["id"]: rec for rec in obj}
    except json.JSONDecodeError:
        pass  # fall through to JSONL load

    # JSONL
    records = {}
    for ln, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"❌  {path.name}:{ln} – bad JSON: {e}", file=sys.stderr)
            continue
        records[rec["id"]] = rec
    return records

# --------------------------  CORE PACKAGING  ------------------------------ #


def package_file(
    split_raw: str,
    cat_key: str,
    in_path: pathlib.Path,
    contracts_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Transform one input file into JSON-Lines target."""
    records = load_source(in_path)
    agent_prompt = category_map[cat_key]

    split = "val" if split_raw == "validation" else split_raw
    written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for cid, rec in records.items():
            full_text = rec.get("text")
            if full_text is None:
                txt_path = contracts_dir / f"{cid}.txt"
                if not txt_path.exists():
                    print(
                        f"⚠️  Missing text for {cid}; skipping", file=sys.stderr)
                    continue
                full_text = txt_path.read_text(encoding="utf-8")

            # Extract the text of each clause
            clause_texts: List[str] = []
            for cl in rec.get("clauses", []):
                # Ensure text exists and is not empty before adding
                clause_text = cl.get("text")
                if clause_text:
                    # Use strip() to remove leading/trailing whitespace
                    clause_texts.append(clause_text.strip())

            # Join the clause texts with a newline separator
            target_string = "\\n".join(clause_texts)

            obj = {
                "input": AGENT_PROMPT_TMPL.format(
                    agent=agent_prompt, contract=full_text
                ),
                "target": target_string,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
            written += 1

    print(f"✓ {in_path.name:60s} → {out_path.name:45s}  ({written} contracts)")

# --------------------------  ENTRY POINT  --------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, type=pathlib.Path)
    ap.add_argument("--contracts_dir", required=True, type=pathlib.Path)
    ap.add_argument("--out_dir", required=True, type=pathlib.Path)
    args = ap.parse_args()

    pattern = re.compile(
        r"^spans_(train|validation|test)_filenames_(.+?)\.jsonl$", re.IGNORECASE
    )

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
        package_file(
            split_raw, cat_key, path,
            contracts_dir=args.contracts_dir,
            out_path=args.out_dir / out_file,
        )


if __name__ == "__main__":
    main()
