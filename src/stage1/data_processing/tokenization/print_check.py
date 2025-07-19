#!/usr/bin/env python
"""
inspect_bio.py — Visualize BIO tags for a given contract and category
=====================================================================
Loads a JSONL file of BIO-tagged contracts (as produced by cuad_bio_processing.py),
filters for one record (contract × broad category), and prints all tokens,
highlighting B- and I- tokens in color for easy inspection.

Usage:
    python inspect_bio.py \
      --input    cuad_bio_train.jsonl \
      --contract "Contract_Title" \
      --category "Termination & Control Rights Agent" [--no-color]

Options:
    --input     Path to the JSONL BIO file
    --contract  Exact contract_id as in the JSONL (e.g., "Contract_0001")
    --category  Broad_label string (must match --broad_label in JSONL)
    --no-color  Disable ANSI color output (highlights with brackets instead)
"""
import argparse
import json
import sys
from pathlib import Path

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def highlight(token: str, label: str, use_color: bool) -> str:
    """Return highlighted token if B- or I-, else as-is."""
    if label.startswith('B-'):
        if use_color:
            return f"{RED}{token}{RESET}"
        else:
            return f"[B]{token}[/B]"
    elif label.startswith('I-'):
        if use_color:
            return f"{GREEN}{token}{RESET}"
        else:
            return f"[I]{token}[/I]"
    else:
        return token


def main():
    parser = argparse.ArgumentParser(
        description='Inspect BIO tags for a specific contract and category')
    parser.add_argument('--input',      '-i', type=Path, default='data/bio_labels_v2.json',
                        required=False, help='Path to cuad_bio_train.jsonl')
    parser.add_argument('--contract',   '-c', type=str, default='LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT',
                        required=False, help='contract_id to inspect')
    parser.add_argument('--category',   '-g', type=str, default='Metadata',
                        required=False, help='broad_label to inspect')
    parser.add_argument('--no-color',         action='store_true',
                        help='Disable ANSI color output')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    record = None
    with args.input.open(encoding='utf-8') as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get('contract_id') == args.contract and obj.get('broad_label') == args.category:
                record = obj
                break

    if record is None:
        print(
            f"No record found for contract='{args.contract}' and category='{args.category}'")
        sys.exit(1)

    tokens = record.get('tokens', [])
    labels = record.get('labels', [])

    # Print tokens on one line, highlights for B- and I- tags
    use_color = not args.no_color
    highlighted = [highlight(tok, lbl, use_color)
                   for tok, lbl in zip(tokens, labels)]
    print(' '.join(highlighted))


if __name__ == '__main__':
    main()
