# tokenize_cuad.py
"""Tokenization pipeline for CUAD-like contracts with Gemma BPE tokenizer (add_bos_token=False).

Usage
-----
python tokenize_cuad.py \
    --json_path cuad_v1.json \
    --output_path cuad_v1_tokenized.json

Each record in the output list contains:
    * input_ids: tokenizer ids (no <bos> token)
    * attention_mask: attention mask
    * offset_mapping: (start, end) character offsets for every token
    * gold_spans: list[tuple[int,int]] → gold clause spans (character offsets)

The script keeps **all** character‑level gold start–end offsets exactly as
provided in the original data – nothing is converted to token indices so that
higher‑level pipelines can decide later whether/how to align tokens and labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer, BatchEncoding

# any Gemma checkpoint shares the same vocab
TOKENIZER_NAME = "google/gemma-3-4b-it"


def get_tokenizer() -> "AutoTokenizer":
    """Instantiate Gemma BPE tokenizer **without** a BOS token."""
    tok = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        add_bos_token=False,  # critical for research methodology
    )

    # Ensure no BOS gets prepended by accident later
    tok.add_bos_token = False
    return tok


def char_spans(answers: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return (start, end) char spans for a list of answers."""
    spans: List[Tuple[int, int]] = []
    for ans in answers:
        start = ans["answer_start"]
        end = start + len(ans["text"])
        spans.append((start, end))
    return spans


def tokenize_context(tokenizer, context: str) -> BatchEncoding:
    """Tokenize *one* context paragraph and return the full encoding."""
    return tokenizer(
        context,
        add_special_tokens=True,  # <eos> is still useful; BOS is disabled globally
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,  # we want full documents; handle chunking upstream if needed
    )


def process_paragraph(tokenizer, paragraph: Dict[str, Any]) -> Dict[str, Any]:
    ctx: str = paragraph["context"]
    enc: BatchEncoding = tokenize_context(tokenizer, ctx)

    # Collect gold clause char‑span annotations for this paragraph
    gold_spans: List[Tuple[int, int]] = []
    for qa in paragraph["qas"]:
        if qa.get("is_impossible", False):
            continue  # skip negative examples – no gold clause span
        gold_spans.extend(char_spans(qa.get("answers", [])))

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offset_mapping": enc["offset_mapping"],
        "gold_spans": gold_spans,
    }


def process_file(json_path: Path) -> List[Dict[str, Any]]:
    """Walk through the CUAD‑style JSON file and tokenize every paragraph."""
    tokenizer = get_tokenizer()

    with json_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed: List[Dict[str, Any]] = []
    for contract in dataset["data"]:
        for paragraph in contract["paragraphs"]:
            processed.append(process_paragraph(tokenizer, paragraph))
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize CUAD contracts with Gemma tokenizer (no BOS)")
    parser.add_argument("--json_path", type=Path,
                        required=True, help="Path to cuad_v1.json")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Where to write the tokenized JSON")
    args = parser.parse_args()

    result = process_file(args.json_path)
    args.output_path.write_text(json.dumps(
        result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Tokenized file with {len(result)} paragraphs saved to {args.output_path}")


if __name__ == "__main__":
    main()
