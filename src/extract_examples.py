import json
import random
import itertools
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer

# Path definitions (same as in check.py)
RAW_FILE = Path("data/CUAD_v1/CUAD_v1.json")
TOK_FILE = Path("data/bio_labels.json")
MAP_FILE = Path("data/category_mapping.json")
TOKENIZER_NAME = "google/gemma-3-4b-it"

# Load data (same as in check.py)
raw = json.loads(RAW_FILE.read_text("utf-8"))
tokenised = json.loads(TOK_FILE.read_text("utf-8"))
fine2super = json.loads(MAP_FILE.read_text("utf-8"))

# Convert names to ints if needed
if all(isinstance(v, int) for v in fine2super.values()):
    pass
else:
    name2id = {}
    for k, v in list(fine2super.items()):
        if v not in name2id:
            name2id[v] = len(name2id)
        fine2super[k] = name2id[v]

# Load tokenizer
tok = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, use_fast=True, add_bos_token=False)


def char2tok(offsets, s, e):
    """inclusive token span or None"""
    ts = te = None
    for i, (a, b) in enumerate(offsets):
        if a <= s < b and ts is None:
            ts = i
        if a < e <= b:
            te = i
            break
    return (ts, te) if ts is not None and te is not None else None

# Function to find unlabeled spans


def find_unlabeled_spans(raw_contract, enc_para):
    """Find examples where char2tok returns None"""
    unlabeled_spans = []
    offsets = enc_para.get("offset_mapping")

    for qa in raw_contract["qas"]:
        if qa.get("is_impossible"):
            continue

        category = qa["id"].split("__")[-1]
        sup = fine2super.get(category)
        if sup is None:
            continue

        for ans in qa["answers"]:
            span = char2tok(offsets, ans["answer_start"],
                            ans["answer_start"] + len(ans["text"]))

            if not span:
                # Found an unlabeled span
                unlabeled_spans.append({
                    "category": category,
                    "super_category": sup,
                    "answer_text": ans["text"],
                    "answer_start": ans["answer_start"],
                    "paragraph_text": get_context_around_span(raw_contract["context"], ans["answer_start"], len(ans["text"]), 100)
                })

    return unlabeled_spans

# Function to find mislabeled spans


def find_mislabeled_spans(raw_contract, enc_para):
    """Find examples where char2tok returns a span but with wrong category"""
    mislabeled_spans = []
    offsets = enc_para.get("offset_mapping")
    labels = enc_para["labels"]

    for qa in raw_contract["qas"]:
        if qa.get("is_impossible"):
            continue

        category = qa["id"].split("__")[-1]
        sup = fine2super.get(category)
        if sup is None:
            continue

        expected_B = 1 + 2*sup
        expected_I = expected_B + 1

        for ans in qa["answers"]:
            span = char2tok(offsets, ans["answer_start"],
                            ans["answer_start"] + len(ans["text"]))

            if span:
                ts, te = span

                # Check if the span is labeled with a different category
                if labels[ts] > 0 and labels[ts] != expected_B:
                    actual_category = (labels[ts] - 1) // 2

                    mislabeled_spans.append({
                        "expected_category": category,
                        "expected_super_category": sup,
                        "actual_super_category": actual_category,
                        "answer_text": ans["text"],
                        "answer_start": ans["answer_start"],
                        "token_span": span,
                        "paragraph_text": get_context_around_span(raw_contract["context"], ans["answer_start"], len(ans["text"]), 100)
                    })

    return mislabeled_spans


def get_context_around_span(text, start, length, context_size=100):
    """Get text around a span with ~context_size characters on each side"""
    context_start = max(0, start - context_size)
    context_end = min(len(text), start + length + context_size)

    prefix = "..." if context_start > 0 else ""
    suffix = "..." if context_end < len(text) else ""

    # Mark the span with ** for visibility
    result = (
        prefix +
        text[context_start:start] +
        "**" + text[start:start+length] + "**" +
        text[start+length:context_end] +
        suffix
    )

    return result


def main():
    print("Searching for examples of unlabeled and mislabeled spans...\n")

    all_unlabeled = []
    all_mislabeled = []

    # Process each contract
    for raw_contract, enc_para in zip(
            itertools.chain.from_iterable(
                c["paragraphs"] for c in raw["data"]),
            tokenised):

        unlabeled = find_unlabeled_spans(raw_contract, enc_para)
        mislabeled = find_mislabeled_spans(raw_contract, enc_para)

        all_unlabeled.extend(unlabeled)
        all_mislabeled.extend(mislabeled)

    # Print examples of unlabeled spans
    print(f"Found {len(all_unlabeled)} unlabeled spans")
    if all_unlabeled:
        print("\nEXAMPLES OF UNLABELED SPANS (char2tok returned None):")
        for i, example in enumerate(all_unlabeled[:2]):  # Show top 2 examples
            print(f"\nUnlabeled Example #{i+1}:")
            print(
                f"Category: {example['category']} (Super category ID: {example['super_category']})")
            print(f"Answer start: {example['answer_start']}")
            print(
                f"Answer text: {example['answer_text'][:100]}{'...' if len(example['answer_text']) > 100 else ''}")
            print(f"Paragraph context:")
            print(example['paragraph_text'])
            print(
                f"Issue: char2tok(offsets, {example['answer_start']}, {example['answer_start'] + len(example['answer_text'])}) returned None")

    # Print examples of mislabeled spans
    print(f"\nFound {len(all_mislabeled)} mislabeled spans")
    if all_mislabeled:
        print("\nEXAMPLES OF MISLABELED SPANS (wrong category):")
        for i, example in enumerate(all_mislabeled[:2]):  # Show top 2 examples
            print(f"\nMislabeled Example #{i+1}:")
            print(
                f"Expected category: {example['expected_category']} (Super category ID: {example['expected_super_category']})")
            print(
                f"Actual super category ID: {example['actual_super_category']}")
            print(f"Answer start: {example['answer_start']}")
            print(f"Token span: {example['token_span']}")
            print(
                f"Answer text: {example['answer_text'][:100]}{'...' if len(example['answer_text']) > 100 else ''}")
            print(f"Paragraph context:")
            print(example['paragraph_text'])
            print(
                f"Issue: Found span {example['token_span']} but with category ID {example['actual_super_category']} instead of {example['expected_super_category']}")


if __name__ == "__main__":
    main()
