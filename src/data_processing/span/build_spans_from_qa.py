"""
CUAD QA ➜ span JSONL   (handles .pdf suffixes & extracts fine label from id)
---------------------------------------------------------------------------
python build_spans_from_qa.py \
       --qa_file data/CUAD_v1/cuad_v1.json \
       --splits_dir data/splits            # train.txt / val.txt / test.txt
       --cat_map  data/category_mapping.json \
       --out_dir  data/jsonl
"""
import argparse
import json
import pathlib
import re
import collections


def strip_pdf(name: str) -> str:
    """Remove .pdf or .PDF (case-insensitive) from a filename string."""
    return re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)


def load_ids(path):
    return {strip_pdf(line.strip()) for line in open(path) if line.strip()}


def char_to_tok(tokenizer, text, cs, ce):
    enc = tokenizer(text, return_offsets_mapping=True,
                    add_special_tokens=False)
    offs = enc.offset_mapping
    start_tok = next(i for i, (s, e) in enumerate(offs)
                     if s <= cs < e)
    # similarly for answer end
    end_tok = next(i for i, (s, e) in enumerate(offs)
                   if s <= ce < e)

    return start_tok, end_tok


def main(qa_file, splits_dir, cat_map, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")  # same as 4B

    cat_map_data = json.load(open(cat_map))
    cat_map = {k.lower(): v for k, v in cat_map_data.items()}  # fine ➜ broad
    # Get all unique broad categories
    all_broad_categories = set(cat_map.values())

    qa_data = json.load(open(qa_file))["data"]

    # --- read split lists
    splits = {n: load_ids(f"{splits_dir}/{n}.txt")
              for n in ["train_filenames", "validation_filenames", "test_filenames"]}

    # --- containers
    span_examples = {n: [] for n in splits}            # full corpora
    by_broad = {n: collections.defaultdict(list) for n in splits}  # per agent

    for doc in qa_data:
        doc_id = strip_pdf(doc["title"])
        split = next((s for s, ids in splits.items() if doc_id in ids), None)
        if split is None:
            continue                                    # skip docs outside the split

        # CUAD stores whole text here
        para = doc["paragraphs"][0]
        context = para.get("context") or para["qas"][0]["context"]

        spans = []
        broad_seen = collections.defaultdict(list)

        for qa_item in para["qas"]:
            if qa_item["is_impossible"]:
                continue
            # -------- fine & broad labels
            fine = qa_item["id"].split(
                "__")[-1].strip().lower()     # after the __
            broad = cat_map[fine]

            for ans in qa_item["answers"]:
                cs, ce = ans["answer_start"], ans["answer_start"] + \
                    len(ans["text"]) - 1
                ts, te = char_to_tok(tok, context, cs, ce)

                span_dict = {
                    "start": cs,
                    "end":   ce,
                    "text":  ans["text"],
                    "fine_label":  fine,
                    "broad_label": broad
                }
                if span_dict not in spans:
                    spans.append(span_dict)
                    broad_seen[broad].append(span_dict)

        example = {"id": doc_id, "text": context, "clauses": spans}
        span_examples[split].append(example)

        # NEW LOGIC: Ensure doc is added to each broad category list, even if empty
        for b_cat in all_broad_categories:
            by_broad[split][b_cat].append(
                {
                    "id": doc_id,
                    "text": context,
                    # Use clauses if seen, else empty list
                    "clauses": broad_seen.get(b_cat, [])
                }
            )

    # --- write files
    out_p = pathlib.Path(out_dir).resolve()
    out_p.mkdir(exist_ok=True, parents=True)

    for split, items in span_examples.items():
        with open(out_p / f"spans_{split}.jsonl", "w") as f:
            for ex in items:
                f.write(json.dumps(ex) + "\n")
        print(f"[{split}]  full corpus  → {len(items)} docs")

        # five specialised agents
        for broad, lst in by_broad[split].items():
            fname = f"spans_{split}_{broad.replace(' & ', '_').replace(' ', '')}.jsonl"
            with open(out_p / fname, "w") as f:
                for ex in lst:
                    f.write(json.dumps(ex) + "\n")
            print(f"          {broad:<35} → {len(lst)} docs")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--qa_file",   required=True)
    pa.add_argument("--splits_dir", required=True)
    pa.add_argument("--cat_map",   required=True)
    pa.add_argument("--out_dir",   required=True)
    main(**vars(pa.parse_args()))
