import json
import random
import unicodedata
from pathlib import Path


def strip_accents(text: str) -> str:
    """
    Remove all accent marks from a string, leaving only the base characters.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG: adjust these paths if needed
# ─────────────────────────────────────────────────────────────────────────────
CAT_DIR = Path(__file__).parent.parent.parent.parent / \
    "data/cuad_by_category"
SPLIT_DIR = Path(__file__).parent.parent.parent.parent / "data/split_filenames"
SPLIT_FILES = {
    "train":      SPLIT_DIR / "train_filenames.txt",
    "test":       SPLIT_DIR / "test_filenames.txt",
    "validation": SPLIT_DIR / "validation_filenames.txt",
}

RNG_SEED = 42
random.seed(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load & normalize split sets
# ─────────────────────────────────────────────────────────────────────────────
splits = {}
for split, path in SPLIT_FILES.items():
    stems = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not (s := line.strip()):
                continue
            stem = Path(s).stem                      # remove .pdf/.PDF
            stem = strip_accents(stem).lower().strip()
            stems.add(stem)
    splits[split] = stems

# ─────────────────────────────────────────────────────────────────────────────
# 2) Known edge‐case overrides
# ─────────────────────────────────────────────────────────────────────────────
EDGE_CASE_MAP = {
    # # missing trailing space
    # "electrameccanica vehicles corp. - manufacturing agreement":
    #     "electrameccanica vehicles corp. - manufacturing agreement ",
    # actual PDF ends with "_Option Agreement"
    "harpoontherapeuticsinc_20200312_10-k_ex-10.18_12051356_ex-10.18_development agreement":
        "harpoontherapeuticsinc_20200312_10-k_ex-10.18_12051356_ex-10.18_development agreement_option agreement",
}

unmatched = set()
NO_ANS_TOKEN = "<no_answer>"
# ─────────────────────────────────────────────────────────────────────────────
# 3) Split each category JSONL
# ─────────────────────────────────────────────────────────────────────────────
for in_path in CAT_DIR.glob("*.jsonl"):
    cat_name = in_path.stem
    # Output files for this category (train, test, validation)
    category_output_files = {
        split_name: open(
            CAT_DIR / f"{cat_name}_{split_name}.jsonl", "w", encoding="utf-8")
        for split_name in splits
    }

    # Buffers for the current category's training data to apply downsampling
    category_train_records_empty = []
    category_train_records_non_empty = []

    with open(in_path, encoding="utf-8") as fin_category_jsonl:
        for line in fin_category_jsonl:  # Each line is a JSON record
            record = json.loads(line)
            qa_id = record.get("id", "")

            # Normalize document title for matching against split filename lists
            doc_title_normalized = strip_accents(
                qa_id.split("__", 1)[0]).lower().strip()

            # Apply any manual override for matching
            key_to_match = EDGE_CASE_MAP.get(
                doc_title_normalized, doc_title_normalized)

            assigned_split = None  # 'train', 'test', or 'validation'

            # Determine which split this record belongs to (exact match priority)
            for split_name, doc_stems_in_split in splits.items():
                if key_to_match in doc_stems_in_split:
                    assigned_split = split_name
                    break

            # Fallback: startswith match if no exact match found
            if assigned_split is None:
                for split_name, doc_stems_in_split in splits.items():
                    if any(doc_stem.startswith(key_to_match) for doc_stem in doc_stems_in_split):
                        assigned_split = split_name
                        break

            if assigned_split:
                # An empty target is where rec["target"] is an empty list []
                is_target_empty = (record.get("target") == json.dumps(
                    [NO_ANS_TOKEN], ensure_ascii=False))

                if assigned_split == "train":
                    if is_target_empty:
                        category_train_records_empty.append(line)
                    else:
                        category_train_records_non_empty.append(line)
                else:
                    # For 'test' and 'validation' splits, write the record line directly
                    category_output_files[assigned_split].write(line)
            else:
                # This document title didn't match any split definition
                unmatched.add(doc_title_normalized)

    # Write all training data for the current category
    # Write non-empty training records
    for record_line in category_train_records_non_empty:
        category_output_files["train"].write(record_line)

    # Write all empty training records
    for record_line in category_train_records_empty:
        category_output_files["train"].write(record_line)

    # Close the output files for this category
    for f in category_output_files.values():
        f.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4) Report
# ─────────────────────────────────────────────────────────────────────────────
print("Done splitting JSONLs into train/test/validation per category.\n")
if unmatched:
    print("⚠️  Unmatched document titles:")
    for title in sorted(unmatched):
        print("  -", title)
else:
    print("✅  All documents assigned successfully.")
