import json
import re
from pathlib import Path
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
#  1) Adjust these paths & model name to taste
# ─────────────────────────────────────────────────────────────────────────────
CUAD_PATH = Path(__file__).parent.parent.parent.parent / \
    "data/CUAD_v1/CUAD_v1.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / \
    "data/cuad_by_category"
MODEL_NAME = "Qwen/Qwen3-4b"   # you can swap for any HuggingFace‐style tokenizer
USE_SLIDING_WINDOW = True

NO_ANS_TOKEN = "<no_answer>"
CLS_TOKEN = "<cls_ans?>"

# ─────────────────────────────────────────────────────────────────────────────
#  2) Your mapping as given
# ─────────────────────────────────────────────────────────────────────────────
MAPPING = {
    "Intellectual Property & Licensing": [
        "IP Ownership Assignment", "Joint IP Ownership", "License Grant",
        "Non-Transferable License", "Affiliate License-Licensor",
        "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License", "Source Code Escrow",
    ],
    "Competition & Exclusivity": [
        "Most Favored Nation", "Competitive Restriction Exception",
        "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
        "No-Solicit Of Employees", "Non-Disparagement", "Rofr/Rofo/Rofn",
    ],
    "Termination & Control Rights": [
        "Termination For Convenience", "Change Of Control",
        "Anti-Assignment", "Post-Termination Services",
    ],
    "Financial & Commercial Terms": [
        "Revenue/Profit Sharing", "Price Restrictions",
        "Minimum Commitment", "Volume Restriction", "Audit Rights",
    ],
    "Legal Protections & Liability": [
        "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
        "Warranty Duration", "Insurance", "Covenant Not To Sue",
        "Third Party Beneficiary",
    ],
    "Metadata": [
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Governing Law"
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Prepare output files & tokenizer
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# open one file handle per category
writers = {
    cat: open(
        OUTPUT_DIR / f"{cat.replace(' ','_')}.jsonl", "w", encoding="utf-8")
    for cat in MAPPING
}
# a fallback file if question doesn't match any key
writers["__other__"] = open(OUTPUT_DIR / "other.jsonl", "w", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 4) Sliding‐window params
# ─────────────────────────────────────────────────────────────────────────────
MAX_LEN = 4096
HEADROOM = 256
SLICE_LEN = MAX_LEN - HEADROOM
STRIDE = 128

# ─────────────────────────────────────────────────────────────────────────────
# 5) Helper to pick which writer based on question text
# ─────────────────────────────────────────────────────────────────────────────


def pick_writer(qa_id: str):
    # Extract the label after the "__" in the QA id
    label = qa_id.split("__")[-1].strip().lower()
    # Loop over each category and its list of sublabels
    for category, sublabels in MAPPING.items():
        for sub in sublabels:
            if sub.strip().lower() == label:
                return writers[category]
    # Fallback writer
    return writers["__other__"]


# ─────────────────────────────────────────────────────────────────────────────
# 6) Main loop
# ─────────────────────────────────────────────────────────────────────────────
with open(CUAD_PATH, "r", encoding="utf-8") as f:
    cuad = json.load(f)

writer2category = {fh: cat for cat, fh in writers.items()}
counts = {cat: 0 for cat in writers}
# Initialize a dictionary to store sets of unique contract IDs per category
unique_contract_counts = {cat: set() for cat in writers}

for doc_idx, doc in enumerate(cuad["data"]):
    contract_title = doc.get("title", "Unknown Title")
    for para_idx, para in enumerate(doc["paragraphs"]):
        context = para["context"].replace("\n", " ").strip()

        # tokenize with sliding window or as a single block
        if USE_SLIDING_WINDOW:
            enc = tokenizer(
                context,
                truncation=True,
                max_length=MAX_LEN,
                stride=STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            all_window_ids = enc["input_ids"]
            all_offset_mappings = enc["offset_mapping"]
        else:
            enc = tokenizer(
                context,
                truncation=True,
                max_length=MAX_LEN,
                # get offsets for the single (truncated) window
                return_offsets_mapping=True
            )
            # Wrap in lists to maintain loop structure for the subsequent processing
            all_window_ids = [enc["input_ids"]]
            all_offset_mappings = [enc["offset_mapping"]]

        for qa_idx, qa in enumerate(para["qas"]):
            question = qa["question"].strip()
            qa_id = qa['id']
            current_label = qa_id.split("__")[-1].strip().lower()

            is_relevant_qa = current_label in [
                "uncapped liability", "document name"]

            if is_relevant_qa:
                print(
                    f"[LOG] Contract: '{contract_title}', QA_ID: '{qa_id}', Question: '{question}'")

            answers = [
                (
                    # if text is a list, join the pieces
                    " ".join(a["text"])
                    if isinstance(a.get("text"), list)
                    # otherwise take it as-is (or empty string)
                    else a.get("text", "")
                ).strip()
                for a in qa.get("answers", [])
            ]
            # build char‐span tuples
            spans = [(a["answer_start"], a["answer_start"] + len(a["text"]))
                     for a in qa.get("answers", [])]

            writer = pick_writer(qa['id'])
            chosen_category = writer2category[writer]

            if is_relevant_qa:
                print(
                    f"[LOG]   Assigned to category: '{chosen_category}'. Has answers: {bool(answers)}. Spans: {spans}")

            # for each window:
            for window_idx, (window_ids, offsets) in enumerate(zip(all_window_ids, all_offset_mappings)):
                # if a window is empty (e.g. empty context or only special tokens), skip
                if not window_ids or not offsets:
                    if is_relevant_qa:
                        print(
                            f"[LOG]   Window {window_idx}: Skipped (empty window_ids or offsets).")
                    continue

                # get character start/end of this window
                window_char_start, _ = offsets[0]
                _, window_char_end = offsets[-1]

                # Determine the actual answers to use for this window's output
                output_answers_for_this_window = []  # Default to empty
                window_contains_an_answer = False

                if spans:  # If there are answers defined for this QA
                    # Check if any of those answers are *within this specific window*
                    window_contains_an_answer = any(
                        window_char_start <= ans_start and ans_end <= window_char_end
                        for ans_start, ans_end in spans
                    )
                    if window_contains_an_answer:
                        # If yes, use the original answers for this QA
                        output_answers_for_this_window = answers
                    # Else (spans exist, but not in this window), output_answers_for_this_window remains []
                # Else (spans is empty, meaning no answers for this QA at all), output_answers_for_this_window remains []

                if is_relevant_qa:
                    log_prefix = f"[LOG]   Window {window_idx} ({window_char_start}-{window_char_end}):"
                    if not spans:
                        print(
                            f"{log_prefix} Writing (QA has no answers overall). Target will be empty.")
                    elif window_contains_an_answer:
                        print(
                            f"{log_prefix} Writing (answer in window). Target: {answers}. Spans: {spans}")
                    else:  # spans exist, but not in this window
                        print(
                            f"{log_prefix} Writing (answer NOT in window, but QA has answers). Target will be empty. Original spans for QA: {spans}")
                        # This implies a single (potentially truncated) context
                        if not USE_SLIDING_WINDOW:
                            for ans_idx, (ans_start, ans_end) in enumerate(spans):
                                if not (window_char_start <= ans_start and ans_end <= window_char_end):
                                    print(
                                        f"[LOG]     Info for empty target: Original answer {ans_idx} ({ans_start}-{ans_end}) for this QA is not fully in this window ({window_char_start}-{window_char_end}).")

                # reconstruct text
                window_text = tokenizer.decode(
                    window_ids, skip_special_tokens=True)

                prompt = (
                    f"[CONTRACT]\n{window_text}\n\n"
                    f"[QUESTION]\n{question}\n\n"
                    f"[RULE] If the clause is missing, an empty list; otherwise output a JSON list of clause text.\n\n"
                    f"[ANSWER]\n"
                )

                example = {
                    "id": qa_id.rsplit("__", 1)[0],
                    "input":  prompt,
                    "target": json.dumps(output_answers_for_this_window, ensure_ascii=False) if output_answers_for_this_window else json.dumps([], ensure_ascii=False),
                    "has_ans": bool(output_answers_for_this_window)
                }
                writer.write(json.dumps(example, ensure_ascii=False) + "\n")
                counts[writer2category[writer]] += 1
                # Add the document index to the set of unique contracts for this category
                unique_contract_counts[chosen_category].add(doc_idx)

# ─────────────────────────────────────────────────────────────────────────────
# 7) clean up
# ─────────────────────────────────────────────────────────────────────────────
for w in writers.values():
    w.close()

print("===== Example counts =====")
for cat, cnt in counts.items():
    print(f"{cat}: {cnt}")

print("\n===== Unique contract counts per category =====")
for cat, contract_set in unique_contract_counts.items():
    print(f"{cat}: {len(contract_set)}")

print("Done — JSONL files are in", OUTPUT_DIR)
