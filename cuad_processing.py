# cuad_processing.py
import os
import json
import io
import re  # Import regular expression module
from collections import Counter
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd  # Added pandas import
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# --- Configuration ---
# !!! PLEASE UPDATE THESE PATHS IF NECESSARY !!!
# Updated PDF directory path
PDF_DIR = "data/CUAD_v1/full_contract_pdf"
# Updated path to the annotations CSV file
ANNOTATIONS_PATH = "data/CUAD_v1/CUAD_v1.json"
# !!! Verify this column name in your master_clauses.csv !!!
ANNOTATION_LABEL_COLUMN = 'Clause Label'
PLOT_OUTPUT_DIR = "output_plots"  # Directory to save the histogram
TOKEN_LIMIT = 120000
# Encoding for token counting (common for GPT-4, GPT-3.5)
TIKTOKEN_ENCODING = "cl100k_base"
# --- End Configuration ---


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    try:
        # Using LAParams to potentially improve layout analysis, adjust if needed
        return extract_text(pdf_path, laparams=LAParams(line_margin=0.5))
    except Exception as e:
        print(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
        return ""


def count_tokens(text: str, encoding_name: str = TIKTOKEN_ENCODING) -> int:
    """Counts tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def process_pdfs(pdf_dir: str, token_limit: int) -> tuple[dict[str, str], list[str]]:
    """
    Loads PDFs recursively, extracts text, counts tokens, and flags oversized documents.

    Returns:
        A tuple containing:
        - A dictionary mapping relative file paths (from pdf_dir) to extracted text.
        - A list of relative file paths exceeding the token limit.
    """
    if not os.path.isdir(pdf_dir):
        print(f"Error: PDF directory not found at {pdf_dir}")
        return {}, []

    extracted_texts = {}
    flagged_files = []
    pdf_file_paths = []

    # Use os.walk to find PDF files recursively
    for root, _, files in os.walk(pdf_dir):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(root, filename)
                # Store path relative to pdf_dir for consistency if needed later
                relative_path = os.path.relpath(full_path, pdf_dir)
                pdf_file_paths.append((full_path, relative_path))

    print(
        f"Found {len(pdf_file_paths)} PDF files in {pdf_dir} and its subdirectories.")

    for i, (pdf_path, relative_path) in enumerate(pdf_file_paths):
        print(f"Processing ({i+1}/{len(pdf_file_paths)}): {relative_path}...")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue  # Skip if text extraction failed

        # Use relative path as the key
        extracted_texts[relative_path] = text
        token_count = count_tokens(text)

        print(f"  - Token count: {token_count}")
        if token_count > token_limit:
            flagged_files.append(relative_path)
            print(f"  - FLAGGED: Exceeds {token_limit} tokens.")

    return extracted_texts, flagged_files


def analyze_clause_frequency(annotations_path: str) -> Counter:
    """
    Analyzes the frequency of each clause label from CUAD annotations JSON.
    Extracts the label name from within double quotes in the 'question' field.
    """
    if not os.path.isfile(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return Counter()

    label_counts = Counter()
    # Regex to find text between the first pair of double quotes
    label_pattern = re.compile(r'"([^"]*)"')

    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'data' in data:
            processed_qas = 0
            found_labels = 0
            for contract in data['data']:
                if 'paragraphs' in contract:
                    for paragraph in contract['paragraphs']:
                        if 'qas' in paragraph:
                            for qa in paragraph['qas']:
                                processed_qas += 1
                                if 'question' in qa:
                                    question_text = qa['question']
                                    match = label_pattern.search(question_text)
                                    if match:
                                        label = match.group(1).strip()
                                        # Only count if the clause has actual answers (not 'is_impossible')
                                        if label and qa.get('answers') and len(qa['answers']) > 0 and not qa.get('is_impossible', False):
                                            label_counts[label] += 1
                                            found_labels += 1
                                    # else: # Optional: Log questions where the pattern wasn't found
                                        # print(f"Warning: Could not extract label from: {question_text}")
        else:
            print("Error: Unexpected annotations format. Key 'data' not found.")
            return Counter()

        print(
            f"Processed {processed_qas} QAs, found {found_labels} valid clause labels.")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotations_path}")
        return Counter()
    except Exception as e:
        print(f"An error occurred during annotation processing: {e}")
        return Counter()

    print(f"Found {len(label_counts)} unique clause labels.")
    return label_counts


def plot_clause_histogram(label_counts: Counter, output_dir: str):
    """Generates and saves a histogram of clause frequencies."""
    if not label_counts:
        print("No label counts to plot.")
        return

    labels, frequencies = zip(*label_counts.most_common())  # Sort by frequency

    # Adjust figure size dynamically
    plt.figure(figsize=(12, max(8, len(labels) * 0.3)))
    plt.barh(labels, frequencies)
    plt.xlabel("Frequency")
    plt.ylabel("Clause Label")
    plt.title("CUAD Clause Frequency Distribution")
    plt.gca().invert_yaxis()  # Display most frequent clauses at the top
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "clause_frequency_histogram.png")
    try:
        plt.savefig(plot_path)
        print(f"Histogram saved to {plot_path}")
    except Exception as e:
        print(f"Error saving histogram: {e}")
    # plt.show() # Uncomment to display the plot interactively


if __name__ == "__main__":
    print("Starting CUAD processing...")

    # 1. Load PDFs, Extract Text, Count Tokens
    print("\n--- Step 1: Processing PDFs ---")
    extracted_data, flagged_docs = process_pdfs(PDF_DIR, TOKEN_LIMIT)
    print(f"\n--- Step 1 Complete ---")
    if flagged_docs:
        print(f"Documents exceeding {TOKEN_LIMIT} tokens:")
        for doc in flagged_docs:
            print(f" - {doc}")
    else:
        print("No documents exceeded the token limit.")

    # 2. Analyze Clause Frequencies from Annotations JSON
    print("\n--- Step 2: Analyzing Annotations ---")
    clause_counts = analyze_clause_frequency(ANNOTATIONS_PATH)
    print(f"\n--- Step 2 Complete ---")
    if clause_counts:
        print("Top 5 most frequent clauses:")
        for label, freq in clause_counts.most_common(5):
            print(f" - {label}: {freq}")

    # 3. Generate Histogram
    print("\n--- Step 3: Generating Histogram ---")
    plot_clause_histogram(clause_counts, PLOT_OUTPUT_DIR)
    print(f"\n--- Step 3 Complete ---")

    print("\nCUAD processing finished.")
