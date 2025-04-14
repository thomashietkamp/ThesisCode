#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import logging
from tqdm import tqdm
from src.utils.contract_to_text import PDFTextExtractor
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path("data/CUAD_v1")
PDF_DIR = BASE_DIR / "full_contract_pdf"
METADATA_FILE = Path("data/created/contracts_metadata_only.json")
OUTPUT_DIR = Path("data/jsonl")
TRAIN_JSONL = OUTPUT_DIR / "training.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PDFTextExtractor"""
    extractor = PDFTextExtractor()
    if extractor.load_pdf(pdf_path):
        text = extractor.extract_text()
        extractor.close()
        return text
    else:
        logger.error(f"Failed to load PDF: {pdf_path}")
        return None


def create_output_json(metadata_entry):
    """Create a JSON object with the required fields from metadata"""
    return {
        "document_name": metadata_entry.get("Document Name-Answer", ""),
        "parties": metadata_entry.get("Parties-Answer", ""),
        "agreement_date": metadata_entry.get("Agreement Date-Answer", ""),
        "effective_date": metadata_entry.get("Effective Date-Answer", ""),
        "expiration_date": metadata_entry.get("Expiration Date-Answer", ""),
        "renewal_term": metadata_entry.get("Renewal Term-Answer", ""),
        "notice_period_to_terminate_renewal": metadata_entry.get("Notice Period To Terminate Renewal- Answer", ""),
        "governing_law": metadata_entry.get("Governing Law-Answer", "")
    }


def find_missing_pdfs(metadata, all_pdf_files):
    """Find PDFs that are in metadata but not in file system"""
    all_pdf_basenames = [os.path.basename(
        pdf_path) for pdf_path in all_pdf_files]
    missing_pdfs = []

    for key in metadata.keys():
        found = False
        for pdf_basename in all_pdf_basenames:
            if pdf_basename in key:
                found = True
                break

        if not found:
            missing_pdfs.append(key)

    return missing_pdfs


def get_pdf_files(directory):
    """Get all PDF files in a directory with both lowercase and uppercase extensions"""
    # Use glob directly with both patterns
    lowercase_pdfs = glob.glob(str(directory / "**" / "*.pdf"), recursive=True)
    uppercase_pdfs = glob.glob(str(directory / "**" / "*.PDF"), recursive=True)
    return lowercase_pdfs + uppercase_pdfs


def main():
    """Main function to create JSONL files"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata
    logger.info(f"Loading metadata from {METADATA_FILE}")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata for {len(metadata)} contracts")

    # Get all PDF files in Part I, II, and III using the new function
    part1_files = get_pdf_files(PDF_DIR / "Part_I")
    part2_files = get_pdf_files(PDF_DIR / "Part_II")
    part3_files = get_pdf_files(PDF_DIR / "Part_III")

    logger.info(f"Found {len(part1_files)} PDF files in Part I")
    logger.info(f"Found {len(part2_files)} PDF files in Part II")
    logger.info(f"Found {len(part3_files)} PDF files in Part III")

    # Find missing PDFs
    all_pdf_files = part1_files + part2_files + part3_files
    missing_pdfs = find_missing_pdfs(metadata, all_pdf_files)
    logger.info(
        f"Found {len(missing_pdfs)} contracts in metadata that don't have corresponding PDFs")
    if missing_pdfs:
        logger.info(f"First 10 missing PDFs: {missing_pdfs[:10]}")

    # Count how many PDFs in filesystem have metadata
    pdfs_with_metadata = 0
    pdfs_without_metadata = []

    for pdf_path in all_pdf_files:
        filename = os.path.basename(pdf_path)
        metadata_key = None
        for key in metadata:
            if filename in key:
                metadata_key = key
                break

        if metadata_key:
            pdfs_with_metadata += 1
        else:
            pdfs_without_metadata.append(filename)

    logger.info(
        f"PDFs with metadata: {pdfs_with_metadata} out of {len(all_pdf_files)}")
    logger.info(f"PDFs without metadata: {len(pdfs_without_metadata)}")
    if pdfs_without_metadata:
        logger.info(
            f"First 10 PDFs without metadata: {pdfs_without_metadata[:10]}")

    # Create training JSONL file from Part I and Part II
    logger.info(f"Creating training JSONL file from Part I and Part III")
    training_count = 0
    with open(TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for pdf_path in tqdm(part1_files + part3_files):
            filename = os.path.basename(pdf_path)

            # Find metadata for this file
            metadata_key = None
            for key in metadata:
                if filename in key:
                    metadata_key = key
                    break

            if not metadata_key:
                logger.warning(f"No metadata found for {filename}, skipping")
                continue

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(
                    f"Failed to extract text from {pdf_path}, skipping")
                continue

            # Create JSON line
            output_data = create_output_json(metadata[metadata_key])
            json_line = {
                "prompt": text,
                "completion": json.dumps(output_data)
            }
            f.write(json.dumps(json_line) + '\n')
            training_count += 1

    # Create test JSONL file from Part III
    logger.info(f"Creating test JSONL file from Part III")
    test_count = 0
    with open(TEST_JSONL, 'w', encoding='utf-8') as f:
        for pdf_path in tqdm(part2_files):
            filename = os.path.basename(pdf_path)

            # Find metadata for this file
            metadata_key = None
            for key in metadata:
                if filename in key:
                    metadata_key = key
                    break

            if not metadata_key:
                logger.warning(f"No metadata found for {filename}, skipping")
                continue

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(
                    f"Failed to extract text from {pdf_path}, skipping")
                continue

            # Create JSON line
            output_data = create_output_json(metadata[metadata_key])
            json_line = {
                "prompt": text,
                "completion": json.dumps(output_data)
            }
            f.write(json.dumps(json_line) + '\n')
            test_count += 1

    logger.info(f"Training examples: {training_count}")
    logger.info(f"Test examples: {test_count}")
    logger.info(f"Training data saved to {TRAIN_JSONL}")
    logger.info(f"Test data saved to {TEST_JSONL}")


if __name__ == "__main__":
    main()
