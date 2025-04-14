import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import os
from PyPDF2 import PdfReader
import json
from tqdm import tqdm
import logging
import re
from scipy import stats
import unicodedata

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
# Fix the path to the master_clauses.csv
CLAUSES_FILE = Path("data/CUAD_v1/master_clauses.csv")
OUTPUT_DIR = Path("data/jsonl")
NEW_TRAIN_JSONL = OUTPUT_DIR / "new_training.jsonl"
NEW_TEST_JSONL = OUTPUT_DIR / "new_test.jsonl"


def normalize_filename(filename):
    """Normalize filename for comparison by removing special characters and converting to lowercase"""
    # Remove file extension
    filename = os.path.splitext(filename)[0]
    # Convert accented characters to ASCII equivalents (é -> e)
    filename = unicodedata.normalize('NFKD', filename).encode(
        'ASCII', 'ignore').decode('ASCII')
    # Remove special characters and convert to lowercase
    normalized = re.sub(r'[^a-zA-Z0-9]', '', filename).lower()
    return normalized


def get_pdf_files(directory):
    """Get all PDF files recursively"""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def get_category(file_path):
    """Get category from file path structure: full_contract_pdf/part_x/category/contract"""
    parts = Path(file_path).parts
    try:
        # Find the index of 'full_contract_pdf' in the path
        pdf_dir_idx = parts.index('full_contract_pdf')
        # Category should be two levels down from 'full_contract_pdf'
        # full_contract_pdf (pdf_dir_idx) / part_x (pdf_dir_idx + 1) / category (pdf_dir_idx + 2)
        if len(parts) > pdf_dir_idx + 2:
            return parts[pdf_dir_idx + 2]
    except ValueError:
        pass
    return "unknown"


def get_file_size(file_path):
    """Get file size in KB"""
    return os.path.getsize(file_path) / 1024


def get_page_count(file_path):
    """Get number of pages in PDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return 0


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return None


def create_output_json(metadata_entry):
    """Create JSON output from metadata entry"""
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


def find_metadata_key(filename, metadata):
    """Find metadata key for a given filename using normalized comparison"""
    normalized_filename = normalize_filename(filename)

    # First try exact match after normalization
    for key in metadata:
        if normalize_filename(key) == normalized_filename:
            return key

    # If no exact match, try partial match
    for key in metadata:
        if normalized_filename in normalize_filename(key) or normalize_filename(key) in normalized_filename:
            return key

    return None


def get_word_count(text):
    """Count words in text, handling None values"""
    if text is None:
        return 0
    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def get_clause_count(filename, clauses_df):
    """Get the number of clauses (yes answers) for a given contract"""
    if clauses_df is None:
        return 0

    # Debug - print column names
    logger.debug(f"Clauses columns: {clauses_df.columns.tolist()}")

    # Get any column that ends with -Answer (these are the clause columns)
    answer_columns = [
        col for col in clauses_df.columns if col.endswith('-Answer')]
    logger.debug(f"Found {len(answer_columns)} answer columns")

    # Get the normalized filename for matching
    normalized_pdf_name = normalize_filename(filename)

    # Try to find the contract in the clauses dataframe using different strategies
    contract_row = None

    # Strategy 1: Direct match on Filename
    if 'Filename' in clauses_df.columns:
        contract_row = clauses_df[clauses_df['Filename'] == filename]
        if len(contract_row) > 0:
            logger.debug(f"Found contract via direct Filename match")

    # Strategy 2: Normalized match on Filename
    if contract_row is None or len(contract_row) == 0:
        if 'Filename' in clauses_df.columns:
            for idx, row in clauses_df.iterrows():
                csv_filename = str(row['Filename'])
                normalized_csv_name = normalize_filename(csv_filename)

                if normalized_csv_name == normalized_pdf_name:
                    contract_row = clauses_df.iloc[[idx]]
                    logger.debug(
                        f"Found contract via normalized Filename match")
                    break
                elif normalized_pdf_name in normalized_csv_name or normalized_csv_name in normalized_pdf_name:
                    contract_row = clauses_df.iloc[[idx]]
                    logger.debug(f"Found contract via partial Filename match")
                    break

    # Strategy 3: Try matching on Document Name column if it exists
    if (contract_row is None or len(contract_row) == 0) and 'Document Name' in clauses_df.columns:
        for idx, row in clauses_df.iterrows():
            doc_name = str(row['Document Name'])
            normalized_doc_name = normalize_filename(doc_name)

            if normalized_doc_name == normalized_pdf_name:
                contract_row = clauses_df.iloc[[idx]]
                logger.debug(
                    f"Found contract via normalized Document Name match")
                break
            elif normalized_pdf_name in normalized_doc_name or normalized_doc_name in normalized_pdf_name:
                contract_row = clauses_df.iloc[[idx]]
                logger.debug(f"Found contract via partial Document Name match")
                break

    # If still no match, log and return 0
    if contract_row is None or len(contract_row) == 0:
        logger.warning(f"No clause data found for {filename}")
        return 0

    # Count 'Yes' values across answer columns (case-insensitive)
    yes_count = 0
    for col in answer_columns:
        try:
            if col in contract_row.columns:
                value = str(contract_row[col].iloc[0]).strip().lower()
                if value == 'yes':
                    yes_count += 1
        except Exception as e:
            logger.error(f"Error processing column {col}: {str(e)}")

    logger.info(f"Found {yes_count} clauses for {filename}")
    return yes_count


def create_balanced_split():
    """Create balanced train/test split"""
    # Load metadata
    logger.info(f"Loading metadata from {METADATA_FILE}")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load clauses data
    try:
        logger.info(f"Loading clauses from {CLAUSES_FILE}")
        clauses_df = pd.read_csv(CLAUSES_FILE, encoding='latin1')
        logger.info(f"Loaded {len(clauses_df)} clause entries")
    except Exception as e:
        logger.warning(f"Could not load clauses file: {str(e)}")
        clauses_df = None

    # Get all PDF files
    all_pdf_files = get_pdf_files(PDF_DIR)
    logger.info(f"Found {len(all_pdf_files)} PDF files")

    # Create DataFrame with file information
    data = []
    skipped_files = []
    logger.info("Collecting file information...")
    for pdf_path in tqdm(all_pdf_files):
        filename = os.path.basename(pdf_path)

        # Find metadata for this file using the new matching function
        metadata_key = find_metadata_key(filename, metadata)

        if not metadata_key:
            logger.warning(f"No metadata found for {filename}, skipping")
            skipped_files.append(filename)
            continue

        # Extract text and get word count
        text = extract_text_from_pdf(pdf_path)
        word_count = get_word_count(text)

        # Get clause count
        clause_count = get_clause_count(filename, clauses_df)

        # Collect file information
        category = get_category(pdf_path)
        file_size = get_file_size(pdf_path)
        page_count = get_page_count(pdf_path)

        data.append({
            'path': pdf_path,
            'filename': filename,
            'category': category,
            'file_size': file_size,
            'page_count': page_count,
            'word_count': word_count,
            'clause_count': clause_count,
            'metadata_key': metadata_key
        })

    df = pd.DataFrame(data)

    # Log summary of skipped files
    if skipped_files:
        logger.info(
            f"\nSkipped {len(skipped_files)} files due to missing metadata:")
        for idx, filename in enumerate(skipped_files[:10], 1):
            logger.info(f"{idx}. {filename}")
        if len(skipped_files) > 10:
            logger.info(f"... and {len(skipped_files) - 10} more")

    # Get category counts
    category_counts = df['category'].value_counts()

    # Separate single-item categories and multi-item categories
    single_item_categories = category_counts[category_counts == 1].index
    multi_item_categories = category_counts[category_counts > 1].index

    logger.info(
        f"Found {len(single_item_categories)} categories with single items (will go to training)")
    logger.info(
        f"Found {len(multi_item_categories)} categories with multiple items (will be split)")

    # Put single-item categories directly into training
    train_dfs = [df[df['category'].isin(single_item_categories)]]
    test_dfs = []

    # Create size and page bins based on percentiles for the remaining data
    multi_item_df = df[df['category'].isin(multi_item_categories)]
    if not multi_item_df.empty:
        multi_item_df['size_bin'] = pd.qcut(
            multi_item_df['file_size'], q=2, labels=['small', 'large'])
        multi_item_df['page_bin'] = pd.qcut(
            multi_item_df['page_count'], q=2, labels=['short', 'long'])

        # Split multi-item categories
        for category in multi_item_categories:
            category_df = multi_item_df[multi_item_df['category'] == category]

            if len(category_df) >= 10:  # Only stratify if we have enough samples
                try:
                    # Try to stratify by both size and page bins
                    cat_train, cat_test = train_test_split(
                        category_df,
                        test_size=0.15,
                        random_state=42,
                        stratify=category_df[['size_bin', 'page_bin']].apply(
                            lambda x: f"{x['size_bin']}_{x['page_bin']}", axis=1)
                    )
                except ValueError:
                    try:
                        # If that fails, try to stratify by just size
                        cat_train, cat_test = train_test_split(
                            category_df,
                            test_size=0.2,
                            random_state=42,
                            stratify=category_df['size_bin']
                        )
                    except ValueError:
                        # If that fails too, don't stratify
                        cat_train, cat_test = train_test_split(
                            category_df,
                            test_size=0.2,
                            random_state=42
                        )
            else:
                # For small categories, don't stratify
                cat_train, cat_test = train_test_split(
                    category_df,
                    test_size=0.2,
                    random_state=42
                )

            train_dfs.append(cat_train)
            test_dfs.append(cat_test)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create new JSONL files
    logger.info("Creating new training JSONL file...")
    train_count = 0
    with open(NEW_TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for _, row in tqdm(train_df.iterrows()):
            text = extract_text_from_pdf(row['path'])
            if not text:
                continue

            output_data = create_output_json(metadata[row['metadata_key']])
            json_line = {
                "prompt": text,
                "completion": json.dumps(output_data)
            }
            f.write(json.dumps(json_line) + '\n')
            train_count += 1

    logger.info("Creating new test JSONL file...")
    test_count = 0
    with open(NEW_TEST_JSONL, 'w', encoding='utf-8') as f:
        for _, row in tqdm(test_df.iterrows()):
            text = extract_text_from_pdf(row['path'])
            if not text:
                continue

            output_data = create_output_json(metadata[row['metadata_key']])
            json_line = {
                "prompt": text,
                "completion": json.dumps(output_data)
            }
            f.write(json.dumps(json_line) + '\n')
            test_count += 1

    # Print statistics
    logger.info("\nSplit Statistics:")
    logger.info(f"Training examples: {train_count}")
    logger.info(f"Test examples: {test_count}")

    # Calculate and print distribution statistics
    print("\nCategory Distribution:")
    print("\nRaw Counts:")
    cat_counts = pd.DataFrame({
        'Train': train_df['category'].value_counts(),
        'Test': test_df['category'].value_counts() if not test_df.empty else pd.Series(dtype=float)
    })
    print(cat_counts)

    print("\nPercentages:")
    cat_dist = pd.DataFrame({
        'Train %': train_df['category'].value_counts(normalize=True) * 100,
        'Test %': test_df['category'].value_counts(normalize=True) * 100 if not test_df.empty else pd.Series(dtype=float)
    })
    print(cat_dist)

    if not test_df.empty:
        print("\nFile Size Statistics:")
        size_stats = pd.DataFrame({
            'Train': [train_df['file_size'].mean(), train_df['file_size'].median()],
            'Test': [test_df['file_size'].mean(), test_df['file_size'].median()]
        }, index=['Mean (KB)', 'Median (KB)'])
        print(size_stats)

        print("\nPage Count Statistics:")
        page_stats = pd.DataFrame({
            'Train': [train_df['page_count'].mean(), train_df['page_count'].median()],
            'Test': [test_df['page_count'].mean(), test_df['page_count'].median()]
        }, index=['Mean', 'Median'])
        print(page_stats)

        print("\nWord Count Statistics:")
        word_stats = pd.DataFrame({
            'Train': [train_df['word_count'].mean(), train_df['word_count'].median(),
                      train_df['word_count'].min(), train_df['word_count'].max()],
            'Test': [test_df['word_count'].mean(), test_df['word_count'].median(),
                     test_df['word_count'].min(), test_df['word_count'].max()]
        }, index=['Mean', 'Median', 'Min', 'Max'])
        print(word_stats)

        print("\nClause Count Statistics:")
        clause_stats = pd.DataFrame({
            'Train': [train_df['clause_count'].mean(), train_df['clause_count'].median(),
                      train_df['clause_count'].min(), train_df['clause_count'].max()],
            'Test': [test_df['clause_count'].mean(), test_df['clause_count'].median(),
                     test_df['clause_count'].min(), test_df['clause_count'].max()]
        }, index=['Mean', 'Median', 'Min', 'Max'])
        print(clause_stats)

        # Perform statistical tests
        size_stat, size_pval = stats.ks_2samp(
            train_df['file_size'], test_df['file_size'])
        page_stat, page_pval = stats.ks_2samp(
            train_df['page_count'], test_df['page_count'])
        word_stat, word_pval = stats.ks_2samp(
            train_df['word_count'], test_df['word_count'])
        clause_stat, clause_pval = stats.ks_2samp(
            train_df['clause_count'], test_df['clause_count'])

        print("\nDistribution Tests (Kolmogorov-Smirnov):")
        print(f"File Size: p-value = {size_pval:.4f}")
        print(f"Page Count: p-value = {page_pval:.4f}")
        print(f"Word Count: p-value = {word_pval:.4f}")
        print(f"Clause Count: p-value = {clause_pval:.4f}")

        # Create word count and clause count bins for visualization
        train_df['word_bin'] = pd.qcut(train_df['word_count'], q=4, labels=[
                                       'very_short', 'short', 'long', 'very_long'])
        test_df['word_bin'] = pd.qcut(test_df['word_count'], q=4, labels=[
                                      'very_short', 'short', 'long', 'very_long'])

        print("\nWord Count Distribution (quartiles):")
        word_dist = pd.DataFrame({
            'Train': train_df['word_bin'].value_counts(normalize=True),
            'Test': test_df['word_bin'].value_counts(normalize=True)
        })
        print(word_dist)

        # Create clause count distribution if we have clause data
        if clauses_df is not None:
            train_df['clause_bin'] = pd.qcut(train_df['clause_count'], q=4, labels=[
                'very_few', 'few', 'many', 'very_many'])
            test_df['clause_bin'] = pd.qcut(test_df['clause_count'], q=4, labels=[
                'very_few', 'few', 'many', 'very_many'])

            print("\nClause Count Distribution (quartiles):")
            clause_dist = pd.DataFrame({
                'Train': train_df['clause_bin'].value_counts(normalize=True),
                'Test': test_df['clause_bin'].value_counts(normalize=True)
            })
            print(clause_dist)

    logger.info(f"\nNew files saved to:")
    logger.info(f"Training: {NEW_TRAIN_JSONL}")
    logger.info(f"Test: {NEW_TEST_JSONL}")


if __name__ == "__main__":
    create_balanced_split()
