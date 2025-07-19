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

# Get the project root directory (3 levels up from this script)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Define paths with absolute references from project root
BASE_DIR = ROOT_DIR / "data/CUAD_v1"
PDF_DIR = BASE_DIR / "full_contract_pdf"
METADATA_FILE = ROOT_DIR / "data/created/contracts_metadata_only.json"
# Fix the path to the master_clauses.csv
CLAUSES_FILE = ROOT_DIR / "data/CUAD_v1/master_clauses.csv"
OUTPUT_DIR = ROOT_DIR / "data/jsonl"
NEW_TRAIN_JSONL = OUTPUT_DIR / "new_training.jsonl"
NEW_VAL_JSONL = OUTPUT_DIR / "new_validation.jsonl"
NEW_TEST_JSONL = OUTPUT_DIR / "new_test.jsonl"
# New paths for filename lists
TRAIN_FILENAMES = OUTPUT_DIR / "train_filenames.txt"
VAL_FILENAMES = OUTPUT_DIR / "validation_filenames.txt"
TEST_FILENAMES = OUTPUT_DIR / "test_filenames.txt"


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
    """Create balanced train/validation/test split (80/10/10)"""
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

    # Separate categories with fewer than 3 items (cannot split 80/10/10) and others
    small_categories = category_counts[category_counts < 3].index
    multi_item_categories = category_counts[category_counts >= 3].index

    logger.info(
        f"Found {len(small_categories)} categories with < 3 items (will go to training)")
    logger.info(
        f"Found {len(multi_item_categories)} categories with >= 3 items (will be split)")

    # Put small categories directly into training
    train_dfs = [df[df['category'].isin(small_categories)]]
    val_dfs = []
    test_dfs = []

    # Create size and page bins based on percentiles for the remaining data
    multi_item_df = df[df['category'].isin(multi_item_categories)]
    if not multi_item_df.empty:
        # Handle potential errors with qcut if too few unique values
        try:
            multi_item_df['size_bin'] = pd.qcut(
                multi_item_df['file_size'], q=2, labels=['small', 'large'], duplicates='drop')
        except ValueError:
            multi_item_df['size_bin'] = 'all'
            logger.warning(
                "Could not create size bins due to insufficient unique values.")
        try:
            multi_item_df['page_bin'] = pd.qcut(
                multi_item_df['page_count'], q=2, labels=['short', 'long'], duplicates='drop')
        except ValueError:
            multi_item_df['page_bin'] = 'all'
            logger.warning(
                "Could not create page bins due to insufficient unique values.")

        # Split multi-item categories
        for category in multi_item_categories:
            # Use copy to avoid SettingWithCopyWarning
            category_df = multi_item_df[multi_item_df['category'] == category].copy(
            )

            # Determine stratification columns
            stratify_cols = []
            if category_df['size_bin'].nunique() > 1:
                stratify_cols.append('size_bin')
            if category_df['page_bin'].nunique() > 1:
                stratify_cols.append('page_bin')

            stratify_on = None
            if len(stratify_cols) == 2:
                # Combine bins for stratification if both have variance
                category_df['stratify_key'] = category_df['size_bin'].astype(
                    str) + '_' + category_df['page_bin'].astype(str)
                if category_df['stratify_key'].nunique() > 1:
                    stratify_on = category_df['stratify_key']
            elif len(stratify_cols) == 1:
                # Use the single available bin for stratification
                if category_df[stratify_cols[0]].nunique() > 1:
                    stratify_on = category_df[stratify_cols[0]]

            # --- First Split: 80% Train, 20% Temp ---
            cat_train = None
            cat_temp = None
            try:
                if stratify_on is not None:
                    # Check if any stratum has only 1 member for the first split
                    counts = stratify_on.value_counts()
                    if (counts == 1).any():
                        logger.warning(
                            f"Category '{category}': Stratum with 1 member detected. Cannot stratify first split. Performing random split.")
                        stratify_on_split1 = None  # Fallback to random split
                    else:
                        stratify_on_split1 = stratify_on

                    cat_train, cat_temp = train_test_split(
                        category_df,
                        test_size=0.18,  # 20% goes to temp set
                        random_state=42,
                        stratify=stratify_on_split1
                    )
                else:
                    # No stratification possible or fallback
                    cat_train, cat_temp = train_test_split(
                        category_df,
                        test_size=0.18,
                        random_state=42
                    )

            except ValueError as e:
                logger.warning(
                    f"Category '{category}': Error during first split stratification: {e}. Falling back to random split.")
                cat_train, cat_temp = train_test_split(
                    category_df,
                    test_size=0.18,
                    random_state=42
                )

            # --- Second Split: 50% Validation, 50% Test from Temp ---
            cat_val = None
            cat_test = None
            # Need at least 2 samples to split
            if cat_temp is not None and len(cat_temp) >= 2:
                # Determine stratification for the second split based on the temp set
                stratify_on_split2 = None
                temp_stratify_key = None
                if stratify_on is not None:  # Check if original stratification was attempted
                    # Recalculate stratify key/column based *only* on the temp set indices
                    temp_stratify_col_name = stratify_on.name
                    temp_stratify_col = category_df.loc[cat_temp.index,
                                                        temp_stratify_col_name]

                    # Check if stratification is possible within the temp set
                    if temp_stratify_col.nunique() > 1:
                        counts_temp = temp_stratify_col.value_counts()
                        if not (counts_temp == 1).any():
                            stratify_on_split2 = temp_stratify_col
                        else:
                            logger.warning(
                                f"Category '{category}': Stratum with 1 member detected in temp set. Cannot stratify second split. Performing random split.")
                    else:
                        logger.warning(
                            f"Category '{category}': Not enough unique strata in temp set. Cannot stratify second split. Performing random split.")

                try:
                    cat_val, cat_test = train_test_split(
                        cat_temp,
                        # 50% of temp goes to test (10% of total)
                        test_size=0.50,
                        random_state=42,
                        stratify=stratify_on_split2
                    )
                except ValueError as e:
                    logger.warning(
                        f"Category '{category}': Error during second split stratification: {e}. Falling back to random split.")
                    cat_val, cat_test = train_test_split(
                        cat_temp,
                        test_size=0.50,
                        random_state=42
                    )
            elif cat_temp is not None and len(cat_temp) == 1:
                # If only one sample left in temp, assign it randomly (e.g., to validation)
                logger.warning(
                    f"Category '{category}': Only 1 sample in temp set after first split. Assigning to validation.")
                cat_val = cat_temp
                cat_test = pd.DataFrame()  # Empty dataframe for test
            else:
                # Handle case where cat_temp might be None or empty
                cat_val = pd.DataFrame()
                cat_test = pd.DataFrame()

            if cat_train is not None:
                train_dfs.append(cat_train)
            if cat_val is not None:
                val_dfs.append(cat_val)
            if cat_test is not None:
                test_dfs.append(cat_test)

    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    validation_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save filenames to text files
    logger.info(f"Saving filenames to text files...")
    with open(TRAIN_FILENAMES, 'w', encoding='utf-8') as f:
        for filename in train_df['filename']:
            f.write(f"{filename}\n")

    with open(VAL_FILENAMES, 'w', encoding='utf-8') as f:  # Save validation filenames
        for filename in validation_df['filename']:
            f.write(f"{filename}\n")

    with open(TEST_FILENAMES, 'w', encoding='utf-8') as f:
        for filename in test_df['filename']:
            f.write(f"{filename}\n")

    logger.info(f"Saved training filenames to {TRAIN_FILENAMES}")
    # Log validation filenames
    logger.info(f"Saved validation filenames to {VAL_FILENAMES}")
    logger.info(f"Saved test filenames to {TEST_FILENAMES}")

    # Create new JSONL files
    logger.info("Creating new training JSONL file...")
    train_count = 0
    with open(NEW_TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
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

    logger.info("Creating new validation JSONL file...")  # Process validation
    val_count = 0
    with open(NEW_VAL_JSONL, 'w', encoding='utf-8') as f:
        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df)):
            text = extract_text_from_pdf(row['path'])
            if not text:
                continue

            output_data = create_output_json(metadata[row['metadata_key']])
            json_line = {
                "prompt": text,
                "completion": json.dumps(output_data)
            }
            f.write(json.dumps(json_line) + '\n')
            val_count += 1

    logger.info("Creating new test JSONL file...")
    test_count = 0
    with open(NEW_TEST_JSONL, 'w', encoding='utf-8') as f:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
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
    logger.info(f"Validation examples: {val_count}")  # Add validation count
    logger.info(f"Test examples: {test_count}")
    logger.info(
        f"Training filenames: {len(train_df['filename'])} (saved to {TRAIN_FILENAMES})")
    logger.info(
        f"Validation filenames: {len(validation_df['filename'])} (saved to {VAL_FILENAMES})")  # Add validation filenames count
    logger.info(
        f"Test filenames: {len(test_df['filename'])} (saved to {TEST_FILENAMES})")

    # Calculate and print distribution statistics
    print("\nCategory Distribution:")
    print("\nRaw Counts:")
    cat_counts = pd.DataFrame({
        'Train': train_df['category'].value_counts() if not train_df.empty else pd.Series(dtype=int),
        # Add validation
        'Validation': validation_df['category'].value_counts() if not validation_df.empty else pd.Series(dtype=int),
        'Test': test_df['category'].value_counts() if not test_df.empty else pd.Series(dtype=int)
    }).fillna(0).astype(int)  # Fill NaNs with 0 and ensure integer type
    print(cat_counts)

    print("\nPercentages:")
    total_train = len(train_df)
    total_val = len(validation_df)
    total_test = len(test_df)

    cat_dist = pd.DataFrame({
        'Train %': train_df['category'].value_counts(normalize=True) * 100 if total_train > 0 else pd.Series(dtype=float),
        # Add validation
        'Validation %': validation_df['category'].value_counts(normalize=True) * 100 if total_val > 0 else pd.Series(dtype=float),
        'Test %': test_df['category'].value_counts(normalize=True) * 100 if total_test > 0 else pd.Series(dtype=float)
    }).fillna(0.0)  # Fill NaNs with 0.0 for percentages
    print(cat_dist.round(2))  # Round percentages for display

    # --- Detailed Stats (only if validation and test sets are non-empty) ---
    if not validation_df.empty and not test_df.empty:
        print("\nFile Size Statistics:")
        size_stats = pd.DataFrame({
            'Train': [train_df['file_size'].mean(), train_df['file_size'].median()],
            # Add validation
            'Validation': [validation_df['file_size'].mean(), validation_df['file_size'].median()],
            'Test': [test_df['file_size'].mean(), test_df['file_size'].median()]
        }, index=['Mean (KB)', 'Median (KB)'])
        print(size_stats.round(2))

        print("\nPage Count Statistics:")
        page_stats = pd.DataFrame({
            'Train': [train_df['page_count'].mean(), train_df['page_count'].median()],
            # Add validation
            'Validation': [validation_df['page_count'].mean(), validation_df['page_count'].median()],
            'Test': [test_df['page_count'].mean(), test_df['page_count'].median()]
        }, index=['Mean', 'Median'])
        print(page_stats.round(2))

        print("\nWord Count Statistics:")
        word_stats = pd.DataFrame({
            'Train': [train_df['word_count'].mean(), train_df['word_count'].median(),
                      train_df['word_count'].min(), train_df['word_count'].max()],
            'Validation': [validation_df['word_count'].mean(), validation_df['word_count'].median(),  # Add validation
                           validation_df['word_count'].min(), validation_df['word_count'].max()],
            'Test': [test_df['word_count'].mean(), test_df['word_count'].median(),
                     test_df['word_count'].min(), test_df['word_count'].max()]
        }, index=['Mean', 'Median', 'Min', 'Max'])
        print(word_stats.round(0).astype(int))  # Round counts to integer

        print("\nClause Count Statistics:")
        clause_stats = pd.DataFrame({
            'Train': [train_df['clause_count'].mean(), train_df['clause_count'].median(),
                      train_df['clause_count'].min(), train_df['clause_count'].max()],
            'Validation': [validation_df['clause_count'].mean(), validation_df['clause_count'].median(),  # Add validation
                           validation_df['clause_count'].min(), validation_df['clause_count'].max()],
            'Test': [test_df['clause_count'].mean(), test_df['clause_count'].median(),
                     test_df['clause_count'].min(), test_df['clause_count'].max()]
        }, index=['Mean', 'Median', 'Min', 'Max'])
        print(clause_stats.round(2))

        # Perform statistical tests (KS test comparing distributions)
        print("\nDistribution Tests (Kolmogorov-Smirnov):")
        # Compare Train vs Validation
        size_stat_tv, size_pval_tv = stats.ks_2samp(
            train_df['file_size'], validation_df['file_size'])
        page_stat_tv, page_pval_tv = stats.ks_2samp(
            train_df['page_count'], validation_df['page_count'])
        word_stat_tv, word_pval_tv = stats.ks_2samp(
            train_df['word_count'], validation_df['word_count'])
        clause_stat_tv, clause_pval_tv = stats.ks_2samp(
            train_df['clause_count'], validation_df['clause_count'])
        print("  Train vs Validation:")
        print(f"    File Size: p-value = {size_pval_tv:.4f}")
        print(f"    Page Count: p-value = {page_pval_tv:.4f}")
        print(f"    Word Count: p-value = {word_pval_tv:.4f}")
        print(f"    Clause Count: p-value = {clause_pval_tv:.4f}")

        # Compare Train vs Test
        size_stat_tt, size_pval_tt = stats.ks_2samp(
            train_df['file_size'], test_df['file_size'])
        page_stat_tt, page_pval_tt = stats.ks_2samp(
            train_df['page_count'], test_df['page_count'])
        word_stat_tt, word_pval_tt = stats.ks_2samp(
            train_df['word_count'], test_df['word_count'])
        clause_stat_tt, clause_pval_tt = stats.ks_2samp(
            train_df['clause_count'], test_df['clause_count'])
        print("  Train vs Test:")
        print(f"    File Size: p-value = {size_pval_tt:.4f}")
        print(f"    Page Count: p-value = {page_pval_tt:.4f}")
        print(f"    Word Count: p-value = {word_pval_tt:.4f}")
        print(f"    Clause Count: p-value = {clause_pval_tt:.4f}")

        # Compare Validation vs Test
        size_stat_vt, size_pval_vt = stats.ks_2samp(
            validation_df['file_size'], test_df['file_size'])
        page_stat_vt, page_pval_vt = stats.ks_2samp(
            validation_df['page_count'], test_df['page_count'])
        word_stat_vt, word_pval_vt = stats.ks_2samp(
            validation_df['word_count'], test_df['word_count'])
        clause_stat_vt, clause_pval_vt = stats.ks_2samp(
            validation_df['clause_count'], test_df['clause_count'])
        print("  Validation vs Test:")
        print(f"    File Size: p-value = {size_pval_vt:.4f}")
        print(f"    Page Count: p-value = {page_pval_vt:.4f}")
        print(f"    Word Count: p-value = {word_pval_vt:.4f}")
        print(f"    Clause Count: p-value = {clause_pval_vt:.4f}")

        # Create word count bins for visualization
        try:
            train_df['word_bin'] = pd.qcut(train_df['word_count'], q=4, labels=[
                                           'q1', 'q2', 'q3', 'q4'], duplicates='drop')
            validation_df['word_bin'] = pd.qcut(validation_df['word_count'], q=4, labels=[
                                                'q1', 'q2', 'q3', 'q4'], duplicates='drop')
            test_df['word_bin'] = pd.qcut(test_df['word_count'], q=4, labels=[
                                          'q1', 'q2', 'q3', 'q4'], duplicates='drop')
        except ValueError:
            logger.warning(
                "Could not create word bins due to insufficient unique values or samples.")
            train_df['word_bin'] = 'all'
            validation_df['word_bin'] = 'all'
            test_df['word_bin'] = 'all'

        print("\nWord Count Distribution (quartiles):")
        word_dist = pd.DataFrame({
            'Train %': train_df['word_bin'].value_counts(normalize=True) * 100,
            # Add validation
            'Validation %': validation_df['word_bin'].value_counts(normalize=True) * 100,
            'Test %': test_df['word_bin'].value_counts(normalize=True) * 100
        }).fillna(0.0)
        print(word_dist.round(2))

        # Create clause count distribution if we have clause data
        if clauses_df is not None and 'clause_count' in train_df.columns and train_df['clause_count'].nunique() >= 4:
            try:
                train_df['clause_bin'] = pd.qcut(train_df['clause_count'], q=4, labels=[
                                                 'q1', 'q2', 'q3', 'q4'], duplicates='drop')
                validation_df['clause_bin'] = pd.qcut(validation_df['clause_count'], q=4, labels=[
                                                      'q1', 'q2', 'q3', 'q4'], duplicates='drop')
                test_df['clause_bin'] = pd.qcut(test_df['clause_count'], q=4, labels=[
                                                'q1', 'q2', 'q3', 'q4'], duplicates='drop')
            except ValueError:
                logger.warning(
                    "Could not create clause bins due to insufficient unique values or samples.")
                train_df['clause_bin'] = 'all'
                validation_df['clause_bin'] = 'all'
                test_df['clause_bin'] = 'all'

            print("\nClause Count Distribution (quartiles):")
            clause_dist = pd.DataFrame({
                'Train %': train_df['clause_bin'].value_counts(normalize=True) * 100,
                # Add validation
                'Validation %': validation_df['clause_bin'].value_counts(normalize=True) * 100,
                'Test %': test_df['clause_bin'].value_counts(normalize=True) * 100
            }).fillna(0.0)
            print(clause_dist.round(2))
        elif clauses_df is not None:
            logger.warning(
                "Could not create clause count bins for distribution display (need >= 4 unique clause counts).")

    elif not train_df.empty:
        logger.warning(
            "Validation or Test set is empty, skipping detailed comparative statistics.")
        # Print basic stats for train if available
        print("\nFile Size Statistics (Train only):")
        print(f"  Mean (KB): {train_df['file_size'].mean():.2f}")
        print(f"  Median (KB): {train_df['file_size'].median():.2f}")
        print("\nPage Count Statistics (Train only):")
        print(f"  Mean: {train_df['page_count'].mean():.2f}")
        print(f"  Median: {train_df['page_count'].median():.2f}")
        print("\nWord Count Statistics (Train only):")
        print(f"  Mean: {train_df['word_count'].mean():.0f}")
        print(f"  Median: {train_df['word_count'].median():.0f}")
        print(f"  Min: {train_df['word_count'].min():.0f}")
        print(f"  Max: {train_df['word_count'].max():.0f}")
        print("\nClause Count Statistics (Train only):")
        print(f"  Mean: {train_df['clause_count'].mean():.2f}")
        print(f"  Median: {train_df['clause_count'].median():.2f}")
        print(f"  Min: {train_df['clause_count'].min():.0f}")
        print(f"  Max: {train_df['clause_count'].max():.0f}")
    else:
        logger.warning(
            "All sets (Train, Validation, Test) are empty. No statistics to display.")

    logger.info(f"\nNew files saved to:")
    logger.info(f"Training: {NEW_TRAIN_JSONL}")
    logger.info(f"Validation: {NEW_VAL_JSONL}")  # Add validation
    logger.info(f"Test: {NEW_TEST_JSONL}")
    logger.info(f"Training filenames: {TRAIN_FILENAMES}")
    logger.info(f"Validation filenames: {VAL_FILENAMES}")  # Add validation
    logger.info(f"Test filenames: {TEST_FILENAMES}")


if __name__ == "__main__":
    create_balanced_split()
