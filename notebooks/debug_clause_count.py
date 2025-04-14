import pandas as pd
import os
import re
import logging
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Set up logging with more verbose output for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
CLAUSES_FILE = Path("data/CUAD_v1/master_clauses.csv")
# The problematic file
TEST_FILE = "LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT.PDF"


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


def get_clause_count(filename, clauses_df):
    """Get the number of clauses (yes answers) for a given contract"""
    if clauses_df is None:
        return 0

    # Print column names for debugging
    print(f"Columns in clauses_df: {clauses_df.columns.tolist()}")

    # Get the normalized filename for matching
    normalized_pdf_name = normalize_filename(filename)
    print(f"Normalized filename: {normalized_pdf_name}")

    # Get all answer columns (ending with -Answer)
    answer_columns = [
        col for col in clauses_df.columns if col.endswith('-Answer')]
    print(
        f"Found {len(answer_columns)} answer columns: {answer_columns[:5]}...")

    # Try to match against each row in the dataframe
    print("\nSearching for matching file...")

    # Check first if 'Filename' column exists
    if 'Filename' in clauses_df.columns:
        print("Using 'Filename' column for matching")

        # Print a few sample Filename entries
        print("Sample entries in 'Filename' column:")
        for i, fname in enumerate(clauses_df['Filename'].head(3)):
            print(
                f"  {i+1}. '{fname}' -> normalized: '{normalize_filename(str(fname))}'")

        # Try direct match
        direct_match = clauses_df[clauses_df['Filename'] == filename]
        if len(direct_match) > 0:
            print(f"Found direct match!")
            contract_row = direct_match
        else:
            print("No direct match found, trying normalized matching...")
            found = False

            # Try each row with normalized comparison
            for idx, row in tqdm(clauses_df.iterrows(), total=len(clauses_df)):
                csv_filename = str(row['Filename'])
                normalized_csv_name = normalize_filename(csv_filename)

                if normalized_csv_name == normalized_pdf_name:
                    print(f"Found exact normalized match: '{csv_filename}'")
                    contract_row = clauses_df.iloc[[idx]]
                    found = True
                    break
                elif normalized_pdf_name in normalized_csv_name or normalized_csv_name in normalized_pdf_name:
                    print(f"Found partial normalized match: '{csv_filename}'")
                    contract_row = clauses_df.iloc[[idx]]
                    found = True
                    break

            if not found:
                print("No match found in 'Filename' column")
                # Try Document Name column if it exists
                if 'Document Name' in clauses_df.columns:
                    print("\nTrying 'Document Name' column...")

                    # Print a few sample Document Name entries
                    print("Sample entries in 'Document Name' column:")
                    for i, name in enumerate(clauses_df['Document Name'].head(3)):
                        print(
                            f"  {i+1}. '{name}' -> normalized: '{normalize_filename(str(name))}'")

                    for idx, row in tqdm(clauses_df.iterrows(), total=len(clauses_df)):
                        doc_name = str(row['Document Name'])
                        normalized_doc_name = normalize_filename(doc_name)

                        if normalized_doc_name == normalized_pdf_name:
                            print(
                                f"Found exact normalized match in Document Name: '{doc_name}'")
                            contract_row = clauses_df.iloc[[idx]]
                            found = True
                            break
                        elif normalized_pdf_name in normalized_doc_name or normalized_doc_name in normalized_pdf_name:
                            print(
                                f"Found partial normalized match in Document Name: '{doc_name}'")
                            contract_row = clauses_df.iloc[[idx]]
                            found = True
                            break

                if not found:
                    print("No match found in either column")
                    return 0
    else:
        print("'Filename' column not found in clauses_df")
        return 0

    # Count 'Yes' values
    print("\nCounting 'Yes' values in answer columns...")
    yes_count = 0
    for col in answer_columns:
        try:
            if col in contract_row.columns:
                value = str(contract_row[col].iloc[0]).strip().lower()
                if value == 'yes':
                    yes_count += 1
                    print(f"Found 'yes' in {col}")
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")

    print(f"\nTotal 'Yes' count: {yes_count}")
    return yes_count


def main():
    """Main function to test the get_clause_count function"""
    # Load the clauses data
    print(f"Loading clauses from {CLAUSES_FILE}")

    # Try different encodings
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            print(f"\nTrying encoding: {encoding}")
            clauses_df = pd.read_csv(CLAUSES_FILE, encoding=encoding)
            print(f"Successfully loaded {len(clauses_df)} rows of clause data")
            print(f"Data shape: {clauses_df.shape}")

            # Extract a few rows to check content
            print("\nFirst few rows:")
            print(clauses_df.head(1).to_string())

            # Get the clause count for the test file
            print(f"\nTesting with file: {TEST_FILE}")
            count = get_clause_count(TEST_FILE, clauses_df)
            print(f"Final clause count for {TEST_FILE}: {count}")

            # If we got here, break out of the loop
            break

        except Exception as e:
            print(f"Error with encoding {encoding}: {str(e)}")
    else:
        print("\nFailed to load the file with any of the attempted encodings.")


if __name__ == "__main__":
    main()
