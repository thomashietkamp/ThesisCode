import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_master_clauses(file_path):
    """
    Load the master clauses CSV file.
    Tries different encodings if the default UTF-8 fails.
    """
    print(f"Loading master clauses from: {file_path}")

    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']

    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")

    raise ValueError(
        f"Could not read file with any of the attempted encodings: {encodings}")


def identify_categories(df):
    """
    Identify the categories in the dataset.
    Returns a list of all unique categories found in column names.
    """
    # Extract all column names without -Answer suffix
    categories = set()
    for col in df.columns:
        if col.endswith('-Answer'):
            category = col.replace('-Answer', '')
            categories.add(category)

    return sorted(list(categories))


def create_subcategory_splits(df, output_dir):
    """
    Split the dataset into 5 subcategories as described in the methodology:
    1. Intellectual Property & Licensing Agent
    2. Competition & Exclusivity Agent
    3. Termination & Control Rights Agent
    4. Financial & Commercial Terms Agent
    5. Legal Protections & Liability Agent
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the 5 subcategories and their associated clause types
    subcategories = {
        "intellectual_property_licensing": [
            "Ip Ownership Assignment",
            "Joint Ip Ownership",
            "License Grant",
            "Non-Transferable License",
            "Affiliate License-Licensor",
            "Affiliate License-Licensee",
            "Unlimited/All-You-Can-Eat-License",
            "Irrevocable Or Perpetual License",
            "Source Code Escrow"
        ],
        "competition_exclusivity": [
            "Most Favored Nation",
            "Competitive Restriction Exception",
            "Non-Compete",
            "Exclusivity",
            "No-Solicit Of Customers",
            "No-Solicit Of Employees",
            "Non-Disparagement",
            "Rofr/Rofo/Rofn"
        ],
        "termination_control": [
            "Termination For Convenience",
            "Change Of Control",
            "Anti-Assignment",
            "Post-Termination Services"
        ],
        "financial_commercial": [
            "Revenue/Profit Sharing",
            "Price Restrictions",
            "Minimum Commitment",
            "Volume Restriction",
            "Audit Rights"
        ],
        "legal_protections_liability": [
            "Uncapped Liability",
            "Cap On Liability",
            "Liquidated Damages",
            "Warranty Duration",
            "Insurance",
            "Covenant Not To Sue",
            "Third Party Beneficiary"
        ]
    }

    # Create datasets for each subcategory
    for subcategory_name, clauses in subcategories.items():
        print(f"Creating dataset for subcategory: {subcategory_name}")

        # Select relevant columns
        selected_columns = ["Filename", "Document Name"]

        # Add the category and category-Answer columns for each clause type
        for clause in clauses:
            if clause in df.columns:
                selected_columns.append(clause)
            if f"{clause}-Answer" in df.columns:
                selected_columns.append(f"{clause}-Answer")

        # Create the subcategory dataset
        subcategory_df = df[selected_columns].copy()

        # Save to CSV
        output_file = os.path.join(
            output_dir, f"{subcategory_name}_clauses.csv")
        subcategory_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved {len(subcategory_df)} rows to {output_file}")

    print("All subcategory datasets created successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Split master clauses into 5 subcategory datasets")
    parser.add_argument("--input", type=str, default="data/CUAD_v1/master_clauses.csv",
                        help="Path to the master clauses CSV file")
    parser.add_argument("--output_dir", type=str, default="data/CUAD_v1/subcategories",
                        help="Directory to save the subcategory datasets")

    args = parser.parse_args()

    # Create absolute paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(os.getcwd()) / input_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(os.getcwd()) / output_dir

    # Load data
    df = load_master_clauses(input_path)

    # Print dataset info
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

    # Create subcategory splits
    create_subcategory_splits(df, output_dir)


if __name__ == "__main__":
    main()
