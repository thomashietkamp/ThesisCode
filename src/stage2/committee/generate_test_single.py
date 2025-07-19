#!/usr/bin/env python3
"""
generate_test_single.py - End-to-End Contract Processing
Author: Thomas Hietkamp

This script provides a simplified end-to-end pipeline for processing test contracts:
1. Reads test contract filenames from test_filenames.txt
2. Loads actual contract text from CUAD dataset
3. Performs direct legal analysis with full contract text
4. Generates comprehensive legal reports

Uses Qwen 72B via OpenRouter with thinking capabilities for direct contract analysis.
"""

from __future__ import annotations
import argparse
import json
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import tqdm

# Import the LLM wrapper
from llm_wrapper_openrouter import chat_with_thinking

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
TEST_FILENAMES_PATH = BASE.parent.parent.parent / \
    "data" / "split_filenames" / "test_filenames.txt"
CUAD_DATA_PATH = BASE.parent.parent.parent / "data" / "CUAD_v1" / "CUAD_v1.json"
PROMPTS_YAML_PATH = BASE / "prompts.yaml"
OUTPUT_DIR = BASE.parent / "data" / "test_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
DEFAULT_MODEL = "qwen/qwen3-32b:free"
ANALYSIS_TEMPERATURE = 0.3


@dataclass
class ContractInfo:
    """Information about a contract file."""
    filename: str
    contract_type: str
    company: str
    date: Optional[str] = None

    @classmethod
    def from_filename(cls, filename: str) -> "ContractInfo":
        """Parse contract information from filename."""
        # Extract contract type from filename
        contract_type = "Unknown"
        if "MAINTENANCE" in filename.upper():
            contract_type = "Maintenance Agreement"
        elif "LICENSE" in filename.upper():
            contract_type = "License Agreement"
        elif "DISTRIBUTOR" in filename.upper():
            contract_type = "Distributor Agreement"
        elif "STRATEGIC ALLIANCE" in filename.upper():
            contract_type = "Strategic Alliance Agreement"
        elif "SPONSORSHIP" in filename.upper():
            contract_type = "Sponsorship Agreement"
        elif "DEVELOPMENT" in filename.upper():
            contract_type = "Development Agreement"
        elif "SERVICE" in filename.upper():
            contract_type = "Service Agreement"
        elif "COLLABORATION" in filename.upper():
            contract_type = "Collaboration Agreement"
        elif "ENDORSEMENT" in filename.upper():
            contract_type = "Endorsement Agreement"
        elif "JOINT VENTURE" in filename.upper():
            contract_type = "Joint Venture Agreement"
        elif "HOSTING" in filename.upper():
            contract_type = "Hosting Agreement"
        elif "SUPPLY" in filename.upper():
            contract_type = "Supply Agreement"
        elif "OUTSOURCING" in filename.upper():
            contract_type = "Outsourcing Agreement"
        elif "INTELLECTUAL PROPERTY" in filename.upper():
            contract_type = "Intellectual Property Agreement"
        elif "MANUFACTURING" in filename.upper():
            contract_type = "Manufacturing Agreement"
        elif "MARKETING" in filename.upper():
            contract_type = "Marketing Agreement"
        elif "FRANCHISE" in filename.upper():
            contract_type = "Franchise Agreement"
        elif "TRANSPORTATION" in filename.upper():
            contract_type = "Transportation Agreement"
        elif "AGENCY" in filename.upper():
            contract_type = "Agency Agreement"
        elif "RESELLER" in filename.upper():
            contract_type = "Reseller Agreement"
        elif "PROMOTION" in filename.upper():
            contract_type = "Promotion Agreement"
        elif "CONSULTING" in filename.upper():
            contract_type = "Consulting Agreement"

        # Extract company name (usually the first part before date)
        company = filename.split(
            '_')[0] if '_' in filename else filename.split('-')[0]

        # Try to extract date
        date_match = re.search(r'(\d{2}_\d{2}_\d{4}|\d{8})', filename)
        date = date_match.group(1) if date_match else None

        return cls(
            filename=filename,
            contract_type=contract_type,
            company=company,
            date=date
        )


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    with open(PROMPTS_YAML_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_cuad_data() -> dict:
    """Load CUAD dataset."""
    with open(CUAD_DATA_PATH, 'r') as f:
        return json.load(f)


def find_cuad_contract(filename: str, cuad_data: dict) -> Optional[dict]:
    """Find matching contract in CUAD data based on filename."""
    test_base = filename.replace('.pdf', '').replace('.PDF', '')

    for contract in cuad_data['data']:
        title = contract['title']
        # Try exact match first
        if test_base == title:
            return contract
        # Try partial match
        if test_base in title or title in test_base:
            return contract

    return None


def extract_contract_text(cuad_contract: dict) -> str:
    """Extract the full contract text from CUAD contract data."""
    if not cuad_contract or not cuad_contract.get('paragraphs'):
        return ""

    # CUAD contracts typically have one paragraph with the full context
    for paragraph in cuad_contract['paragraphs']:
        if 'context' in paragraph:
            return paragraph['context']

    return ""


def render_prompt(template: str, **kwargs) -> str:
    """Render prompt template with variables."""
    for key, value in kwargs.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
        template = template.replace(f"{{{{ {key} }}}}", str(value))
    return template


def analyze_contract_directly(prompts: dict, contract_info: ContractInfo, contract_text: str,
                              model_id: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Analyze contract directly with full text in a single call.
    """

    # Get persona description
    persona = "neutral_legal"
    personas_dict = prompts.get("personas", {})
    persona_desc = personas_dict.get(persona, persona)

    # Create a comprehensive prompt for direct analysis
    analysis_prompt = f"""
You are {persona_desc} responsible for analyzing a {contract_info.contract_type} for {contract_info.company}.

Please provide a comprehensive legal analysis of the contract below. Your analysis should follow this structure:

## Summary of Key Clauses
- Identify and summarize the most critical clauses that are needed for a human to review. Focus on max 3/5 clauses. 

## Legal Risks

## Contractual Obligations

## Recommended Actions

CONTRACT TO ANALYZE:
{contract_text}
"""

    try:
        thinking, final_report = chat_with_thinking(
            [{"role": "user", "content": analysis_prompt}],
            model_id=model_id,
            temperature=ANALYSIS_TEMPERATURE
        )

        return {
            "contract_info": {
                "filename": contract_info.filename,
                "contract_type": contract_info.contract_type,
                "company": contract_info.company,
                "date": contract_info.date
            },
            "analysis": {
                "prompt": analysis_prompt,
                "final_report": final_report,
                "thinking": thinking
            },
            "model_used": model_id
        }

    except Exception as e:
        print(f"Error analyzing contract {contract_info.filename}: {e}")
        return {
            "contract_info": {
                "filename": contract_info.filename,
                "contract_type": contract_info.contract_type,
                "company": contract_info.company,
                "date": contract_info.date
            },
            "analysis": {
                "final_report": f"Error analyzing contract: {str(e)}",
                "thinking": "",
                "prompt": ""
            },
            "model_used": model_id
        }


def read_test_filenames() -> List[str]:
    """Read test contract filenames from the file."""
    try:
        with open(TEST_FILENAMES_PATH, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        return filenames
    except FileNotFoundError:
        print(
            f"Error: Could not find test filenames file at {TEST_FILENAMES_PATH}")
        return []


def get_output_filename(filename: str, output_dir: Path, suffix: str = "_report.json") -> Path:
    """Get the output filename for a contract."""
    safe_filename = filename.replace(
        "/", "_").replace("\\", "_").replace(".", "_")
    return output_dir / f"{safe_filename}{suffix}"


def is_contract_processed(filename: str, output_dir: Path) -> bool:
    """Check if a contract has already been processed."""
    report_file = get_output_filename(filename, output_dir, "_report.json")
    text_file = get_output_filename(filename, output_dir, "_final_report.txt")
    return report_file.exists() and text_file.exists()


def filter_processed_contracts(filenames: List[str], output_dir: Path, force: bool = False) -> tuple[List[str], List[str]]:
    """Filter out already processed contracts unless force is True."""
    if force:
        return filenames, []

    to_process = []
    already_processed = []

    for filename in filenames:
        if is_contract_processed(filename, output_dir):
            already_processed.append(filename)
        else:
            to_process.append(filename)

    return to_process, already_processed


def print_processing_summary(total_files: List[str], to_process: List[str], already_processed: List[str]):
    """Print a summary of what will be processed."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total contracts found: {len(total_files)}")
    print(f"Already processed: {len(already_processed)}")
    print(f"To be processed: {len(to_process)}")

    if already_processed:
        print(
            f"\nSkipping {len(already_processed)} already processed contracts:")
        for filename in sorted(already_processed)[:5]:  # Show first 5
            print(f"  - {filename}")
        if len(already_processed) > 5:
            print(f"  ... and {len(already_processed) - 5} more")

    if to_process:
        print(f"\nWill process {len(to_process)} contracts:")
        for filename in sorted(to_process)[:5]:  # Show first 5
            print(f"  - {filename}")
        if len(to_process) > 5:
            print(f"  ... and {len(to_process) - 5} more")

    print("="*60)

    if to_process:
        response = input("\nProceed with processing? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Processing cancelled.")
            exit(0)
    else:
        print("\nNothing to process. All contracts already completed.")
        exit(0)


def process_single_contract(filename: str, prompts: dict, cuad_data: dict, model_id: str) -> Dict[str, Any]:
    """Process a single contract through direct analysis."""

    # Parse contract information
    contract_info = ContractInfo.from_filename(filename)

    print(
        f"Processing: {contract_info.contract_type} for {contract_info.company}")

    # Find contract in CUAD data
    print("  - Finding contract in CUAD dataset...")
    cuad_contract = find_cuad_contract(filename, cuad_data)

    if cuad_contract:
        print(f"  - Found CUAD contract: {cuad_contract['title']}")
        contract_text = extract_contract_text(cuad_contract)
        print(f"  - Contract text length: {len(contract_text)} characters")
    else:
        print(f"  - No matching CUAD contract found for {filename}")
        contract_text = ""

    # Skip if no contract text available
    if not contract_text.strip():
        print(f"  - Skipping {filename} - no contract text available")
        return None

    # Run direct contract analysis
    print("  - Running direct contract analysis...")
    result = analyze_contract_directly(
        prompts, contract_info, contract_text, model_id)

    # Add CUAD information to result
    result['cuad_info'] = {
        'found_match': cuad_contract is not None,
        'cuad_title': cuad_contract['title'] if cuad_contract else None,
        'contract_text_length': len(contract_text)
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end contract processing for test files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model ID to use")
    parser.add_argument("--limit", type=int,
                        help="Limit number of contracts to process")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for reports")
    parser.add_argument("--force", action="store_true",
                        help="Process all contracts, even if already processed")
    args = parser.parse_args()

    # Load prompts
    print("Loading prompts...")
    prompts = load_prompts()

    # Load CUAD data
    print("Loading CUAD dataset...")
    cuad_data = load_cuad_data()
    print(f"Loaded {len(cuad_data['data'])} contracts from CUAD dataset")

    # Read test filenames
    print("Reading test contract filenames...")
    all_test_filenames = read_test_filenames()

    if not all_test_filenames:
        print("No test filenames found. Exiting.")
        return

    # Apply limit if specified
    if args.limit:
        all_test_filenames = all_test_filenames[:args.limit]

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already processed contracts
    test_filenames, already_processed = filter_processed_contracts(
        all_test_filenames, output_dir, args.force)

    # Print summary and get user confirmation
    if not args.force:
        print_processing_summary(
            all_test_filenames, test_filenames, already_processed)
    else:
        print(f"FORCE mode: Processing all {len(test_filenames)} contracts")

    if not test_filenames:
        print("No contracts to process.")
        return

    # Process each contract
    results = []
    skipped = []
    for filename in tqdm.tqdm(test_filenames, desc="Processing contracts"):
        try:
            result = process_single_contract(
                filename, prompts, cuad_data, args.model)

            if result is None:
                skipped.append(filename)
                continue

            results.append(result)

            # Save individual report
            output_file = get_output_filename(
                filename, output_dir, "_report.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Also save just the final report as text
            report_file = get_output_filename(
                filename, output_dir, "_final_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Contract Analysis Report\n")
                f.write(f"{'='*50}\n")
                f.write(f"Contract: {result['contract_info']['filename']}\n")
                f.write(f"Type: {result['contract_info']['contract_type']}\n")
                f.write(f"Company: {result['contract_info']['company']}\n")
                f.write(f"Model: {result['model_used']}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"FINAL REPORT\n")
                f.write(f"{'='*50}\n")
                f.write(result['analysis']['final_report'])

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save summary
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_contracts": len(all_test_filenames),
            "already_processed": len(already_processed) if not args.force else 0,
            "successfully_processed": len(results),
            "skipped_no_text": len(skipped),
            "skipped_files_no_text": skipped,
            "already_processed_files": already_processed if not args.force else [],
            "force_mode": args.force,
            "model_used": args.model,
            "contracts": [r['contract_info'] for r in results]
        }, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete!")
    print(f"Total contracts found: {len(all_test_filenames)}")
    if not args.force and already_processed:
        print(
            f"Already processed (skipped): {len(already_processed)} contracts")
    print(f"Successfully processed: {len(results)} contracts")
    if skipped:
        print(f"Skipped (no contract text): {len(skipped)} contracts")
    print(f"Reports saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
