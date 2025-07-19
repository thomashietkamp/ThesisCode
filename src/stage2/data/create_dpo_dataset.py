#!/usr/bin/env python3
"""
Script to convert votes data and stage2 JSON outputs into DPO training format.
This creates preference pairs for the coordinator → reasoner → validator pipeline.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path


def load_votes_data(csv_path: str) -> pd.DataFrame:
    """Load and filter votes data for ai_gemini user session."""
    df = pd.read_csv(csv_path)
    # Filter for ai_gemini user session
    ai_gemini_df = df[df['user_session'].str.contains('ai_gemini', na=False)]
    return ai_gemini_df


def load_json_data(json_dir: str) -> Dict[str, Dict]:
    """Load all JSON files from the stage2_out directory."""
    json_data = {}
    json_dir = Path(json_dir)

    for json_file in json_dir.glob("*.json"):
        contract_id = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            json_data[contract_id] = data

    return json_data


def extract_reasoning_chain(variant_data: Dict) -> Tuple[str, str, str, str, str, str]:
    """
    Extract the reasoning chain from a variant.
    Returns: (coord_think, coord_out, reason_think, reason_out, valid_think, valid_out)
    """
    try:
        coord_think = variant_data.get('coordinator_thinking', '')
        coord_out = variant_data.get('coordinator_draft', '')
        reason_think = variant_data.get('reasoner_thinking', '')
        reason_out = variant_data.get('reasoner_analysis', '')
        valid_think = variant_data.get('final_thinking', '')
        valid_out = variant_data.get('final_report', '')

        return coord_think, coord_out, reason_think, reason_out, valid_think, valid_out
    except Exception as e:
        print(f"Error extracting reasoning chain: {e}")
        return '', '', '', '', '', ''


def create_prompt_from_json(variant_data: Dict) -> str:
    """Extract the coordinator prompt as the starting prompt."""
    return variant_data.get('coordinator_prompt', '')


def process_vote_pair(vote_row: pd.Series, json_data: Dict[str, Dict]) -> Optional[Dict]:
    """
    Process a single vote to create a DPO training example.

    Args:
        vote_row: Row from votes DataFrame with winner/loser info
        json_data: All JSON data indexed by contract_id

    Returns:
        Dictionary with DPO training format or None if data missing
    """
    contract_id = vote_row['contract_id']
    option1_key = str(vote_row['option1_key'])
    option2_key = str(vote_row['option2_key'])
    winner_key = str(vote_row['winner_key'])

    # Skip unclear votes
    if winner_key == 'UNCLEAR':
        return None

    # Get JSON data for this contract
    if contract_id not in json_data:
        print(f"Warning: No JSON data found for contract {contract_id}")
        return None

    contract_data = json_data[contract_id]

    # Check if both variants exist
    if option1_key not in contract_data or option2_key not in contract_data:
        print(
            f"Warning: Missing variant data for {contract_id} - {option1_key} or {option2_key}")
        return None

    # Extract reasoning chains for both options
    option1_chain = extract_reasoning_chain(contract_data[option1_key])
    option2_chain = extract_reasoning_chain(contract_data[option2_key])

    # Get the starting prompt (should be same for both variants)
    prompt = create_prompt_from_json(contract_data[option1_key])

    # Determine winner and loser chains
    if winner_key == option1_key:
        winner_chain = option1_chain
        loser_chain = option2_chain
    elif winner_key == option2_key:
        winner_chain = option2_chain
        loser_chain = option1_chain
    else:
        print(
            f"Warning: Winner key {winner_key} doesn't match either option for {contract_id}")
        return None

    # Create DPO training example
    dpo_example = {
        'prompt': prompt,
        'contract_id': contract_id,
        'pair_identifier': vote_row['pair_identifier'],
        'winner_key': winner_key,
        'voted_at': vote_row['voted_at'],

        # Winner trajectory (preferred)
        'coord_think_plus': winner_chain[0],
        'coord_out_plus': winner_chain[1],
        'reason_think_plus': winner_chain[2],
        'reason_out_plus': winner_chain[3],
        'valid_think_plus': winner_chain[4],
        'valid_out_plus': winner_chain[5],

        # Loser trajectory (rejected)
        'coord_think_minus': loser_chain[0],
        'coord_out_minus': loser_chain[1],
        'reason_think_minus': loser_chain[2],
        'reason_out_minus': loser_chain[3],
        'valid_think_minus': loser_chain[4],
        'valid_out_minus': loser_chain[5],
    }

    return dpo_example


def main():
    parser = argparse.ArgumentParser(
        description='Convert votes and JSON data to DPO format')
    parser.add_argument('--votes_csv', type=str,
                        default='src/stage2/data/votes_export_20250611_110658.csv',
                        help='Path to votes CSV file')
    parser.add_argument('--json_dir', type=str,
                        default='src/stage2/data/stage2_out',
                        help='Directory containing JSON files')
    parser.add_argument('--output_path', type=str,
                        default='src/stage2/data/dpo_training_data.jsonl',
                        help='Output path for DPO training data')
    parser.add_argument('--user_session', type=str,
                        default='ai_gemini',
                        help='User session to filter for (default: ai_gemini)')

    args = parser.parse_args()

    print("Loading votes data...")
    votes_df = load_votes_data(args.votes_csv)
    print(f"Found {len(votes_df)} votes from {args.user_session} user session")

    print("Loading JSON data...")
    json_data = load_json_data(args.json_dir)
    print(f"Loaded {len(json_data)} JSON files")

    print("Processing vote pairs...")
    dpo_examples = []
    skipped_count = 0

    for idx, vote_row in votes_df.iterrows():
        dpo_example = process_vote_pair(vote_row, json_data)
        if dpo_example:
            dpo_examples.append(dpo_example)
        else:
            skipped_count += 1

    print(f"Created {len(dpo_examples)} DPO training examples")
    print(
        f"Skipped {skipped_count} votes due to missing data or unclear outcomes")

    # Save to JSONL format
    print(f"Saving to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for example in dpo_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(
        f"Successfully saved {len(dpo_examples)} DPO training examples to {args.output_path}")

    # Print some statistics
    print("\n--- Statistics ---")
    contracts_with_data = set(ex['contract_id'] for ex in dpo_examples)
    print(f"Unique contracts with training data: {len(contracts_with_data)}")

    # Show a sample example structure
    if dpo_examples:
        print("\n--- Sample DPO Example Structure ---")
        sample = dpo_examples[0]
        for key in sample.keys():
            if key.startswith(('coord_', 'reason_', 'valid_')):
                value_preview = sample[key][:100] + \
                    "..." if len(sample[key]) > 100 else sample[key]
                print(f"{key}: {value_preview}")
            else:
                print(f"{key}: {sample[key]}")


if __name__ == "__main__":
    main()
