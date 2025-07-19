#!/usr/bin/env python3
"""
test_cuad_integration.py - Test script to verify CUAD data integration

This script tests that we can successfully load CUAD data and match it with test filenames.
"""

import json
from pathlib import Path


def main():
    """Test CUAD data integration."""

    # Paths
    BASE = Path(__file__).resolve().parent
    TEST_FILENAMES_PATH = BASE.parent.parent.parent / \
        "data" / "split_filenames" / "test_filenames.txt"
    CUAD_DATA_PATH = BASE.parent.parent.parent / "data" / "CUAD_v1" / "CUAD_v1.json"

    print("Testing CUAD Data Integration")
    print("=" * 50)

    # Load CUAD data
    print("1. Loading CUAD dataset...")
    try:
        with open(CUAD_DATA_PATH, 'r') as f:
            cuad_data = json.load(f)
        print(
            f"   ✓ Loaded {len(cuad_data['data'])} contracts from CUAD dataset")
    except Exception as e:
        print(f"   ✗ Error loading CUAD data: {e}")
        return False

    # Load test filenames
    print("2. Loading test filenames...")
    try:
        with open(TEST_FILENAMES_PATH, 'r') as f:
            test_filenames = [line.strip() for line in f if line.strip()]
        print(f"   ✓ Loaded {len(test_filenames)} test filenames")
    except Exception as e:
        print(f"   ✗ Error loading test filenames: {e}")
        return False

    # Test matching function
    print("3. Testing contract matching...")

    def find_cuad_contract(filename: str, cuad_data: dict):
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

    # Test first 10 filenames
    matches_found = 0
    for i, filename in enumerate(test_filenames[:10]):
        cuad_contract = find_cuad_contract(filename, cuad_data)
        if cuad_contract:
            matches_found += 1
            print(f"   ✓ {filename}")
            print(f"     -> {cuad_contract['title']}")

            # Test contract text extraction
            if cuad_contract.get('paragraphs'):
                for paragraph in cuad_contract['paragraphs']:
                    if 'context' in paragraph:
                        contract_text = paragraph['context']
                        print(
                            f"     -> Contract text: {len(contract_text)} characters")
                        print(f"     -> Preview: {contract_text[:100]}...")
                        break
        else:
            print(f"   ✗ {filename} (no match found)")

    print(f"\n4. Summary:")
    print(f"   Tested: {min(10, len(test_filenames))} filenames")
    print(f"   Matches found: {matches_found}")
    print(
        f"   Success rate: {matches_found/min(10, len(test_filenames))*100:.1f}%")

    if matches_found > 0:
        print("\n   ✓ CUAD integration is working correctly!")
        return True
    else:
        print("\n   ✗ No matches found - check filename matching logic")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
