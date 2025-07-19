import json
from difflib import SequenceMatcher
from collections import defaultdict
import argparse
import re

# BASE toggle: when True, strips ```json\n at beginning and \n``` at end if present
BASE = True

ap = argparse.ArgumentParser()


def similarity(a: str, b: str) -> float:
    """
    Calculates the Jaccard similarity coefficient between two strings.

    Args:
        a (str): First string
        b (str): Second string

    Returns:
        float: Jaccard similarity coefficient (0 to 1)
    """
    # Tokenize and normalize the strings
    def tokenize(text):
        # Convert to lowercase and split by non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    set_a = tokenize(a)
    set_b = tokenize(b)

    if not set_a and not set_b:
        return 1.0  # Both empty, perfect match

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union if union > 0 else 0.0


def conservative_json_parse(json_str: str) -> list:
    """
    Conservative JSON parsing that avoids over-extraction and handles malformed JSON.

    Args:
        json_str (str): The JSON string to parse

    Returns:
        list: Parsed list or empty list if parsing fails
    """
    if not json_str or not json_str.strip():
        return []

    # Strip markdown JSON formatting if BASE is True
    if BASE:
        # Strip ```json\n at the beginning if present
        if json_str.strip().startswith('```json\n'):
            json_str = json_str.strip()[8:]  # Remove ```json\n
        elif json_str.strip().startswith('```json'):
            json_str = json_str.strip()[7:]  # Remove ```json

        # Strip \n``` at the end if present
        if json_str.rstrip().endswith('\n```'):
            json_str = json_str.rstrip()[:-4]  # Remove \n```
        elif json_str.rstrip().endswith('```'):
            json_str = json_str.rstrip()[:-3]  # Remove ```

    # First attempt: Try direct parsing
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, str):
            # Double encoded - try parsing again
            try:
                inner_parsed = json.loads(parsed)
                if isinstance(inner_parsed, list):
                    return [clause for clause in inner_parsed if isinstance(clause, str)]
                elif isinstance(inner_parsed, dict) and BASE:
                    # Handle BASE format where it's an object with key containing array
                    all_clauses = []
                    for key, value in inner_parsed.items():
                        if isinstance(value, list):
                            all_clauses.extend(
                                [clause for clause in value if isinstance(clause, str)])
                    return all_clauses
            except json.JSONDecodeError:
                pass
        elif isinstance(parsed, list):
            return [clause for clause in parsed if isinstance(clause, str)]
        elif isinstance(parsed, dict) and BASE:
            # Handle BASE format where it's an object with key containing array
            all_clauses = []
            for key, value in parsed.items():
                if isinstance(value, list):
                    all_clauses.extend(
                        [clause for clause in value if isinstance(clause, str)])
            return all_clauses
    except json.JSONDecodeError:
        pass

    # Second attempt: Try to fix common JSON issues
    try:
        # Fix unterminated strings by finding the last complete string in the array
        if json_str.strip().startswith('[') and not json_str.strip().endswith(']'):
            # Find the last complete quote
            last_quote = json_str.rfind('"')
            if last_quote > 0:
                # Try to close the array after the last quote
                fixed_str = json_str[:last_quote + 1] + ']'
                parsed = json.loads(fixed_str)
                if isinstance(parsed, list):
                    return [clause for clause in parsed if isinstance(clause, str)]

        # Try to fix missing delimiters by finding broken strings
        if '"' in json_str and '[' in json_str:
            # Extract strings using regex as a fallback
            import re
            # Find all quoted strings
            string_pattern = r'"([^"\\]|\\.)*"'
            matches = re.findall(string_pattern, json_str)
            if matches:
                # Remove escape characters and return as list
                return [match.replace('\\"', '"').replace('\\\\', '\\') for match in matches if isinstance(match, str) and len(match.strip()) > 0]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Third attempt: Extract quoted strings manually as last resort
    try:
        if '"' in json_str:
            import re
            # More aggressive string extraction
            quotes = []
            in_string = False
            current_string = ""
            escape_next = False

            for char in json_str:
                if escape_next:
                    if in_string:
                        current_string += char
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    if in_string:
                        current_string += char
                    continue

                if char == '"':
                    if in_string:
                        # End of string
                        if current_string.strip():  # Only add non-empty strings
                            quotes.append(current_string)
                        current_string = ""
                        in_string = False
                    else:
                        # Start of string
                        in_string = True
                elif in_string:
                    current_string += char

            # Filter out very short or empty strings
            valid_quotes = [q for q in quotes if len(q.strip()) > 10]
            if valid_quotes:
                return valid_quotes
    except Exception:
        pass

    return []


def evaluate_contract_sets(file_path: str, similarity_threshold: float = 0.75) -> dict:
    """
    Evaluates model outputs against expected outputs by grouping clauses by contract ID
    and treating each contract as a set comparison.

    Args:
        file_path (str): Path to the JSON file containing input, expected, and generated outputs.
        similarity_threshold (float): The threshold for considering two clauses as a match.

    Returns:
        dict: A dictionary containing contract-level precision, recall, and F1 score.
    """
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.jsonl'):
                # Handle JSONL format (one JSON object per line)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                # Handle standard JSON format
                data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"precision": 0, "recall": 0, "f1_score": 0, "error": "File not found"}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return {"precision": 0, "recall": 0, "f1_score": 0, "error": "JSON decode error"}

    # Group clauses by contract ID
    contract_clauses = defaultdict(
        lambda: {"expected": set(), "generated": set()})

    total_instances = len(data)
    successfully_parsed_expected = 0
    successfully_parsed_generated = 0

    # Track parsing failures for debugging
    expected_parse_failures = []
    generated_parse_failures = []

    print(f"Total instances in JSON file: {total_instances}")

    # Group all clauses by contract ID
    for idx, item in enumerate(data):
        contract_id = item.get("id", f"unknown_{idx}")

        # Parse expected outputs
        expected_output_val = item.get("expected_output")
        if expected_output_val:
            expected_clauses = conservative_json_parse(expected_output_val)
            if expected_clauses or expected_output_val.strip() == '[]':
                successfully_parsed_expected += 1
                # Add to set (automatically deduplicates)
                for clause in expected_clauses:
                    if clause.strip():  # Only add non-empty clauses
                        contract_clauses[contract_id]["expected"].add(
                            clause.strip())
            else:
                expected_parse_failures.append({
                    'index': idx,
                    'contract_id': contract_id,
                    'content': expected_output_val[:200] + '...' if len(expected_output_val) > 200 else expected_output_val
                })

        # Parse generated outputs
        generated_output_val = item.get("generated_output")
        if generated_output_val:
            generated_clauses = conservative_json_parse(generated_output_val)
            if generated_clauses or generated_output_val.strip() == '[]':
                successfully_parsed_generated += 1
                # Add to set (automatically deduplicates)
                for clause in generated_clauses:
                    if clause.strip():  # Only add non-empty clauses
                        contract_clauses[contract_id]["generated"].add(
                            clause.strip())
            else:
                generated_parse_failures.append({
                    'index': idx,
                    'contract_id': contract_id,
                    'content': generated_output_val[:200] + '...' if len(generated_output_val) > 200 else generated_output_val
                })

    print(
        f"Successfully parsed expected outputs: {successfully_parsed_expected}/{total_instances}")
    print(
        f"Successfully parsed generated outputs: {successfully_parsed_generated}/{total_instances}")

    # Report parsing failures if any
    if expected_parse_failures:
        print(
            f"\nExpected output parsing failures: {len(expected_parse_failures)}")
        for failure in expected_parse_failures[:3]:  # Show first 3
            print(
                f"  - Item {failure['index']} ({failure['contract_id']}): {repr(failure['content'])}")

    if generated_parse_failures:
        print(
            f"\nGenerated output parsing failures: {len(generated_parse_failures)}")
        for failure in generated_parse_failures[:3]:  # Show first 3
            print(
                f"  - Item {failure['index']} ({failure['contract_id']}): {repr(failure['content'])}")

    print(f"Unique contracts found: {len(contract_clauses)}")

    # Calculate metrics for each contract
    contract_results = []
    total_contract_tp = 0
    total_contract_fp = 0
    total_contract_fn = 0

    contracts_with_data = 0
    total_expected_clauses = 0
    total_generated_clauses = 0

    for contract_id, clauses in contract_clauses.items():
        expected_set = clauses["expected"]
        generated_set = clauses["generated"]

        if not expected_set and not generated_set:
            continue  # Skip contracts with no data

        contracts_with_data += 1
        total_expected_clauses += len(expected_set)
        total_generated_clauses += len(generated_set)

        # Find matches using similarity threshold
        matched_expected = set()
        matched_generated = set()
        contract_tp = 0

        # For each generated clause, find best match in expected
        for gen_clause in generated_set:
            best_match_score = 0
            best_match_expected = None

            for exp_clause in expected_set:
                if exp_clause in matched_expected:
                    continue  # Already matched

                sim_score = similarity(gen_clause, exp_clause)
                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match_expected = exp_clause

            if best_match_score > similarity_threshold and best_match_expected:
                contract_tp += 1
                matched_expected.add(best_match_expected)
                matched_generated.add(gen_clause)

        contract_fp = len(generated_set) - len(matched_generated)
        contract_fn = len(expected_set) - len(matched_expected)

        # Contract-level metrics
        contract_precision = contract_tp / \
            len(generated_set) if len(generated_set) > 0 else 0
        contract_recall = contract_tp / \
            len(expected_set) if len(expected_set) > 0 else 0
        contract_f1 = 2 * (contract_precision * contract_recall) / (contract_precision +
                                                                    contract_recall) if (contract_precision + contract_recall) > 0 else 0

        # Contract-level IoU
        contract_union = contract_tp + contract_fp + contract_fn
        contract_iou = contract_tp / contract_union if contract_union > 0 else 0

        contract_results.append({
            "contract_id": contract_id,
            "expected_clauses": len(expected_set),
            "generated_clauses": len(generated_set),
            "true_positives": contract_tp,
            "false_positives": contract_fp,
            "false_negatives": contract_fn,
            "precision": contract_precision,
            "recall": contract_recall,
            "f1_score": contract_f1,
            "iou": contract_iou
        })

        total_contract_tp += contract_tp
        total_contract_fp += contract_fp
        total_contract_fn += contract_fn

    # Overall metrics
    overall_precision = total_contract_tp / \
        (total_contract_tp + total_contract_fp) if (total_contract_tp +
                                                    total_contract_fp) > 0 else 0
    overall_recall = total_contract_tp / \
        (total_contract_tp + total_contract_fn) if (total_contract_tp +
                                                    total_contract_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision +
                                                             overall_recall) if (overall_precision + overall_recall) > 0 else 0

    overall_union = total_contract_tp + total_contract_fp + total_contract_fn
    overall_iou = total_contract_tp / overall_union if overall_union > 0 else 0

    # Calculate average contract-level metrics
    if contract_results:
        avg_contract_precision = sum(
            r["precision"] for r in contract_results) / len(contract_results)
        avg_contract_recall = sum(r["recall"]
                                  for r in contract_results) / len(contract_results)
        avg_contract_f1 = sum(r["f1_score"]
                              for r in contract_results) / len(contract_results)
        avg_contract_iou = sum(r["iou"]
                               for r in contract_results) / len(contract_results)

        # Count contracts meeting IoU threshold
        contracts_above_05_iou = sum(
            1 for r in contract_results if r["iou"] >= 0.5)
        contracts_above_03_iou = sum(
            1 for r in contract_results if r["iou"] >= 0.3)
    else:
        avg_contract_precision = avg_contract_recall = avg_contract_f1 = avg_contract_iou = 0
        contracts_above_05_iou = contracts_above_03_iou = 0

    print(f"\n=== CONTRACT-LEVEL STATISTICS ===")
    print(f"Contracts with data: {contracts_with_data}")
    print(
        f"Total expected clauses across all contracts: {total_expected_clauses}")
    print(
        f"Total generated clauses across all contracts: {total_generated_clauses}")
    print(
        f"Average expected clauses per contract: {total_expected_clauses/contracts_with_data if contracts_with_data > 0 else 0:.2f}")
    print(
        f"Average generated clauses per contract: {total_generated_clauses/contracts_with_data if contracts_with_data > 0 else 0:.2f}")

    print(f"\n=== OVERALL AGGREGATED METRICS ===")
    print(f"Total True Positives: {total_contract_tp}")
    print(f"Total False Positives: {total_contract_fp}")
    print(f"Total False Negatives: {total_contract_fn}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall IoU (Jaccard Index): {overall_iou:.4f}")

    print(f"\n=== AVERAGE CONTRACT-LEVEL METRICS ===")
    print(f"Average Contract Precision: {avg_contract_precision:.4f}")
    print(f"Average Contract Recall: {avg_contract_recall:.4f}")
    print(f"Average Contract F1: {avg_contract_f1:.4f}")
    print(f"Average Contract IoU: {avg_contract_iou:.4f}")

    print(f"\n=== CONTRACT PERFORMANCE DISTRIBUTION ===")
    print(
        f"Contracts with IoU ≥ 0.5: {contracts_above_05_iou}/{contracts_with_data} ({contracts_above_05_iou/contracts_with_data*100 if contracts_with_data > 0 else 0:.1f}%)")
    print(
        f"Contracts with IoU ≥ 0.3: {contracts_above_03_iou}/{contracts_with_data} ({contracts_above_03_iou/contracts_with_data*100 if contracts_with_data > 0 else 0:.1f}%)")

    # Show top and bottom performing contracts
    if contract_results:
        sorted_contracts = sorted(
            contract_results, key=lambda x: x["iou"], reverse=True)

        print(f"\n=== TOP 5 PERFORMING CONTRACTS ===")
        for i, result in enumerate(sorted_contracts[:5]):
            print(
                f"{i+1}. Contract {result['contract_id']}: IoU={result['iou']:.3f}, F1={result['f1_score']:.3f} (E:{result['expected_clauses']}, G:{result['generated_clauses']}, TP:{result['true_positives']})")

        print(f"\n=== BOTTOM 5 PERFORMING CONTRACTS ===")
        for i, result in enumerate(sorted_contracts[-5:]):
            print(
                f"{len(sorted_contracts)-4+i}. Contract {result['contract_id']}: IoU={result['iou']:.3f}, F1={result['f1_score']:.3f} (E:{result['expected_clauses']}, G:{result['generated_clauses']}, TP:{result['true_positives']})")

    return {
        "total_instances": total_instances,
        "unique_contracts": len(contract_clauses),
        "contracts_with_data": contracts_with_data,
        "successfully_parsed_expected": successfully_parsed_expected,
        "successfully_parsed_generated": successfully_parsed_generated,
        "expected_parse_failures": len(expected_parse_failures),
        "generated_parse_failures": len(generated_parse_failures),
        "total_expected_clauses": total_expected_clauses,
        "total_generated_clauses": total_generated_clauses,
        "overall_true_positives": total_contract_tp,
        "overall_false_positives": total_contract_fp,
        "overall_false_negatives": total_contract_fn,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1_score": overall_f1,
        "overall_iou": overall_iou,
        "avg_contract_precision": avg_contract_precision,
        "avg_contract_recall": avg_contract_recall,
        "avg_contract_f1": avg_contract_f1,
        "avg_contract_iou": avg_contract_iou,
        "contracts_above_05_iou": contracts_above_05_iou,
        "contracts_above_03_iou": contracts_above_03_iou,
        "contract_results": contract_results
    }


if __name__ == "__main__":
    # Path relative to the workspace root
    ap.add_argument("--json_file_path", type=str,
                    default="../out/Competition_&_Exclusivity_4b_outputs.json")
    ap.add_argument("--base", action="store_true", default=True,
                    help="Enable BASE mode for handling base model outputs (default: True)")
    args = ap.parse_args()
    json_file_path = args.json_file_path

    # Override the global BASE variable with command line argument
    BASE = args.base

    results = evaluate_contract_sets(json_file_path)

    print(f"\n" + "="*60)
    print(f"CONTRACT-LEVEL EVALUATION SUMMARY")
    print(f"="*60)
    print(f"File: {json_file_path}")
    print(f"Total Instances: {results.get('total_instances', 'N/A')}")
    print(f"Unique Contracts: {results.get('unique_contracts', 'N/A')}")
    print(f"Contracts with Data: {results.get('contracts_with_data', 'N/A')}")
    print(f"Overall IoU: {results.get('overall_iou', 'N/A'):.4f}")
    print(
        f"Average Contract IoU: {results.get('avg_contract_iou', 'N/A'):.4f}")
    print(
        f"Contracts achieving IoU ≥ 0.5: {results.get('contracts_above_05_iou', 'N/A')}")
    if "error" in results:
        print(f"Error: {results['error']}")
