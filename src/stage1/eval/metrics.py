import json
import re
from difflib import SequenceMatcher

# BASE toggle: when True, strips ```json\n at beginning and \n``` at end if present
BASE = False


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
    Conservative JSON parsing that avoids over-extraction.

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

    return []


def evaluate_outputs(file_path: str, similarity_threshold: float = 0.6) -> dict:
    """
    Evaluates model outputs against expected outputs from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing input, expected, and generated outputs.
        similarity_threshold (float): The threshold for considering two clauses as a match.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
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

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Add counters for tracking
    total_instances = len(data)
    successfully_parsed_expected = 0
    successfully_parsed_generated = 0
    instances_with_valid_data = 0
    fallback_parsing_used = 0

    # Add detailed clause counting
    total_expected_clauses = 0
    total_generated_clauses = 0
    instances_with_expected_clauses = 0
    instances_with_generated_clauses = 0
    clause_count_distribution = {
        'expected': {},
        'generated': {}
    }

    print(f"Total instances in JSON file: {total_instances}")

    if not data:  # No items to evaluate
        return {
            "total_instances": 0,
            "successfully_parsed_expected": 0,
            "successfully_parsed_generated": 0,
            "instances_with_valid_data": 0,
            "fallback_parsing_used": 0,
            "total_expected_clauses": 0,
            "total_generated_clauses": 0,
            "instances_with_expected_clauses": 0,
            "instances_with_generated_clauses": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "accuracy": 0,
            "error": "No data to evaluate"
        }

    for idx, item in enumerate(data):
        expected_outputs = []
        generated_outputs = []
        expected_parsed_successfully = False
        generated_parsed_successfully = False

        # Parse expected_outputs using conservative parsing
        expected_output_val = item.get("expected_output")
        if expected_output_val:
            expected_outputs = conservative_json_parse(expected_output_val)
            expected_parsed_successfully = len(
                expected_outputs) > 0 or expected_output_val.strip() == '[]'

        # Parse generated_outputs using conservative parsing
        generated_output_val = item.get("generated_output")
        if generated_output_val:
            generated_outputs = conservative_json_parse(generated_output_val)
            generated_parsed_successfully = len(
                generated_outputs) > 0 or generated_output_val.strip() == '[]'

        # Update clause counts
        expected_count = len(expected_outputs)
        generated_count = len(generated_outputs)

        total_expected_clauses += expected_count
        total_generated_clauses += generated_count

        if expected_count > 0:
            instances_with_expected_clauses += 1
            clause_count_distribution['expected'][expected_count] = clause_count_distribution['expected'].get(
                expected_count, 0) + 1

        if generated_count > 0:
            instances_with_generated_clauses += 1
            clause_count_distribution['generated'][generated_count] = clause_count_distribution['generated'].get(
                generated_count, 0) + 1

        # Update counters
        if expected_parsed_successfully:
            successfully_parsed_expected += 1
        if generated_parsed_successfully:
            successfully_parsed_generated += 1
        if expected_outputs or generated_outputs:  # At least one has valid data
            instances_with_valid_data += 1

        current_tp = 0

        # Keep track of which expected and generated clauses have been matched
        # to ensure one-to-one matching where possible.
        matched_expected_indices = set()
        matched_generated_indices = set()

        # Try to match generated outputs to expected outputs
        for i, gen_clause in enumerate(generated_outputs):
            best_match_score = 0
            best_match_expected_idx = -1
            for j, exp_clause in enumerate(expected_outputs):
                if j in matched_expected_indices:  # If this expected clause is already matched, skip
                    continue

                sim_score = similarity(gen_clause, exp_clause)
                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match_expected_idx = j

            if best_match_score > similarity_threshold and best_match_expected_idx != -1:
                current_tp += 1
                matched_expected_indices.add(best_match_expected_idx)
                matched_generated_indices.add(i)

        # False positives are generated clauses that were not matched to any expected clause
        current_fp = len(generated_outputs) - len(matched_generated_indices)

        # False negatives are expected clauses that were not matched by any generated clause
        current_fn = len(expected_outputs) - len(matched_expected_indices)

        total_tp += current_tp
        total_fp += current_fp
        total_fn += current_fn

    precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    # Calculate Accuracy (Jaccard Index)
    # Union = TP + FP + FN
    # Intersection = TP
    # Jaccard = Intersection / Union
    union_of_sets = total_tp + total_fp + total_fn
    if union_of_sets == 0:
        # This case implies all items had empty expected and empty generated lists,
        # or there were items but they all resulted in TP=0, FP=0, FN=0.
        # If len(data) > 0 and union_of_sets == 0, it means perfect prediction of nothingness for all.
        accuracy = 1.0 if len(data) > 0 else 0.0
    else:
        accuracy = total_tp / union_of_sets

    print(
        f"Successfully parsed expected outputs: {successfully_parsed_expected}/{total_instances}")
    print(
        f"Successfully parsed generated outputs: {successfully_parsed_generated}/{total_instances}")
    print(
        f"Instances with valid data (expected or generated): {instances_with_valid_data}/{total_instances}")

    # Print detailed clause statistics
    print(f"\n=== CLAUSE STATISTICS ===")
    print(f"Total expected clauses: {total_expected_clauses}")
    print(f"Total generated clauses: {total_generated_clauses}")
    print(
        f"Instances with expected clauses: {instances_with_expected_clauses}/{total_instances}")
    print(
        f"Instances with generated clauses: {instances_with_generated_clauses}/{total_instances}")

    if instances_with_expected_clauses > 0:
        avg_expected = total_expected_clauses / instances_with_expected_clauses
        print(
            f"Average expected clauses per instance (with expected): {avg_expected:.2f}")

    if instances_with_generated_clauses > 0:
        avg_generated = total_generated_clauses / instances_with_generated_clauses
        print(
            f"Average generated clauses per instance (with generated): {avg_generated:.2f}")

    if instances_with_valid_data > 0:
        avg_expected_valid = total_expected_clauses / instances_with_valid_data
        avg_generated_valid = total_generated_clauses / instances_with_valid_data
        print(
            f"Average expected clauses per instance (with valid data): {avg_expected_valid:.2f}")
        print(
            f"Average generated clauses per instance (with valid data): {avg_generated_valid:.2f}")

    # Show distribution of clause counts (top 10 only)
    print(f"\n=== EXPECTED CLAUSE COUNT DISTRIBUTION ===")
    expected_items = sorted(clause_count_distribution['expected'].items())
    for count, instances in expected_items[:10]:
        print(f"  {count} clauses: {instances} instances")

    print(f"\n=== GENERATED CLAUSE COUNT DISTRIBUTION ===")
    generated_items = sorted(clause_count_distribution['generated'].items())
    for count, instances in generated_items[:10]:
        print(f"  {count} clauses: {instances} instances")

    return {
        "total_instances": total_instances,
        "successfully_parsed_expected": successfully_parsed_expected,
        "successfully_parsed_generated": successfully_parsed_generated,
        "fallback_parsing_used": fallback_parsing_used,
        "instances_with_valid_data": instances_with_valid_data,
        "total_expected_clauses": total_expected_clauses,
        "total_generated_clauses": total_generated_clauses,
        "instances_with_expected_clauses": instances_with_expected_clauses,
        "instances_with_generated_clauses": instances_with_generated_clauses,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Example usage:
    # Make sure the JSON file path is correct and accessible.
    # You might need to adjust the path depending on where you run the script from.
    # For example, if metrics.py is in src/stage1/eval/ and the json is in src/stage1/out/
    # the relative path from metrics.py would be ../out/Competition_&_Exclusivity_test_outputs.json

    # Assuming the script is run from the workspace root (/Users/thomas/Documents/GIT/Thesis)
    # or that the path is relative to where the script is run.
    # For clarity, using a path relative to the script's assumed location if it were in the project root.
    # If you run this script directly, ensure the path to the JSON file is correct.

    # Path relative to the workspace root
    json_file_path = "../out/Legal_Protections_Liability_test_outputs_8B.json"

    results = evaluate_outputs(json_file_path)

    print(f"\nEvaluation Results for {json_file_path}:")
    print(f"  Total Instances: {results.get('total_instances', 'N/A')}")
    print(
        f"  Successfully Parsed Expected: {results.get('successfully_parsed_expected', 'N/A')}")
    print(
        f"  Successfully Parsed Generated: {results.get('successfully_parsed_generated', 'N/A')}")
    print(
        f"  Fallback Parsing Used: {results.get('fallback_parsing_used', 'N/A')}")
    print(
        f"  Instances with Valid Data: {results.get('instances_with_valid_data', 'N/A')}")
    print(
        f"  Total Expected Clauses: {results.get('total_expected_clauses', 'N/A')}")
    print(
        f"  Total Generated Clauses: {results.get('total_generated_clauses', 'N/A')}")
    print(f"  True Positives: {results.get('true_positives', 'N/A')}")
    print(f"  False Positives: {results.get('false_positives', 'N/A')}")
    print(f"  False Negatives: {results.get('false_negatives', 'N/A')}")
    print(f"  Precision: {results.get('precision', 'N/A'):.4f}")
    print(f"  Recall: {results.get('recall', 'N/A'):.4f}")
    print(f"  F1 Score: {results.get('f1_score', 'N/A'):.4f}")
    print(f"  Accuracy (Jaccard Index): {results.get('accuracy', 'N/A'):.4f}")
    if "error" in results:
        print(f"  Error: {results['error']}")
