import json
import os
import re
from collections import defaultdict


def conservative_json_parse(json_str: str) -> list:
    """
    Conservative JSON parsing that avoids over-extraction and handles malformed JSON.
    Copied from contract_level_metrics.py for consistency.

    Args:
        json_str (str): The JSON string to parse

    Returns:
        list: Parsed list or empty list if parsing fails
    """
    if not json_str or not json_str.strip():
        return []

    # First attempt: Try direct parsing
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, str):
            # Double encoded - try parsing again
            try:
                inner_parsed = json.loads(parsed)
                if isinstance(inner_parsed, list):
                    return [clause for clause in inner_parsed if isinstance(clause, str)]
            except json.JSONDecodeError:
                pass
        elif isinstance(parsed, list):
            return [clause for clause in parsed if isinstance(clause, str)]
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


def concatenate_clauses(input_dir, output_file):
    """
    Concatenates all unique clauses from multiple JSON files in a directory,
    grouping them by contract ID.

    Args:
        input_dir (str): The directory containing the JSON files.
        output_file (str): The path to the output JSON file.
    """
    contract_clauses = defaultdict(set)
    total_entries_processed = 0
    files_processed = 0

    for filename in os.listdir(input_dir):
        if filename.endswith("8b.json"):
            filepath = os.path.join(input_dir, filename)
            files_processed += 1
            print(f"Processing file {files_processed}: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)  # Load entire JSON array

                    if not isinstance(data_list, list):
                        print(
                            f"File {filepath} does not contain a JSON array. Skipping.")
                        continue

                    print(f"  Found {len(data_list)} entries in {filename}")

                    for i, data in enumerate(data_list):
                        total_entries_processed += 1
                        try:
                            contract_id = data.get("id")
                            # Changed from "prediction"
                            prediction_str = data.get("generated_output")

                            # Debug: Show first few entries
                            if i < 3:
                                print(
                                    f"    Entry {i}: contract_id='{contract_id}', generated_output='{prediction_str[:100] if prediction_str else None}...'")

                            if contract_id and isinstance(prediction_str, str):
                                # Use conservative JSON parser
                                parsed_clauses = conservative_json_parse(
                                    prediction_str)
                                if parsed_clauses:
                                    print(
                                        f"    Found {len(parsed_clauses)} clauses for contract {contract_id}")
                                    for clause in parsed_clauses:
                                        if clause.strip():  # Only add non-empty clauses
                                            contract_clauses[contract_id].add(
                                                clause.strip())
                                elif prediction_str.strip() == "[]":
                                    # Empty prediction
                                    if i < 3:
                                        print(
                                            f"    Empty generated_output for contract {contract_id}")
                                else:
                                    # If conservative parser returns empty but string is not empty
                                    # Add as single clause only if it's reasonably short
                                    if len(prediction_str) < 1000 and prediction_str.strip():
                                        print(
                                            f"    Adding unparseable generated_output as single clause for contract {contract_id}: {prediction_str[:100]}...")
                                        contract_clauses[contract_id].add(
                                            prediction_str.strip())
                                    else:
                                        print(
                                            f"    Generated_output for {contract_id} could not be parsed and is too long to be a single clause ({len(prediction_str)} chars). Skipping.")
                            elif contract_id and isinstance(data.get("generated_output"), list):
                                # Fallback if 'generated_output' is already a list
                                clauses_list = data.get("generated_output", [])
                                print(
                                    f"    Found {len(clauses_list)} clauses (already list) for contract {contract_id}")
                                for clause in clauses_list:
                                    if isinstance(clause, str) and clause.strip():
                                        contract_clauses[contract_id].add(
                                            clause.strip())
                            elif contract_id and prediction_str == "[]":
                                # Empty prediction
                                if i < 3:
                                    print(
                                        f"    Empty generated_output for contract {contract_id}")
                        except Exception as e:
                            print(
                                f"An unexpected error occurred while processing an entry in {filepath}: {e} - Entry: {data}")
            except Exception as e:
                print(f"Could not read or process file {filepath}: {e}")

    print(
        f"\nProcessed {files_processed} files with {total_entries_processed} total entries")
    print(f"Found clauses for {len(contract_clauses)} unique contracts")

    # Convert sets of clauses to lists for JSON serialization
    output_data = {contract_id: list(
        clauses) for contract_id, clauses in contract_clauses.items()}

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully concatenated clauses to {output_file}")
        print(f"Found {len(output_data)} unique contracts with clauses")
    except Exception as e:
        print(f"Could not write output file {output_file}: {e}")


if __name__ == "__main__":
    # This should be the correct path from workspace root
    input_directory = "src/stage1/out"
    output_json_file = "src/stage1/concatenated_clauses_test.json"

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    concatenate_clauses(input_directory, output_json_file)
