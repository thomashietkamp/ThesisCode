#!/usr/bin/env python3
"""
Script to evaluate the base Qwen model (without fine-tuning) on the 
Competition & Exclusivity test dataset using OpenRouter API.

This script reads the test JSONL file, runs inference using the base Qwen model,
and outputs results in the same format as the fine-tuned model outputs.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add the stage2/committee directory to the path to import the OpenRouter wrapper
sys.path.append(os.path.join(os.path.dirname(__file__),
                '..', '..', '..', 'stage2', 'committee'))

try:
    from llm_wrapper_openrouter import chat_complete
except ImportError as e:
    print(f"Error importing OpenRouter wrapper: {e}")
    print("Make sure the OpenRouter wrapper is available at src/stage2/committee/llm_wrapper_openrouter.py")
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate base model using OpenRouter"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="../../training/Competition_&_Exclusivity_test.jsonl",
        help="Path to the input JSONL test file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="../../out/base/Competition_&_Exclusivity_test_outputs_base_llama.json",
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/llama-3.3-70b-instruct:free",
        help="OpenRouter model ID to use (default: meta-llama/llama-3.3-70b-instruct:free)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1 for more deterministic output)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    return parser.parse_args()


def load_test_data(jsonl_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load test data from JSONL file."""
    data = []

    print(f"Loading test data from: {jsonl_path}")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break

                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Could not parse line {line_num + 1}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: Could not find input file: {jsonl_path}")
        sys.exit(1)

    print(f"Loaded {len(data)} samples")
    return data


def create_prompt(contract_text: str) -> str:
    """
    Create the prompt for the model based on the contract text.
    This extracts the contract and question parts from the input text.
    """
    # The input text contains [CONTRACT], [QUESTION], [RULE], etc.
    # We need to preserve this structure for the model
    return contract_text


def run_inference_on_sample(item: Dict[str, Any], model_id: str, temperature: float) -> Dict[str, Any]:
    """Run inference on a single sample."""

    # Extract the prompt from the input
    prompt = create_prompt(item["input"])

    # Use the OpenRouter wrapper to get the model response
    try:
        # Use the chat_messages function for consistency
        response = chat_complete(
            prompt=prompt,
            model_id=model_id,
            temperature=temperature,
        )

        # Extract the actual answer part
        # The model should output the answer after [ANSWER]
        generated_output = response.strip()

        # If the response contains [ANSWER], extract the part after it
        if "[ANSWER]" in generated_output:
            generated_output = generated_output.split("[ANSWER]")[-1].strip()

    except Exception as e:
        print(f"Error during inference for ID {item['id']}: {e}")
        generated_output = "[]"  # Default to empty list on error

    # Create output in the expected format
    result = {
        "id": item["id"],
        "input_contract_snippet": item["input"][:200] + "...",
        "expected_output": item["target"],
        "generated_output": generated_output
    }

    return result


def save_results_incrementally(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_existing_results(output_path: str) -> List[Dict[str, Any]]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                print(
                    f"Found existing results file with {len(existing_results)} samples")
                return existing_results
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load existing results: {e}")
            return []
    return []


def filter_unprocessed_samples(test_data: List[Dict[str, Any]], existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out samples that have already been processed."""
    if not existing_results:
        return test_data

    # Get set of already processed IDs
    processed_ids = {result["id"] for result in existing_results}

    # Filter out already processed samples
    unprocessed_data = [
        item for item in test_data if item["id"] not in processed_ids]

    print(f"Already processed: {len(processed_ids)} samples")
    print(f"Remaining to process: {len(unprocessed_data)} samples")

    return unprocessed_data


def main():
    """Main function."""
    args = parse_arguments()

    print("="*60)
    print("Base Qwen Model Evaluation Script")
    print("="*60)
    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output_json}")
    print(f"Model: {args.model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"API delay: {args.delay}s")
    print("="*60)

    # Check if OpenRouter API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key before running this script")
        sys.exit(1)

    print(f"OpenRouter API key found: {api_key[:10]}...")

    # Load existing results to check what's already been processed
    existing_results = load_existing_results(args.output_json)

    # Load test data
    test_data = load_test_data(args.input_jsonl, args.max_samples)

    if not test_data:
        print("No test data loaded. Exiting.")
        sys.exit(1)

    # Filter out already processed samples
    unprocessed_data = filter_unprocessed_samples(test_data, existing_results)

    if not unprocessed_data:
        print("All samples have already been processed!")
        print(f"Results are available at: {args.output_json}")
        sys.exit(0)

    # Start with existing results
    results = existing_results.copy()

    print(
        f"\nStarting inference on {len(unprocessed_data)} remaining samples...")
    print(
        f"Total progress: {len(existing_results)}/{len(test_data)} samples completed")

    for i, item in enumerate(tqdm(unprocessed_data, desc="Processing samples")):
        try:
            result = run_inference_on_sample(
                item, args.model_id, args.temperature)
            results.append(result)

            # Save incrementally every 10 samples
            if (len(results) - len(existing_results)) % 10 == 0:
                save_results_incrementally(results, args.output_json)
                print(
                    f"Saved {len(results)} total results ({len(results) - len(existing_results)} new)...")

            # Add delay to respect API rate limits
            if args.delay > 0:
                time.sleep(args.delay)

        except KeyboardInterrupt:
            print(
                f"\nInterrupted by user. Saving {len(results)} total results...")
            break
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Continue with next sample
            continue

    # Save final results
    save_results_incrementally(results, args.output_json)

    print(f"\nEvaluation complete!")
    print(f"Total samples processed: {len(results)}")
    print(
        f"New samples processed in this run: {len(results) - len(existing_results)}")
    print(f"Results saved to: {args.output_json}")

    # Show some sample outputs
    if results:
        print(f"\nSample results (from latest processed):")
        # Show the last few results processed in this run
        start_idx = max(0, len(existing_results))
        recent_results = results[start_idx:start_idx+3]
        for i, result in enumerate(recent_results):
            print(f"\nSample {start_idx + i + 1}:")
            print(f"ID: {result['id']}")
            print(f"Expected: {result['expected_output'][:100]}...")
            print(f"Generated: {result['generated_output'][:100]}...")


if __name__ == "__main__":
    main()
