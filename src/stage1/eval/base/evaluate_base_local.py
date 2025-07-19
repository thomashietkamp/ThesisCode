#!/usr/bin/env python3
"""
Script to evaluate the base Qwen model (without fine-tuning) on the 
Competition & Exclusivity test dataset using local HuggingFace transformers.

This script reads the test JSONL file, runs inference using the base Qwen model locally,
and outputs results in the same format as the fine-tuned model outputs.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen model on Competition & Exclusivity test dataset using local HuggingFace transformers"
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
        default="../../out/base/Competition_&_Exclusivity_test_outputs_base_local.json",
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name to use (default: Qwen/Qwen3-1.7B)"
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
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate (default: 500)"
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization for memory efficiency (CUDA only)"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, use_quantization: bool = False):
    """Load the base model and tokenizer."""
    # Determine the device
    if torch.backends.mps.is_available():
        device = "mps"
        print("MPS device found. Using MPS.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("CUDA device found. Using CUDA.")
    else:
        device = "cpu"
        print("MPS and CUDA not available. Using CPU.")

    print(f"Loading base model: {model_name} onto {device}")

    # Configuration for loading the model
    model_args = {
        "device_map": device,
        "trust_remote_code": True,
    }

    if device == "cuda" and use_quantization:
        # For CUDA, use 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model_args["quantization_config"] = bnb_config
    elif device == "mps":
        # For MPS, use bfloat16 without BitsAndBytes quantization
        model_args["torch_dtype"] = torch.bfloat16
    # For CPU, use default configuration without quantization

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_args
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        tokenizer.padding_side = "left"

        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval()  # Set the model to evaluation mode

        print("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


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


def run_inference(model, tokenizer, prompt: str, temperature: float = 0.7, max_new_tokens: int = 500) -> str:
    """Run inference on a single prompt."""

    # Build the messages list
    messages = [{"role": "user", "content": prompt}]

    # Render into one string, disabling the <think>…</think> step if available
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # disable Qwen3's reasoning wrapper if available
        )
    except TypeError:
        # Fallback for models that don't support enable_thinking parameter
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Tokenize and move to device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # Strip prompt tokens and decode only the newly generated tokens
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return result.strip()


def run_inference_on_sample(item: Dict[str, Any], model, tokenizer, temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    """Run inference on a single sample."""

    # Extract the prompt from the input
    prompt = item["input"]

    try:
        # Run inference using the local model
        generated_output = run_inference(
            model, tokenizer, prompt, temperature, max_new_tokens
        )

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


def main():
    """Main function."""
    args = parse_arguments()

    print("="*60)
    print("Base Qwen Model Evaluation Script (Local)")
    print("="*60)
    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output_json}")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Use quantization: {args.use_quantization}")
    print("="*60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.use_quantization)

    # Load test data
    test_data = load_test_data(args.input_jsonl, args.max_samples)

    if not test_data:
        print("No test data loaded. Exiting.")
        sys.exit(1)

    # Process each sample
    results = []

    print(f"\nStarting inference on {len(test_data)} samples...")

    for i, item in enumerate(tqdm(test_data, desc="Processing samples")):
        try:
            result = run_inference_on_sample(
                item, model, tokenizer, args.temperature, args.max_new_tokens
            )
            results.append(result)

            # Save incrementally every 10 samples
            if (i + 1) % 10 == 0:
                save_results_incrementally(results, args.output_json)
                print(f"Saved {len(results)} results so far...")

        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Saving {len(results)} results...")
            break
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Continue with next sample
            continue

    # Save final results
    save_results_incrementally(results, args.output_json)

    print(f"\nEvaluation complete!")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {args.output_json}")

    # Show some sample outputs
    if results:
        print(f"\nSample results:")
        for i, result in enumerate(results[:3]):
            print(f"\nSample {i+1}:")
            print(f"ID: {result['id']}")
            print(f"Expected: {result['expected_output'][:100]}...")
            print(f"Generated: {result['generated_output'][:100]}...")


if __name__ == "__main__":
    main()
