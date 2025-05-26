import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference using a fine-tuned model to extract clauses from contracts in a JSONL file.")
    parser.add_argument("--input_jsonl_path", required=True,
                        help="Path to the input JSONL file containing contracts.")
    parser.add_argument("--output_dir", default="out",
                        help="Directory to save the output JSON file (default: out).")
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to the fine-tuned LoRA adapter directory.")
    parser.add_argument("--base_model_name", default="Qwen/Qwen3-1.7B",
                        help="Name of the base model used for fine-tuning.")
    parser.add_argument("--config_path", default="config.yaml",
                        help="Path to the configuration file.")
    return parser.parse_args()


def load_model_and_tokenizer(base_model_name, adapter_dir):
    """Loads the base model, tokenizer, and applies the LoRA adapter."""
    # Determine the device
    if torch.backends.mps.is_available():
        device = "mps"
        print("MPS device found. Using MPS.")
    elif torch.cuda.is_available():
        device = "cuda"  # Keep cuda as an option if user switches hardware
        print("CUDA device found. Using CUDA.")
    else:
        device = "cpu"
        print("MPS and CUDA not available. Using CPU.")

    print(f"Loading base model: {base_model_name} onto {device}")

    # Configuration for loading the model in 4-bit for efficiency if needed
    # Use different model loading configurations based on the device
    model_args = {
        "device_map": device,
        "trust_remote_code": True,
    }

    if device == "cuda":
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_args
    )
    base_model = model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<no_answer>"]})

    # Set padding token if not already set (common for Llama/Mistral)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Loading adapter from: {adapter_dir}")
    # Check if adapter directory exists
    if not os.path.isdir(adapter_dir):
        print(f"Error: Adapter directory '{adapter_dir}' not found.")
        exit(1)

    model.resize_token_embeddings(len(tokenizer))

    # Load the LoRA adapter
    try:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        try:
            print("Loaded adapters:", model.get_adapter_names())
        except AttributeError:
            print("Could not retrieve adapter names using model.get_adapter_names().")
        except Exception as e:
            print(f"Error checking adapter names: {e}")
        try:

            # Attempt to access peft_config through the active adapter if direct access fails
            # This structure might vary depending on the peft library version
            active_adapter = getattr(model, 'active_adapter', None)
            if active_adapter and hasattr(model, 'peft_config') and active_adapter in model.peft_config:
                print("PEFT config:", model.peft_config[active_adapter])
            else:
                print(
                    "Could not retrieve specific PEFT config, attempting direct access (might fail).")
                # Fallback or alternative access method if needed
                # print("PEFT config (direct):", model.peft_config) # Example fallback
        except AttributeError:
            print("Could not retrieve PEFT config using model.peft_config.")
        except Exception as e:
            print(f"Error checking PEFT config: {e}")

        print("Adapter loaded successfully.")  # Simplified success message
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print(
            "Ensure the adapter directory contains 'adapter_config.json' and model weights.")
        exit(1)

    model.eval()  # Set the model to evaluation mode

    return model, tokenizer


def run_inference(model, tokenizer, prompt):
    import torch

    print("\n--- Running Inference (thinking disabled) ---")

    # 1) Build the messages list
    messages = [{"role": "user", "content": prompt}]

    # 2) Render into one string, disabling the <think>…</think> step
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # disable Qwen3's reasoning wrapper
    )

    # 3) Tokenize and move to device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # 4) Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7
        )

    # 5) Strip prompt tokens and decode only the newly generated tokens
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("--- Inference Complete ---")
    return result


def main():
    args = parse_arguments()
    config = load_config(args.config_path)

    model, tokenizer = load_model_and_tokenizer(
        args.base_model_name, args.adapter_dir)

    results = []

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct output file path
    base_input_filename = os.path.splitext(
        os.path.basename(args.input_jsonl_path))[0]
    output_filename = f"{base_input_filename}_outputs.json"
    output_filepath = os.path.join(args.output_dir, output_filename)
    print(f"Output will be incrementally saved to {output_filepath}")

    try:
        with open(args.input_jsonl_path, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in):
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(
                        f"Skipping line {line_num + 1} due to JSON decode error: {e}")
                    continue

                if 'input' not in item:
                    print(
                        f"Skipping line {line_num + 1} as 'input' key is missing.")
                    continue

                contract_text = item['input']
                print(
                    f"\nProcessing contract {line_num + 1} from {args.input_jsonl_path}...")
                # Optional: print contract snippet
                # print(f"Contract snippet: {contract_text[:200]}")

                # Use item['input'] directly as the prompt
                prompt = contract_text

                # Optional: print the formatted prompt for verification for each item
                print("\n--- Prompt ---")
                print(prompt)

                generated_text = run_inference(model, tokenizer, prompt)

                results.append({
                    # Store a snippet for reference
                    "input_contract_snippet": contract_text[:200] + "...",
                    # Store full original input
                    "expected_output": item.get("target"),
                    "generated_output": generated_text
                })
                print(f"--- Generated Output for contract {line_num + 1} ---")
                print(generated_text)

                # Save results incrementally
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                        json.dump(results, f_out, indent=4, ensure_ascii=False)
                    print(
                        f"Successfully saved {len(results)} results to {output_filepath}")
                except Exception as e:
                    print(
                        f"Error writing intermediate results to JSON file: {e}")
                    # Decide if you want to exit or continue if intermediate save fails
                    # For now, it will print error and continue

    except FileNotFoundError:
        print(f"Error: Input JSONL file '{args.input_jsonl_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error processing JSONL file: {e}")
        exit(1)

    # Final confirmation message after the loop completes
    if results:
        print(
            f"\nProcessing complete. Final results saved to {output_filepath}")
    else:
        print("\nNo results were generated or saved.")


if __name__ == "__main__":
    main()
