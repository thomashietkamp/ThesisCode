import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import warnings
import argparse
from huggingface_hub import login

login(token=os.environ['HF_TOKEN'])

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")
# Avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
# IMPORTANT: Match these with your training setup
# The original base model used for fine-tuning
BASE_MODEL_NAME = "google/gemma-3-4b-it"
# Path where the trained adapter weights and tokenizer are saved
ADAPTER_PATH = "src/models/metadata_agent/"

# --- Optional Configuration ---
# Use float16 or bfloat16 for loading/merging. float16 is more widely compatible.
# Check GPU compatibility if using bfloat16
compute_dtype = torch.float16
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    print("BF16 is supported. Using BF16.")
else:
    print("BF16 not supported. Using FP16.")

# Load token from environment variable if needed
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# The specific instruction query used during training
QUESTION_TEXT = "Please extract the document name, parties, agreement date, effective date, expiration date, renewal term, notice period to terminate renewal, and governing law from the contract. Return the information in a JSON format."

# Generation parameters (tune these for desired output quality/style)
MAX_NEW_TOKENS = 512   # Max tokens for the generated JSON output
TEMPERATURE = 0.1      # Lower value -> more deterministic/focused output
DO_SAMPLE = True       # Set True to enable sampling (using temperature/top_p)
TOP_P = 0.9            # Nucleus sampling (used if do_sample=True)
# Set TOP_K = 50 (or other value) if you prefer top-k sampling

# --- Function to format the prompt ---


def format_inference_prompt(contract_text):
    """Creates the input prompt string in the format expected by the model."""
    # IMPORTANT: This MUST exactly match the user turn format used during training
    return f"<start_of_turn>user\n{QUESTION_TEXT}\n{contract_text}<end_of_turn>\n<start_of_turn>model\n"

# --- Main Inference Function ---


def run_inference(contract_text_to_process):
    """Loads the model and adapter, runs inference, and processes the output."""
    print(f"--- Loading Base Model: {BASE_MODEL_NAME} ---")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=compute_dtype,  # Load in specified dtype for merging
            device_map="auto",      # Automatically use available GPU(s) or CPU
            trust_remote_code=True,
            token=os.environ['HF_TOKEN'],
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Ensure you have internet access, the correct model name, and HF_TOKEN if required.")
        return None

    print(f"--- Loading LoRA Adapter from: {ADAPTER_PATH} ---")
    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Adapter path not found: {ADAPTER_PATH}")
        print("Make sure the path is correct and the training script saved the adapter.")
        return None

    try:
        # Load the LoRA adapter onto the base model
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        print("--- Merging Adapter into Model ---")
        # Merge the adapter into the base model. This creates a standard model instance
        # in memory, making inference potentially faster than using the adapter directly.
        model = model.merge_and_unload()
        model.eval()  # Set the merged model to evaluation mode (disables dropout etc.)

    except Exception as e:
        print(f"Error loading or merging adapter: {e}")
        return None

    print(f"--- Loading Tokenizer from: {ADAPTER_PATH} ---")
    # Load the tokenizer associated with the adapter/model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH, trust_remote_code=True)
        # Ensure PAD token is set (important for batching and generation)
        if tokenizer.pad_token is None:
            print("Tokenizer missing PAD token. Setting PAD token to EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Use left padding for generation

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

    print("--- Preparing Inference Pipeline ---")
    try:
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=compute_dtype,  # Ensure pipeline uses the same dtype
            device_map="auto"         # Use the same device mapping
        )
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return None

    print("--- Formatting Prompt ---")
    formatted_prompt = format_inference_prompt(contract_text_to_process)
    # Optional: print truncated prompt for verification
    # print(f"\n--- Sending Prompt (first 300 chars): ---\n{formatted_prompt[:300]}...\n")

    print("--- Generating Completion ---")
    try:
        # Generate the text
        # Note: We pass the prompt directly. The pipeline handles tokenization.
        sequences = pipe(
            formatted_prompt,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            # top_k=TOP_K if DO_SAMPLE else None, # Alternative sampling
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,  # Important for stopping generation
            max_new_tokens=MAX_NEW_TOKENS,       # Limit the length of the JSON output
        )
        generated_text = sequences[0]['generated_text']

    except Exception as e:
        print(f"Error during generation: {e}")
        return None

    print("--- Processing Output ---")
    # Extract only the model's response part after the prompt ends
    # Find the beginning of the model's turn marker *that ends the prompt*
    prompt_end_marker = "<start_of_turn>model\n"
    model_response_start_index = generated_text.find(prompt_end_marker)

    if model_response_start_index != -1:
        # Get text *after* the marker
        model_response = generated_text[model_response_start_index +
                                        len(prompt_end_marker):].strip()
    else:
        # Fallback if the marker isn't found (shouldn't happen with this prompt format)
        print("Warning: Model start marker not found in the expected position. Output might be incomplete.")
        # Try finding the last occurrence in case the model repeated the prompt structure (less likely)
        model_response_start_index = generated_text.rfind(prompt_end_marker)
        if model_response_start_index != -1:
            model_response = generated_text[model_response_start_index + len(
                prompt_end_marker):].strip()
        else:
            model_response = generated_text  # Use the whole text as a last resort

    # Clean up potential trailing special tokens
    if model_response.endswith(tokenizer.eos_token):
        model_response = model_response[:-len(tokenizer.eos_token)].strip()
    if model_response.endswith("<end_of_turn>"):
        model_response = model_response[:-len("<end_of_turn>")].strip()

    print("\n--- Raw Model Response ---")
    print(model_response)

    print("\n--- Validating JSON ---")
    try:
        # Attempt to parse the cleaned response as JSON
        parsed_json = json.loads(model_response)
        print("Successfully parsed response as JSON.")
        return parsed_json  # Return the structured dictionary
    except json.JSONDecodeError as e:
        print(f"WARNING: Could not parse response as JSON. Error: {e}")
        print("Returning the raw string response instead.")
        return model_response  # Return the raw string if parsing failed


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a contract using a fine-tuned Gemma model.")
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to the text file containing the contract."
    )
    parser.add_argument(
        "-t", "--text",
        type=str,
        help="Direct text of the contract (use quotes if it contains spaces)."
    )
    args = parser.parse_args()

    contract_input_text = None
    file = "data/CUAD_v1/full_contract_txt/OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY AGREEMENT1.txt"
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                contract_input_text = f.read()
            print(f"--- Loaded contract from file: {file} ---")
        except FileNotFoundError:
            print(f"Error: Input file not found at {file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    elif args.text:
        contract_input_text = args.text
        print("--- Using contract text provided via argument ---")
    else:
        # If no file or text argument, fallback to pasting
        print("No input file or text provided via arguments.")
        print("Please paste your contract text below, then press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) to finish:")
        try:
            # Read multi-line input until EOF
            contract_input_text = "".join(iter(input, ""))
        except EOFError:
            pass  # Expected way to end input

    # --- Run Inference ---
    if contract_input_text and contract_input_text.strip():
        # Run the main inference function
        extracted_data = run_inference(contract_input_text)

        # --- Display Final Results ---
        if extracted_data:
            print("\n" + "="*30 + " FINAL EXTRACTED DATA " + "="*30)
            if isinstance(extracted_data, dict):
                # Pretty print the dictionary if JSON parsing succeeded
                print(json.dumps(extracted_data, indent=2))
            else:
                # Print the raw string if JSON parsing failed
                print(extracted_data)
            print("="*80)
        else:
            print("\nInference failed. No data extracted.")
    else:
        print("\nNo contract text provided or input was empty. Exiting.")

    # Optional: Clear GPU cache if running multiple inferences in a larger script
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
