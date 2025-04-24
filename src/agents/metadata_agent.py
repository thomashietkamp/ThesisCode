import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import time
import getpass

# Get the Hugging Face token
print("Authenticating with HuggingFace...")
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN not found in environment variables.")
    hf_token = getpass.getpass("Enter your Hugging Face token: ")

try:
    login(token=hf_token)
    print("Authentication completed successfully.")
except Exception as e:
    print(f"Authentication failed: {str(e)}")
    print("Proceeding without authentication (may limit access to some models)...")


def format_prompt(prompt_text, examples, question_text):
    """Format the prompt according to Gemma 3's expected format."""
    # Updated to match Gemma 3's exact expected format with proper system prompt
    formatted = f"<start_of_turn>user\n{question_text}\n\nHere are some examples:\n{examples}\n\nDocument to analyze:\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    return formatted


def generate_response(prompt_text, question_text, examples, model_name="google/gemma-3-4b-it", max_length=1024):
    """
    Generate a response using the Gemma 3 4B model.

    Args:
        prompt_text: The document/text content to analyze
        question_text: The question or instruction for the model
        model_name: The model to use
        max_length: Maximum length of generated response

    Returns:
        Generated text response
    """
    try:
        print(f"Loading model {model_name}...")
        # Load model and tokenizer - use token if authentication succeeded
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("Model loaded successfully.")
        except Exception as e:
            # If authentication failed, try without token (for models that don't require authentication)
            print(f"Failed to load model with authentication: {str(e)}")
            print("Trying to load model without authentication...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=False
            )
            print("Model loaded successfully without authentication.")

        # Format the prompt
        print("Formatting prompt...")
        formatted_prompt = format_prompt(prompt_text, examples, question_text)
        print(f"Prompt formatted. Length: {len(formatted_prompt)} characters")

        # Debug - Print first 200 characters of formatted prompt
        print(f"Prompt start: {formatted_prompt[:200]}...")

        # Tokenize the prompt
        print("Tokenizing prompt...")
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt").to(model.device)
        print(
            f"Prompt tokenized. Input IDs length: {len(inputs['input_ids'][0])}")

        # Generate response
        print("Generating response (this may take some time)...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,  # Add top_p sampling for better quality
                repetition_penalty=1.1,  # Penalize repetition
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        end_time = time.time()
        print(f"Response generated in {end_time - start_time:.2f} seconds.")

        # Decode and return the generated text
        print("Decoding response...")
        # Keep special tokens for debugging
        generated_text = tokenizer.decode(
            outputs[0], skip_special_tokens=False)
        print(f"Full generated text length: {len(generated_text)} characters")

        # Check if the model's start and end tokens are present
        print(f"Last 100 characters: {generated_text[-100:]}")

        # Extract only the model's response
        # Look for start and end of model turn
        model_turn_start = "<start_of_turn>model\n"
        model_turn_end = "<end_of_turn>"

        start_idx = formatted_prompt.find(
            model_turn_start) + len(model_turn_start)
        if start_idx == -1 + len(model_turn_start):
            start_idx = len(formatted_prompt)

        end_idx = generated_text.find(model_turn_end, start_idx)
        if end_idx == -1:
            # If there's no end token, take everything after the prompt
            response = generated_text[len(formatted_prompt):]
        else:
            response = generated_text[start_idx:end_idx]

        print("Response extracted successfully.")
        return response.strip()
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Load the contract
        contract_path = "data/CUAD_v1/full_contract_txt/ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"
        print(f"Reading contract from: {contract_path}")

        if not os.path.exists(contract_path):
            print(f"ERROR: Contract file not found at path: {contract_path}")
            exit(1)

        with open(contract_path, "r") as f:
            prompt_text = f.read()

        print(
            f"Contract loaded successfully. Length: {len(prompt_text)} characters")

        question_text = "Please extract the document name, parties, agreement date, effective date, expiration date, renewal term, notice period to terminate renewal, and governing law from the contract. Return the information in a JSON format."
        examples = """Here are some examples of the information you will be extracting: \n\n   {
            "Document Name-Answer": "MARKETING AFFILIATE AGREEMENT",
            "Parties-Answer": "Birch First Global Investments Inc. (\"Company\"); Mount Kowledge Holdings Inc. (\"Marketing Affiliate\", \"MA\")",
            "Agreement Date-Answer": "5/8/14",
            "Effective Date-Answer": "",
            "Expiration Date-Answer": "12/31/14",
            "Renewal Term-Answer": "successive 1 year",
            "Notice Period To Terminate Renewal- Answer": "30 days",
            "Governing Law-Answer": "Nevada"
        }, \n\n
        {
            "Document Name-Answer": "VIDEO-ON-DEMAND CONTENT LICENSE AGREEMENT",
            "Parties-Answer": "Rogers Cable Communications Inc. (\"Rogers\"); EuroMedia Holdings Corp. (\"Licensor\")",
            "Agreement Date-Answer": "7/11/06",
            "Effective Date-Answer": "7/11/06",
            "Expiration Date-Answer": "6/30/10",
            "Renewal Term-Answer": "2 years",
            "Notice Period To Terminate Renewal- Answer": "60 days",
            "Governing Law-Answer": "Ontario, Canada"
        },"""

        print("Starting inference...")
        response = generate_response(
            prompt_text, question_text, examples
        )

        print("\n--- GENERATED RESPONSE ---")
        print(response)
        print("--- END OF RESPONSE ---")

        # As a fallback, try with a smaller LLAMA2 model if the response is empty
        if not response.strip():
            print("\nNo response from Gemma 3 model. Trying with LLAMA2 7B model...")
            try:
                response = generate_response(
                    prompt_text, question_text, examples, model_name="meta-llama/Llama-2-7b-chat-hf"
                )
                print("\n--- GENERATED RESPONSE (LLAMA2 Model) ---")
                print(response)
                print("--- END OF RESPONSE ---")
            except Exception as e:
                print(f"Failed to use LLAMA2 model: {str(e)}")
                print("Falling back to local Gemma-2-2b model...")
                try:
                    response = generate_response(
                        prompt_text, question_text, examples, model_name="google/gemma-2-2b-it"
                    )
                    print("\n--- GENERATED RESPONSE (Gemma-2-2b Model) ---")
                    print(response)
                    print("--- END OF RESPONSE ---")
                except Exception as e2:
                    print(f"Failed to use Gemma-2-2b model: {str(e2)}")
                    print("All model attempts failed.")
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
