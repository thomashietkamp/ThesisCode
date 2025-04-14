# Gemma 3 Inference Script

A simple script for generating responses using Google's Gemma 3 4B model.

## Requirements

```
pip install torch transformers huggingface_hub
```

## Authentication

This script requires authentication with Hugging Face to download the Gemma model. You need to:

1. Create a Hugging Face account and generate an access token at https://huggingface.co/settings/tokens
2. Set the token as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token
```

Or pass it directly before running the script:

```bash
HF_TOKEN=your_huggingface_token python src/gemma_inference.py --prompt "..."
```

## Usage

Run the script with a prompt text and optional question:

```bash
python src/gemma_inference.py --prompt "Your document text here" --question "Your question here"
```

### Default Parameters

- The default model is `google/gemma-3-4b-it` (instruction-tuned version)
- Default question (if not specified): "Please extract the document name, parties, agreement date, effective date, expiration date, renewal term, notice period to terminate renewal, and governing law from the contract. Return the information in a JSON format."
- Default max length: 1024 tokens

### Example

```bash
python src/gemma_inference.py --prompt "This agreement is made on January 15, 2023 between ABC Corp and XYZ Inc. The effective date is February 1, 2023 and it expires on January 31, 2024. The agreement will automatically renew for additional 1-year terms unless terminated with 30 days notice. This agreement is governed by the laws of the State of California."
```
