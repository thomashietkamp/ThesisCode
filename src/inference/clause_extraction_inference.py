import argparse
import json
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"

# Categories
CATEGORIES = {
    "intellectual_property_licensing": "Intellectual Property & Licensing",
    "competition_exclusivity": "Competition & Exclusivity",
    "termination_control": "Termination & Control Rights",
    "financial_commercial": "Financial & Commercial Terms",
    "legal_protections_liability": "Legal Protections & Liability"
}


def load_model(category):
    """Load the model and tokenizer for a specific category"""
    model_path = MODELS_DIR / f"gemma-3-{category}"

    if not model_path.exists():
        logger.error(f"Model for {category} not found at {model_path}")
        return None, None

    try:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def extract_clauses(text, category, model, tokenizer):
    """Extract clauses from the text using the model"""
    # Format prompt
    prompt = f"Analyze the following legal text and extract all clauses related to {CATEGORIES[category]}. " \
             f"For each clause, return its exact text and the corresponding label.\n\n" \
             f"Text:\n{text}\n\nOutput (JSON):"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=4096,  # Adjust as needed
            temperature=0.1,  # Lower temperature for more deterministic outputs
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON part
    json_str = output_text.split("Output (JSON):")[1].strip()

    try:
        # Parse JSON
        extracted_clauses = json.loads(json_str)
        return extracted_clauses
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON output: {e}")
        logger.error(f"Raw output: {json_str}")
        return []


def process_file(file_path, category=None):
    """Process a file and extract clauses"""
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    results = {}

    if category:
        # Process specific category
        model, tokenizer = load_model(category)
        if model and tokenizer:
            results[category] = extract_clauses(
                text, category, model, tokenizer)
    else:
        # Process all categories
        for cat in CATEGORIES.keys():
            model, tokenizer = load_model(cat)
            if model and tokenizer:
                results[cat] = extract_clauses(text, cat, model, tokenizer)

    return results


def save_results(results, output_file):
    """Save results to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract clauses from legal documents")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to the input file")
    parser.add_argument("--category", type=str, choices=list(CATEGORIES.keys()),
                        help="Specific category to extract")
    parser.add_argument(
        "--output", type=str, help="Path to the output file (default: extracted_clauses.json)")
    args = parser.parse_args()

    # Set default output file if not provided
    output_file = args.output if args.output else "extracted_clauses.json"

    # Process file
    results = process_file(args.file, args.category)

    # Save results
    save_results(results, output_file)


if __name__ == "__main__":
    main()
