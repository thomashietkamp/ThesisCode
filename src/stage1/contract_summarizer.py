#!/usr/bin/env python3
"""
Contract Summarizer using Google Gemini API

This script reads contracts from the CUAD_v1.json dataset and generates
approximately 200-word summaries for each contract using Google's Gemini API.
Each summary is saved incrementally as a JSON file with the contract ID as the filename.
"""

import json
import os
import time
import google.generativeai as genai
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class ContractSummarizer:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview"):
        """
        Initialize the ContractSummarizer.

        Args:
            api_key (str): Google Gemini API key
            model (str): Gemini model to use for summarization
        """
        self.api_key = api_key
        self.model_name = model

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(model)

        # Generation config for consistent outputs
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 350,  # ~200-250 words
        }

    def load_contracts(self, file_path: str) -> list:
        """Load contracts from CUAD_v1.json file."""
        print(f"Loading contracts from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data['data'])} contracts")
        return data['data']

    def load_existing_summaries(self, output_file: str) -> dict:
        """Load existing summaries from the output JSON file."""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(
                    f"Warning: Could not load existing summaries from {output_file}, starting fresh")
                return {}
        return {}

    def create_summary_prompt(self, contract_text: str, contract_title: str) -> str:
        """Create a prompt for summarizing the contract."""
        return f"""Please provide a comprehensive summary of the following contract in approximately 200 words. 
The summary should capture the key terms, parties involved, main obligations, and important provisions.

Contract Title: {contract_title}

Contract Text:
{contract_text}

Please provide a clear, concise summary that would be useful for legal professionals to quickly understand the contract's main components. Focus on:
- Parties involved
- Main purpose/type of agreement
- Key obligations and responsibilities
- Important terms and conditions
- Duration/termination clauses (if any)
- Financial arrangements (if any)

Summary:"""

    def generate_summary(self, contract_text: str, contract_title: str) -> str:
        """Generate a summary using Google Gemini API."""
        prompt = self.create_summary_prompt(contract_text, contract_title)

        try:
            # Generate content using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            if response.text:
                return response.text.strip()
            else:
                print(f"Empty response for contract: {contract_title}")
                return None

        except Exception as e:
            print(f"API request failed for {contract_title}: {e}")
            return None

    def save_summaries_to_file(self, summaries_dict: dict, output_file: str):
        """Save all summaries to a single JSON file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries_dict, f, indent=2, ensure_ascii=False)

    def save_progress_log(self, output_dir: str, processed_count: int, total_count: int,
                          last_processed_id: str):
        """Save progress log for resuming processing."""
        progress_file = Path(output_dir) / "processing_progress.json"

        progress_data = {
            "processed_count": processed_count,
            "total_count": total_count,
            "last_processed_id": last_processed_id,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completion_percentage": round((processed_count / total_count) * 100, 2)
        }

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)

    def process_contracts(self, input_file: str, output_file: str = "contract_summaries.json",
                          start_index: int = 0, max_contracts: int = None):
        """
        Process all contracts and generate summaries in a single JSON file.

        Args:
            input_file (str): Path to CUAD_v1.json
            output_file (str): Path to output JSON file with all summaries
            start_index (int): Index to start processing from (for resuming)
            max_contracts (int): Maximum number of contracts to process (None for all)
        """
        contracts = self.load_contracts(input_file)

        # Load existing summaries
        summaries_dict = self.load_existing_summaries(output_file)
        print(f"Loaded {len(summaries_dict)} existing summaries")

        # Determine which contracts to process
        if max_contracts:
            contracts_to_process = contracts[start_index:start_index + max_contracts]
            total_contracts = max_contracts
        else:
            contracts_to_process = contracts[start_index:]
            total_contracts = len(contracts) - start_index

        print(
            f"Processing {len(contracts_to_process)} contracts starting from index {start_index}")
        print(f"Output file: {output_file}")
        print(f"Model: {self.model_name}")
        print("-" * 60)

        processed_count = 0
        skipped_count = 0

        for i, contract in enumerate(tqdm(contracts_to_process, desc="Generating summaries")):
            current_index = start_index + i + 1
            contract_id = contract['title']
            contract_text = contract['paragraphs'][0]['context']

            # Check if summary already exists
            if contract_id in summaries_dict:
                print(
                    f"⏭️  Summary for contract {current_index} already exists, skipping: {contract_id}")
                skipped_count += 1
                continue

            print(
                f"\n🔄 Processing contract {current_index}/{len(contracts)}: {contract_id}")
            print(f"   Contract length: {len(contract_text):,} characters")

            # Generate summary
            summary = self.generate_summary(contract_text, contract_id)

            if summary:
                # Add summary to the dictionary
                summaries_dict[contract_id] = {
                    "contract_index": current_index,
                    "contract_id": contract_id,
                    "contract_title": contract_id,
                    "summary": summary,
                    "contract_length_chars": len(contract_text),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": self.model_name
                }

                # Save the entire dictionary to file
                self.save_summaries_to_file(summaries_dict, output_file)

                print(f"✓ Saved summary {current_index}: {contract_id}")
                processed_count += 1

                # Update progress log every 10 contracts
                if processed_count % 10 == 0:
                    output_dir = os.path.dirname(output_file) or "."
                    self.save_progress_log(output_dir, processed_count + skipped_count,
                                           len(contracts), contract_id)

            else:
                print(f"❌ Failed to generate summary for: {contract_id}")

            # Add a small delay to respect rate limits
            # Gemini has generous rate limits, so shorter delay
            time.sleep(0.5)

        # Final progress update
        output_dir = os.path.dirname(output_file) or "."
        self.save_progress_log(output_dir, processed_count + skipped_count,
                               len(contracts), contract_id if contracts_to_process else "None")

        print(f"\n{'='*60}")
        print(f"✅ Processing completed!")
        print(f"   New summaries generated: {processed_count}")
        print(f"   Skipped (already existed): {skipped_count}")
        print(f"   Total processed: {processed_count + skipped_count}")
        print(f"   Total summaries in file: {len(summaries_dict)}")
        print(f"   Output file: {output_file}")
        print(f"{'='*60}")


def main():
    """Main function to run the contract summarizer."""
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("You can get an API key from https://aistudio.google.com/app/apikey")
        print("\nTo set it, run:")
        print("export GOOGLE_API_KEY='your_api_key_here'")
        return

    # Configuration
    input_file = "../../data/CUAD_v1/CUAD_v1.json"
    output_file = "contract_summaries.json"

    # Initialize summarizer (using gemini-1.5-flash for speed and cost efficiency)
    summarizer = ContractSummarizer(api_key, model="gemini-1.5-flash")

    # Process contracts
    try:
        # For testing, you might want to limit the number of contracts initially
        # summarizer.process_contracts(input_file, output_file, max_contracts=5)

        # To process all contracts:
        summarizer.process_contracts(input_file, output_file)

    except KeyboardInterrupt:
        print("\n\n🛑 Processing interrupted by user")
        print("You can resume processing by running the script again - it will skip already processed contracts.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
