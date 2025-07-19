#!/usr/bin/env python3
"""
Inference script for trained DPO models (coordinator, reasoner, validator).

This script can:
1. Load the three trained models from checkpoints
2. Run inference on new prompts using prompts from prompts.yaml
3. Test preference alignment on validation data
4. Compare outputs before/after DPO training

Usage:
    python inference_dpo_models.py --checkpoint-dir /path/to/checkpoints --prompt "Your prompt here"
    python inference_dpo_models.py --checkpoint-dir /path/to/checkpoints --test-dataset --samples 10
"""

import argparse
import json
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Tuple, Optional
import time
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOInferenceSystem:
    """System for running inference with trained DPO models."""

    def __init__(self, checkpoint_dir: str, prompts_yaml_path: str = "src/stage2/committee/prompts.yaml", device: str = "auto"):
        """
        Initialize the DPO inference system.

        Args:
            checkpoint_dir: Path to the checkpoint directory containing coordinator/reasoner/validator models
            prompts_yaml_path: Path to prompts.yaml file containing templates and settings
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prompts_yaml_path = Path(prompts_yaml_path)
        self.device = self._setup_device(device)

        # Load prompts and settings from YAML
        self._load_prompts_config()

        # Model containers
        self.coordinator = None
        self.reasoner = None
        self.validator = None
        self.tokenizer = None

        self._load_models()

    def _load_prompts_config(self):
        """Load prompts and configuration from YAML file."""
        logger.info(f"📝 Loading prompts from {self.prompts_yaml_path}")

        if not self.prompts_yaml_path.exists():
            raise FileNotFoundError(
                f"Prompts YAML file not found: {self.prompts_yaml_path}")

        with open(self.prompts_yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract the neutral persona
        self.persona = config['personas']['neutral_legal']
        logger.info(f"🎭 Using persona: {self.persona}")

        # Extract prompt templates
        self.coordinator_template = config['coordinator']
        self.reasoner_template = config['reasoner']
        self.validator_template = config['validator']

        # Extract temperature settings
        # Take first value from list
        self.coord_temperature = config['coord_T'][0]
        self.reasoner_temperature = config['reas_T'][0]
        self.validator_temperature = config['val_T'][0]

        logger.info(
            f"🌡️ Temperatures - Coordinator: {self.coord_temperature}, Reasoner: {self.reasoner_temperature}, Validator: {self.validator_temperature}")

    def _setup_device(self, device: str) -> str:
        """Setup device configuration."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _load_models(self):
        """Load all three models and tokenizer."""
        logger.info(f"🔄 Loading models from {self.checkpoint_dir}")

        # Check if checkpoint directory exists
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Load tokenizer
        tokenizer_path = self.checkpoint_dir / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        logger.info("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Load models
        model_configs = [
            ("coordinator", "🎯"),
            ("reasoner", "🧠"),
            ("validator", "✅")
        ]

        for model_name, emoji in model_configs:
            model_path = self.checkpoint_dir / model_name
            if not model_path.exists():
                raise FileNotFoundError(
                    f"{model_name} model not found: {model_path}")

            logger.info(f"{emoji} Loading {model_name} model...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                model = model.to(self.device)

            model.eval()
            setattr(self, model_name, model)

            logger.info(f"✅ {model_name} loaded successfully")

        logger.info("🎉 All models loaded successfully!")

    def generate_response(self, model, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response from a model."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048)

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated part (after the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return response.strip()

    def run_full_trajectory(self, prompt: str, max_length_per_step: int = 256) -> Dict[str, str]:
        """
        Run a full trajectory through all three models using prompts from YAML.

        Args:
            prompt: The input prompt (treated as clauses to analyze)
            max_length_per_step: Max tokens to generate per model

        Returns:
            Dictionary containing all outputs
        """
        logger.info(f"🚀 Running full trajectory for prompt: {prompt[:100]}...")

        # Step 1: Coordinator - create analysis plan
        coord_prompt = self.coordinator_template.replace(
            "{{persona}}", self.persona).replace("{{clauses}}", prompt)
        coord_output = self.generate_response(
            self.coordinator, coord_prompt, max_length_per_step, self.coord_temperature)

        # Step 2: Reasoner - create detailed draft report
        reasoner_prompt = self.reasoner_template.replace(
            "{{persona}}", self.persona).replace("{{clauses}}", prompt)
        reasoner_output = self.generate_response(
            self.reasoner, reasoner_prompt, max_length_per_step, self.reasoner_temperature)

        # Step 3: Validator - review and polish the draft
        validator_prompt = self.validator_template.replace(
            "{{persona}}", self.persona).replace("{{draft}}", reasoner_output)
        validator_output = self.generate_response(
            self.validator, validator_prompt, max_length_per_step, self.validator_temperature)

        return {
            "prompt": prompt,
            "persona": self.persona,
            "coordinator_plan": coord_output,
            "reasoner_draft": reasoner_output,
            "validator_final_report": validator_output,
            "temperatures": {
                "coordinator": self.coord_temperature,
                "reasoner": self.reasoner_temperature,
                "validator": self.validator_temperature
            }
        }

    def test_preference_alignment(self, dataset_path: str, num_samples: int = 10) -> Dict:
        """
        Test how well the models prefer 'plus' trajectories over 'minus' trajectories.

        Args:
            dataset_path: Path to the DPO training dataset
            num_samples: Number of samples to test

        Returns:
            Dictionary with test results
        """
        logger.info(
            f"🧪 Testing preference alignment on {num_samples} samples...")

        # Load dataset
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        # Sample random examples
        if num_samples > len(data):
            num_samples = len(data)

        indices = np.random.choice(len(data), num_samples, replace=False)
        test_samples = [data[i] for i in indices]

        results = {
            "total_samples": num_samples,
            "coordinator_correct": 0,
            "reasoner_correct": 0,
            "validator_correct": 0,
            "all_correct": 0,
            "sample_results": []
        }

        for i, sample in enumerate(test_samples):
            logger.info(f"Testing sample {i+1}/{num_samples}")

            # Test each model's preference
            sample_result = {
                "prompt": sample["prompt"][:100] + "...",
                "coordinator_prefers_plus": self._test_model_preference(
                    self.coordinator, sample, "coord"
                ),
                "reasoner_prefers_plus": self._test_model_preference(
                    self.reasoner, sample, "reason"
                ),
                "validator_prefers_plus": self._test_model_preference(
                    self.validator, sample, "valid"
                )
            }

            # Count correct preferences
            if sample_result["coordinator_prefers_plus"]:
                results["coordinator_correct"] += 1
            if sample_result["reasoner_prefers_plus"]:
                results["reasoner_correct"] += 1
            if sample_result["validator_prefers_plus"]:
                results["validator_correct"] += 1
            if all(sample_result[k] for k in ["coordinator_prefers_plus", "reasoner_prefers_plus", "validator_prefers_plus"]):
                results["all_correct"] += 1

            results["sample_results"].append(sample_result)

        # Calculate percentages
        results["coordinator_accuracy"] = results["coordinator_correct"] / num_samples
        results["reasoner_accuracy"] = results["reasoner_correct"] / num_samples
        results["validator_accuracy"] = results["validator_correct"] / num_samples
        results["all_correct_accuracy"] = results["all_correct"] / num_samples

        return results

    def _test_model_preference(self, model, sample: Dict, model_type: str) -> bool:
        """Test if a model prefers the plus trajectory over minus."""
        # Build plus and minus trajectories based on model type using YAML templates
        if model_type == "coord":
            plus_input = self.coordinator_template.replace("{{persona}}", self.persona).replace(
                "{{clauses}}", sample['prompt']) + sample['coord_out_plus']
            minus_input = self.coordinator_template.replace("{{persona}}", self.persona).replace(
                "{{clauses}}", sample['prompt']) + sample['coord_out_minus']
        elif model_type == "reason":
            plus_input = self.reasoner_template.replace("{{persona}}", self.persona).replace(
                "{{clauses}}", sample['prompt']) + sample['reason_out_plus']
            minus_input = self.reasoner_template.replace("{{persona}}", self.persona).replace(
                "{{clauses}}", sample['prompt']) + sample['reason_out_minus']
        else:  # validator
            plus_input = self.validator_template.replace("{{persona}}", self.persona).replace(
                "{{draft}}", sample['reason_out_plus']) + sample['valid_out_plus']
            minus_input = self.validator_template.replace("{{persona}}", self.persona).replace(
                "{{draft}}", sample['reason_out_minus']) + sample['valid_out_minus']

        # Calculate log probabilities
        plus_logprob = self._calculate_logprob(model, plus_input)
        minus_logprob = self._calculate_logprob(model, minus_input)

        return plus_logprob > minus_logprob

    def _calculate_logprob(self, model, text: str) -> float:
        """Calculate the log probability of a text sequence."""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=2048)

        if self.device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits

            # Calculate log probabilities
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs['input_ids'][:, 1:].contiguous()
            shift_mask = inputs['attention_mask'][:, 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_logprobs * shift_mask

            # Normalize by sequence length
            seq_length = shift_mask.sum(dim=1)
            normalized_logprob = token_logprobs.sum(dim=1) / seq_length

            return normalized_logprob.item()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained DPO models")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to checkpoint directory containing coordinator/reasoner/validator models")
    parser.add_argument("--prompts-yaml", type=str, default="src/stage2/committee/prompts.yaml",
                        help="Path to prompts.yaml file")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to run inference on (treated as clauses to analyze)")
    parser.add_argument("--test-dataset", action="store_true",
                        help="Test preference alignment on dataset")
    parser.add_argument("--dataset-path", type=str, default="dpo_training_data.jsonl",
                        help="Path to DPO training dataset")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to test for preference alignment")
    parser.add_argument("--output-file", type=str, default="dpo_inference_results.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use for inference")
    parser.add_argument("--max-length", type=int, default=2024,
                        help="Max tokens to generate per model step")

    args = parser.parse_args()

    # Initialize the inference system
    try:
        dpo_system = DPOInferenceSystem(
            args.checkpoint_dir, args.prompts_yaml, args.device)
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return 1

    results = {}

    # Single prompt inference
    if args.prompt:
        logger.info("🎯 Running single prompt inference...")
        start_time = time.time()

        trajectory = dpo_system.run_full_trajectory(
            args.prompt,
            args.max_length
        )

        inference_time = time.time() - start_time
        trajectory["inference_time_seconds"] = inference_time

        results["single_prompt"] = trajectory

        # Print results
        print("\n" + "="*80)
        print("🎯 INFERENCE RESULTS")
        print("="*80)
        print(f"Prompt (Clauses): {args.prompt}")
        print(f"Persona: {trajectory['persona']}")
        print(f"\n🎯 Coordinator Plan:")
        print(f"  {trajectory['coordinator_plan']}")
        print(f"\n🧠 Reasoner Draft Report:")
        print(f"  {trajectory['reasoner_draft']}")
        print(f"\n✅ Validator Final Report:")
        print(f"  {trajectory['validator_final_report']}")
        print(f"\n🌡️ Temperatures Used: {trajectory['temperatures']}")
        print(f"⏱️ Inference Time: {inference_time:.2f} seconds")

    # Dataset testing
    if args.test_dataset:
        logger.info("🧪 Running preference alignment testing...")

        if not Path(args.dataset_path).exists():
            logger.error(f"❌ Dataset not found: {args.dataset_path}")
            return 1

        start_time = time.time()
        test_results = dpo_system.test_preference_alignment(
            args.dataset_path, args.samples)
        test_time = time.time() - start_time

        test_results["test_time_seconds"] = test_time
        results["preference_test"] = test_results

        # Print results
        print("\n" + "="*80)
        print("🧪 PREFERENCE ALIGNMENT TEST RESULTS")
        print("="*80)
        print(f"Total Samples: {test_results['total_samples']}")
        print(
            f"Coordinator Accuracy: {test_results['coordinator_accuracy']:.2%}")
        print(f"Reasoner Accuracy: {test_results['reasoner_accuracy']:.2%}")
        print(f"Validator Accuracy: {test_results['validator_accuracy']:.2%}")
        print(
            f"All Models Correct: {test_results['all_correct_accuracy']:.2%}")
        print(f"Test Time: {test_time:.2f} seconds")

    # Save results
    if results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")

    logger.info("✅ Inference completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
