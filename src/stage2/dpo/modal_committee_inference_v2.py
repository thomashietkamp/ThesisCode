#!/usr/bin/env python3
"""
Modal Labs script to run committee inference using DPO-trained models.
Follows the same workflow as committee.py but simplified to 1 variant per contract.

This script:
1. Loads the DPO-trained checkpoints from Modal volume
2. Runs committee-based inference with 1 variant (DPO + neutral persona)
3. Saves individual JSON files for each contract
4. Uses the same prompt templates and processing logic
5. Includes thinking outputs using Qwen's thinking mode
"""

import modal
from pathlib import Path
import json
import torch
from typing import List, Dict, Any

# Define paths
ROOT = Path(__file__).parent
REMOTE = "/workspace"
DATA_DIR = "/data"

# Create Modal app
app = modal.App("committee-inference-v2")

# Define the container image with all dependencies
image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
    .pip_install([
        "transformers==4.51.0",
        "datasets==3.5.0",
        "bitsandbytes==0.45.1",
        "peft==0.15.2",
        "accelerate",
        "sentencepiece",
        "torch",
        "numpy",
        "tqdm",
        "fire",
        "pyyaml",
    ])
    .workdir(REMOTE)
    .add_local_dir(ROOT, remote_path=REMOTE)
    .add_local_file(ROOT.parent / "committee" / "llm_wrapper.py", remote_path=f"{REMOTE}/llm_wrapper.py")
)

# Volume for checkpoints and outputs
volume = modal.Volume.from_name("dpo-data", create_if_missing=True)


@app.function(
    # Use 3x A100 80GB GPUs for the 3x 14B models
    gpu="A100-80GB:3",
    memory=192_000,  # 192GB RAM
    timeout=7200,    # 2 hours timeout
    image=image,
    volumes={DATA_DIR: volume},
)
def run_committee_inference(
    checkpoint_name: str = "dpo-qwen3-14b",
    input_dataset: str = "concatenated_clauses_test.json",
    output_dir: str = "fine_tuned_committee",
    max_new_tokens: int = 2048,  # Increased default
    prompts_config: str = "prompts.yaml",
    force: bool = False,
):
    """
    Run committee inference using DPO fine-tuned models following committee.py structure.

    Args:
        checkpoint_name: Name of the DPO checkpoint to use
        input_dataset: Path to input dataset JSON file
        output_dir: Directory to save individual contract JSON files
        max_new_tokens: Maximum tokens to generate
        prompts_config: Path to prompts YAML configuration
        force: Process all contracts even if already processed
    """
    import json
    import yaml
    import random
    import torch
    from pathlib import Path
    from dataclasses import dataclass, asdict
    from typing import Any, Dict, List
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from tqdm import tqdm

    @dataclass
    class VariantCfg:
        coord_T: float = 0.3
        coord_top_p: float = 0.92
        persona: str = "neutral"
        model_id: str = "Qwen/Qwen3-14B"
        seed: int = 42
        val_T: float = 0.4
        reas_T: float = 0.2
        coord_model_id: str = "Qwen/Qwen3-14B"
        val_model_id: str = "Qwen/Qwen3-14B"
        reas_model_id: str = "Qwen/Qwen3-14B"
        self_critique: bool = False
        debate: bool = False

    def render_prompt(template: str, **repl) -> str:
        """Render template with replacements - matches committee.py exactly."""
        for k, v in repl.items():
            template = template.replace(f"{{{{{k}}}}}", v)
            template = template.replace(f"{{{{ {k} }}}}", v)
        return template

    def get_output_filename(contract_id: str, output_dir: Path) -> Path:
        """Get output filename for contract - matches committee.py exactly."""
        safe_filename = contract_id.replace("/", "_").replace("\\", "_")
        return output_dir / f"{safe_filename}.json"

    def is_contract_processed(contract_id: str, output_dir: Path) -> bool:
        """Check if contract already processed - matches committee.py exactly."""
        return get_output_filename(contract_id, output_dir).exists()

    print(f"Starting committee inference for checkpoint: {checkpoint_name}")

    # Load prompts configuration - matches committee.py exactly
    prompts_path = Path(REMOTE) / prompts_config
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts config not found: {prompts_config}")

    with open(prompts_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract prompts (same as committee.py)
    PROMPT_KEYS = {
        "coordinator", "common_instructions",
        "validator", "reasoner", "personas",
        "self_critique", "revise"
    }
    prompts = {k: v for k, v in config.items() if k in PROMPT_KEYS}

    # Create single variant: DPO fine-tuned model with neutral persona
    variants = [
        VariantCfg(
            persona="neutral_legal",
            model_id="dpo-finetuned",
            seed=42,
            coord_T=0.3,
            val_T=0.4,
            reas_T=0.2
        ),
    ]

    print(
        f"Created {len(variants)} variant (DPO fine-tuned with neutral persona)")

    # Load input dataset - matches committee.py logic
    input_path = Path(REMOTE) / input_dataset
    if not input_path.exists():
        input_path = Path(DATA_DIR) / input_dataset
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {input_dataset}")

    print(f"Loading input dataset from: {input_path}")

    # Handle concatenated format (matches committee.py)
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} contracts from {input_dataset}")

    # Create output directory
    output_path = Path(DATA_DIR) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Check which contracts need processing (matches committee.py)
    all_contracts = list(data.keys())
    if not force:
        already_processed = [
            c for c in all_contracts if is_contract_processed(c, output_path)]
        contracts_to_process = [
            c for c in all_contracts if not is_contract_processed(c, output_path)]
        print(
            f"Found {len(already_processed)} already processed, {len(contracts_to_process)} to process")

        if not contracts_to_process:
            print("All contracts already processed. Use --force to reprocess.")
            return "All contracts already processed"
    else:
        contracts_to_process = all_contracts
        print(
            f"Force mode: processing all {len(contracts_to_process)} contracts")

    # Load models
    print("Loading models...")

    # Set up paths
    data_dir = Path(DATA_DIR)
    checkpoint_dir = data_dir / "checkpoints" / checkpoint_name / "final"

    # Set device mapping for 3 GPUs
    device_map = {
        "coordinator": "cuda:0",
        "reasoner": "cuda:1",
        "validator": "cuda:2"
    }

    # Load base tokenizer (same for all models)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}

    # Load each model with its checkpoint
    for role in ["coordinator", "reasoner", "validator"]:
        print(f"Loading {role} model...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-14B",
            torch_dtype=torch.float16,
            device_map=device_map[role],
            trust_remote_code=True
        )

        # Load PEFT adapter if available
        adapter_path = checkpoint_dir / role
        if adapter_path.exists():
            print(f"Loading PEFT adapter for {role} from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model = model.merge_and_unload()  # Merge for faster inference
        else:
            print(f"No PEFT adapter found for {role}, using base model")
            model = base_model

        model.eval()
        models[role] = model

    def generate_response_with_thinking(model, tokenizer, prompt: str, temperature: float, device: str) -> tuple[str, str]:
        """Generate response with thinking from model using chat format."""
        try:
            # Create messages format
            messages = [{"role": "user", "content": prompt}]

            # Build the full prompt, trying with thinking mode first
            try:
                chat_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            except TypeError:
                # Fallback if enable_thinking is not supported
                print(
                    f"Warning: enable_thinking not supported, using standard chat template")
                chat_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Manually add thinking prompt
                chat_prompt = chat_prompt.replace(
                    "<|im_start|>assistant\n",
                    "<|im_start|>assistant\n<think>\nLet me think through this step by step.\n</think>\n\n"
                )

            # Tokenize & move to correct device
            inputs = tokenizer([chat_prompt], return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[-1]

            # Generate with increased token limits and better parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Increased from max_new_tokens parameter
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )

            # Decode the full response first
            full_response = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=False
            ).strip()

            print(f"    Generated response length: {len(full_response)}")
            print(f"    Response preview: {full_response[:200]}...")

            # Look for thinking tags in the response
            thinking_content = ""
            final_content = ""

            if "<think>" in full_response and "</think>" in full_response:
                # Extract thinking content
                think_start = full_response.find("<think>") + len("<think>")
                think_end = full_response.find("</think>")
                if think_start > len("<think>") - 1 and think_end > think_start:
                    thinking_content = full_response[think_start:think_end].strip(
                    )
                    # Extract content after thinking
                    after_think = full_response[think_end +
                                                len("</think>"):].strip()
                    final_content = after_think
                else:
                    # Malformed thinking tags, treat as regular response
                    final_content = full_response.replace(
                        "<think>", "").replace("</think>", "").strip()
            else:
                # No thinking tags found, treat entire response as final content
                final_content = full_response.replace(
                    "<think>", "").replace("</think>", "").strip()

            # Clean up any remaining special tokens
            thinking_content = thinking_content.replace(
                "<|im_end|>", "").replace("<|im_start|>", "").strip()
            final_content = final_content.replace(
                "<|im_end|>", "").replace("<|im_start|>", "").strip()

            return thinking_content, final_content

        except Exception as e:
            print(f"Error in generate_response_with_thinking: {e}")
            # Fallback to simple generation without thinking
            try:
                inputs = tokenizer([prompt], return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[-1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

                return "", response  # Empty thinking, full response as content

            except Exception as e2:
                print(f"Fallback generation also failed: {e2}")
                return "", "Error: Failed to generate response"

    def run_single_variant(prompts: dict, meta: dict, clauses: list, cfg: VariantCfg) -> Dict[str, Any]:
        """Run single variant - matches committee.py structure exactly."""
        random.seed(cfg.seed)

        # Get persona description
        personas_dict = prompts.get("personas", {})
        persona_desc = personas_dict.get(cfg.persona, cfg.persona)

        # Step 1: Coordinator draft (matches committee.py)
        coord_prompt = render_prompt(
            prompts["coordinator"],
            persona=persona_desc,
            meta=json.dumps(meta, indent=2),
            clauses=json.dumps(clauses, indent=2)
        )

        print(f"  Running coordinator with temperature {cfg.coord_T}")
        coord_thinking, coord_draft = generate_response_with_thinking(
            models["coordinator"], tokenizer, coord_prompt,
            cfg.coord_T, device_map["coordinator"]
        )
        print(
            f"  Coordinator generated {len(coord_thinking)} thinking chars, {len(coord_draft)} content chars")

        # Step 2: Reasoner detailed analysis (matches committee.py)
        reasoner_prompt = render_prompt(
            prompts["reasoner"],
            persona=persona_desc,
            draft=coord_draft,
            clauses=json.dumps(clauses, indent=2)
        )

        print(f"  Running reasoner with temperature {cfg.reas_T}")
        reasoner_thinking, reasoner_analysis = generate_response_with_thinking(
            models["reasoner"], tokenizer, reasoner_prompt,
            cfg.reas_T, device_map["reasoner"]
        )
        print(
            f"  Reasoner generated {len(reasoner_thinking)} thinking chars, {len(reasoner_analysis)} content chars")

        # Step 3: Validator final polishing (matches committee.py)
        validator_prompt = render_prompt(
            prompts["validator"],
            persona=persona_desc,
            draft=reasoner_analysis,
            clauses=json.dumps(clauses, indent=2)
        )

        print(f"  Running validator with temperature {cfg.val_T}")
        validator_thinking, final_report = generate_response_with_thinking(
            models["validator"], tokenizer, validator_prompt,
            cfg.val_T, device_map["validator"]
        )
        print(
            f"  Validator generated {len(validator_thinking)} thinking chars, {len(final_report)} content chars")

        # Return exact same structure as committee.py
        return {
            "cfg": asdict(cfg),
            "coordinator_prompt": coord_prompt,
            "coordinator_draft": coord_draft,
            "coordinator_thinking": coord_thinking,
            "reasoner_analysis": reasoner_analysis,
            "reasoner_thinking": reasoner_thinking,
            "final_report": final_report,
            "final_thinking": validator_thinking
        }

    # Process contracts (matches committee.py exactly)
    processed_count = 0

    for contract_id in tqdm(contracts_to_process, desc="contracts"):
        print(f"Processing contract: {contract_id}")

        clauses = data[contract_id]
        meta = {
            "contract_id": contract_id,
            "source": input_dataset
        }

        # Run single variant for this contract (DPO fine-tuned with neutral persona)
        contract_output: Dict[str, Any] = {
            str(vid): run_single_variant(prompts, meta, clauses, cfg)
            for vid, cfg in enumerate(variants)
        }

        # Save individual contract file (same structure as committee.py)
        output_file = get_output_filename(contract_id, output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(contract_output, f, indent=2, ensure_ascii=False)

        processed_count += 1
        if processed_count % 10 == 0:
            print(
                f"Processed {processed_count}/{len(contracts_to_process)} contracts")

    print(
        f"Committee inference completed. Processed {processed_count} contracts.")
    print(f"Output saved to: {output_path}/")
    print(f"Each contract saved as individual JSON file with 1 variant each (DPO + neutral persona)")

    return f"Successfully processed {processed_count} contracts with 1 variant each (DPO fine-tuned + neutral persona)."


@app.function(image=image, volumes={DATA_DIR: volume})
def list_available_checkpoints():
    """List available checkpoints in the data volume."""
    from pathlib import Path

    data_dir = Path(DATA_DIR)
    checkpoint_dir = data_dir / "checkpoints"

    if not checkpoint_dir.exists():
        return "No checkpoint directory found"

    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Check if it has final directory with model files
            final_dir = item / "final"
            if final_dir.exists():
                # Look for agent directories (coordinator, reasoner, validator)
                agent_dirs = []
                for agent in ["coordinator", "reasoner", "validator"]:
                    agent_path = final_dir / agent
                    if agent_path.exists():
                        agent_dirs.append(agent)

                checkpoints.append({
                    "name": item.name,
                    "path": str(item),
                    "has_final_dir": True,
                    "available_agents": agent_dirs,
                    "agent_count": len(agent_dirs)
                })
            else:
                checkpoints.append({
                    "name": item.name,
                    "path": str(item),
                    "has_final_dir": False,
                    "available_agents": [],
                    "agent_count": 0
                })

    return checkpoints


@app.function(image=image, volumes={DATA_DIR: volume})
def upload_test_dataset(local_path: str = None):
    """Upload a test dataset to Modal volume for inference."""
    import shutil
    import json
    from pathlib import Path

    data_dir = Path(DATA_DIR)

    # If no local path provided, try to find in workspace
    if local_path is None:
        # Look for contract clauses test file first, then other formats
        workspace_files = [
            Path(REMOTE) / "concatenated_clauses_test.json",
            Path(REMOTE) / "test_data.jsonl",
            Path(REMOTE) / "inference_data.jsonl",
            Path(REMOTE) / "dpo_test_data.jsonl"
        ]

        for file_path in workspace_files:
            if file_path.exists():
                local_path = str(file_path)
                break

    if local_path and Path(local_path).exists():
        file_path = Path(local_path)
        target_path = data_dir / file_path.name
        shutil.copy(local_path, target_path)

        # Count examples
        if file_path.name.endswith('.json'):
            # Count contracts and clauses in JSON format
            with open(target_path, 'r') as f:
                data = json.load(f)
            total_clauses = sum(len(clauses) for clauses in data.values())
            return f"Uploaded {len(data)} contracts (with {total_clauses} total clauses) to {target_path}. Will process {len(data)} examples (one per contract)."
        else:
            # Count lines for JSONL
            with open(target_path, 'r') as f:
                line_count = sum(1 for _ in f)
            return f"Uploaded {line_count} examples to {target_path}"
    else:
        return f"No test dataset found. Checked: {local_path}"


@app.local_entrypoint()
def main(
    checkpoint_name: str = "dpo-qwen3-14b",
    input_dataset: str = "concatenated_clauses_test.json",
    output_dir: str = "fine_tuned_committee",
    max_new_tokens: int = 2048,  # Increased default
    prompts_config: str = "prompts.yaml",
    force: bool = False,
    list_checkpoints: bool = False,
    upload_data: bool = False,
):
    """
    Local entrypoint to run committee inference on Modal.

    Usage:
        # List available checkpoints
        modal run modal_committee_inference_v2.py --list-checkpoints

        # Upload test data
        modal run modal_committee_inference_v2.py --upload-data

        # Run inference
        modal run modal_committee_inference_v2.py --checkpoint-name "dpo-qwen3-14b" --input-dataset "concatenated_clauses_test.json"

        # Run with force (reprocess all)
        modal run modal_committee_inference_v2.py --checkpoint-name "dpo-qwen3-14b" --force
    """

    if list_checkpoints:
        print("Listing available checkpoints...")
        checkpoints = list_available_checkpoints.remote()
        print("Available checkpoints:")
        for checkpoint in checkpoints:
            print(
                f"  - {checkpoint['name']}: {checkpoint['agent_count']} agents ({', '.join(checkpoint['available_agents'])})")
        return

    if upload_data:
        print("Uploading test dataset...")
        result = upload_test_dataset.remote()
        print(result)
        return

    print(f"Starting committee inference:")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Input dataset: {input_dataset}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Prompts config: {prompts_config}")
    print(f"  Force reprocess: {force}")
    print(f"  Expected output: Individual JSON files per contract with 1 variant each (DPO + neutral)")

    # Run committee inference
    print("Launching committee inference on Modal...")
    result = run_committee_inference.remote(
        checkpoint_name=checkpoint_name,
        input_dataset=input_dataset,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
        prompts_config=prompts_config,
        force=force,
    )

    print("Committee inference completed!")
    print(f"Summary: {result}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
