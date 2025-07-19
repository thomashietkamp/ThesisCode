"""
Stage-2 Joint Reasoning Committee for Legal Contract Analysis

This module implements a multi-agent committee system for analyzing legal contracts
using fine-tuned models with different personas and temperature settings. The committee
consists of coordinator, validator, and reasoner agents that work together to produce
comprehensive legal analysis reports.

The system generates multiple variants per contract by combining different:
- Personas (neutral, risk_averse, aggressive)
- Temperature settings for generation diversity
- Model configurations

Author: Thomas Hietkamp
Date: 2025-06-02
"""

from __future__ import annotations
import argparse
import itertools
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import tqdm

# --------------------------------------------------------------------------- #
#  0. Lightweight LLM wrapper stubs (replace with your own implementation)    #
# --------------------------------------------------------------------------- #
from llm_wrapper_openrouter import chat_with_thinking

# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
Loader = yaml.SafeLoader                      # Safe YAML loader
# Default I/O dirs (override via CLI)
SRC_DIR = BASE.parent / "data" / "stage1_out"
DST_DIR = BASE.parent / "data" / "stage2_out"
DST_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Variant-grid dataclass                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class VariantCfg:
    # generation knobs
    coord_T: float = 0.3
    coord_top_p: float = 0.92
    persona: str = "neutral"
    model_id: str = "Qwen/Qwen3-14B"
    seed: int = 42
    # validator / reasoner temps
    val_T: float = 0.4
    reas_T: float = 0.2
    # model IDs for different components
    coord_model_id: str = "Qwen/Qwen3-14B"
    val_model_id: str = "Qwen/Qwen3-14B"
    reas_model_id: str = "Qwen/Qwen3-14B"
    # optional behaviors
    self_critique: bool = False
    debate: bool = False

    @staticmethod
    def grid(grid_dict: Dict[str, list]) -> List["VariantCfg"]:
        """Cartesian product over lists in grid_dict → VariantCfg list."""
        if not grid_dict:
            return [VariantCfg()]                            # single default
        keys, vals = zip(*grid_dict.items())
        combos = (dict(zip(keys, combo)) for combo in itertools.product(*vals))
        return [VariantCfg(**c) for c in combos]


# --------------------------------------------------------------------------- #
# 2. YAML loader – splits PROMPTS vs GRID                                     #
# --------------------------------------------------------------------------- #
PROMPT_KEYS = {
    "coordinator", "common_instructions",
    "validator", "reasoner", "personas",
    "self_critique", "revise"
}


def load_yaml(yaml_path: Path, single_variant: bool = False) -> tuple[dict, List[VariantCfg]]:
    """Return (prompts_dict, list_of_variant_cfg)."""
    raw = yaml.load(open(yaml_path, "r"), Loader=Loader)

    # Handle personas conversion before splitting
    if "personas" in raw:
        # Keep personas in prompts for template rendering
        # But also add persona list to raw for grid generation
        raw["persona"] = list(raw["personas"].keys())

    prompts = {k: v for k, v in raw.items() if k in PROMPT_KEYS}

    if single_variant:
        # Create a single variant with Qwen 3 14B and neutral persona
        variant = VariantCfg(
            model_id="Qwen/Qwen3-14B",
            coord_model_id="Qwen/Qwen3-14B",
            val_model_id="Qwen/Qwen3-14B",
            reas_model_id="Qwen/Qwen3-14B",
            persona="neutral"
        )
        return prompts, [variant]

    grid = {k: v for k, v in raw.items() if k not in PROMPT_KEYS}

    # Convert models list to model_id for grid generation
    if "models" in grid:
        grid["model_id"] = grid["models"]
        del grid["models"]

    # enforce exactly 4 variants: 2 persona × 2 models
    required_keys = {"persona", "model_id"}
    assert required_keys.issubset(
        grid), "YAML must define persona and models lists"

    # Don't set component-specific model IDs as lists in the grid
    # They will be set to match model_id in the VariantCfg constructor
    # Remove them from grid if they exist to avoid cartesian product explosion
    if "coord_model_id" in grid:
        del grid["coord_model_id"]
    if "val_model_id" in grid:
        del grid["val_model_id"]
    if "reas_model_id" in grid:
        del grid["reas_model_id"]

    # Set default values for single-value parameters
    if "coord_T" not in grid:
        grid["coord_T"] = [0.6]
    if "seed" not in grid:
        grid["seed"] = [42]
    if "val_T" not in grid:
        grid["val_T"] = [0.4]
    if "reas_T" not in grid:
        grid["reas_T"] = [0.2]

    # Remove the inconsistent key if it exists
    if "coor_model_id" in grid:
        del grid["coor_model_id"]

    variants = VariantCfg.grid(grid)

    # Set component-specific model IDs to match the main model_id for each variant
    for variant in variants:
        variant.coord_model_id = variant.model_id
        variant.val_model_id = variant.model_id
        variant.reas_model_id = variant.model_id

    assert len(variants) == 4, f"Expected 4 variants, got {len(variants)}"
    return prompts, variants


# --------------------------------------------------------------------------- #
# 3. Prompt rendering helpers                                                 #
# --------------------------------------------------------------------------- #
def render(template: str, **repl) -> str:
    for k, v in repl.items():
        # Handle both formats: {{key}} and {{ key }}
        template = template.replace(f"{{{{{k}}}}}", v)  # {{key}}
        template = template.replace(f"{{{{ {k} }}}}", v)  # {{ key }}
    return template


def coord_prompt(prompts, meta: dict, clauses: list[str], persona: str) -> str:
    """Render coordinator prompt with persona description."""
    # Get persona description from prompts
    personas_dict = prompts.get("personas", {})
    persona_desc = personas_dict.get(persona, persona)

    return render(
        prompts["coordinator"],
        persona=persona_desc,
        meta=json.dumps(meta, indent=2),
        clauses=json.dumps(clauses, indent=2)
    )


def reasoner_prompt(prompts, draft: str, clauses: list[str], persona: str) -> str:
    """Render reasoner prompt with persona description."""
    personas_dict = prompts.get("personas", {})
    persona_desc = personas_dict.get(persona, persona)

    return render(prompts["reasoner"],
                  persona=persona_desc,
                  draft=draft,
                  clauses=json.dumps(clauses, indent=2))


def validator_prompt(prompts, reasoning: str, clauses: list[str], persona: str) -> str:
    """Render validator prompt with persona description."""
    personas_dict = prompts.get("personas", {})
    persona_desc = personas_dict.get(persona, persona)

    return render(prompts["validator"],
                  persona=persona_desc,
                  draft=reasoning,
                  clauses=json.dumps(clauses, indent=2))


# --------------------------------------------------------------------------- #
# 4. Contract processing check                                                #
# --------------------------------------------------------------------------- #
def get_output_filename(contract_id: str, output_dir: Path) -> Path:
    """Get the output filename for a contract ID."""
    safe_filename = contract_id.replace("/", "_").replace("\\", "_")
    return output_dir / f"{safe_filename}.json"


def is_contract_processed(contract_id: str, output_dir: Path) -> bool:
    """Check if a contract has already been processed."""
    output_file = get_output_filename(contract_id, output_dir)
    return output_file.exists()


def get_contracts_to_process(input_path: Path) -> List[str]:
    """Get list of all contract IDs that would be processed from input."""
    contracts = []

    if input_path.is_file() and input_path.name == "concatenated_clauses_validation.json":
        # Handle concatenated format
        data = json.loads(input_path.read_text())
        contracts = list(data.keys())
    elif input_path.is_file():
        # Handle single file that's not the concatenated format
        data = json.loads(input_path.read_text())
        contracts = list(data.keys())
    else:
        # Handle original format (directory of individual JSON files)
        files = sorted(input_path.glob("*.json"))
        contracts = [f.stem for f in files]

    return contracts


def print_processing_summary(all_contracts: List[str], contracts_to_process: List[str],
                             already_processed: List[str]):
    """Print a summary of what will be processed."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total contracts found: {len(all_contracts)}")
    print(f"Already processed: {len(already_processed)}")
    print(f"To be processed: {len(contracts_to_process)}")

    if already_processed:
        print(
            f"\nSkipping {len(already_processed)} already processed contracts:")
        for contract_id in sorted(already_processed)[:10]:  # Show first 10
            print(f"  - {contract_id}")
        if len(already_processed) > 10:
            print(f"  ... and {len(already_processed) - 10} more")

    if contracts_to_process:
        print(f"\nWill process {len(contracts_to_process)} contracts:")
        for contract_id in sorted(contracts_to_process)[:10]:  # Show first 10
            print(f"  - {contract_id}")
        if len(contracts_to_process) > 10:
            print(f"  ... and {len(contracts_to_process) - 10} more")

    print("="*60)

    if contracts_to_process:
        response = input("\nProceed with processing? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Processing cancelled.")
            exit(0)
    else:
        print("\nNothing to process. All contracts already completed.")
        exit(0)


# --------------------------------------------------------------------------- #
# 5. Single-variant generation                                                #
# --------------------------------------------------------------------------- #
def run_variant(prompts: dict, meta: dict, clauses: list[str],
                cfg: VariantCfg) -> Dict[str, Any]:
    random.seed(cfg.seed)

    # Step 1: Coordinator draft
    c_prompt = coord_prompt(prompts, meta, clauses, cfg.persona)
    messages = [{"role": "user", "content": c_prompt}]
    draft_thinking, draft = chat_with_thinking(messages, temperature=cfg.coord_T,
                                               model_id=cfg.coord_model_id)

    # Step 2: Reasoner detailed analysis
    r_prompt = reasoner_prompt(prompts, draft, clauses, cfg.persona)
    messages = [{"role": "user", "content": r_prompt}]
    reasoning_thinking, reasoning = chat_with_thinking(messages,
                                                       temperature=cfg.reas_T, model_id=cfg.reas_model_id)

    # Step 3: Validator final polishing
    v_prompt = validator_prompt(prompts, reasoning, clauses, cfg.persona)
    messages = [{"role": "user", "content": v_prompt}]
    final_thinking, final = chat_with_thinking(messages,
                                               temperature=cfg.val_T, model_id=cfg.val_model_id)

    return {
        "cfg": asdict(cfg),
        "coordinator_prompt": c_prompt,
        "coordinator_draft": draft,
        "coordinator_thinking": draft_thinking,
        "reasoner_analysis": reasoning,
        "reasoner_thinking": reasoning_thinking,
        "final_report": final,
        "final_thinking": final_thinking
    }


# --------------------------------------------------------------------------- #
# 6. CLI runner                                                               #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_dir",  default=SRC_DIR,
                    help="Stage-1 clause JSON dir or path to concatenated_clauses.json")
    ap.add_argument("--out", dest="out_dir", default=DST_DIR,
                    help="Where to write variant JSON")
    ap.add_argument("--cfg", required=True,
                    help="Combined YAML (prompts + grid)")
    ap.add_argument("--force", action="store_true",
                    help="Process all contracts, even if already processed")
    ap.add_argument("--single-variant", action="store_true",
                    help="Create a single variant with Qwen 3 14B and neutral persona")
    args = ap.parse_args()

    prompts, variants = load_yaml(Path(args.cfg), args.single_variant)
    if args.single_variant:
        print(
            f"DEBUG: Single variant mode - loaded 1 variant (Qwen 3 14B + neutral persona)")
    else:
        print(f"DEBUG: Multi-variant mode - loaded {len(variants)} variants")

    # Check if input is the concatenated file or a directory
    input_path = Path(args.in_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DEBUG: Input path: {input_path}")
    print(f"DEBUG: Is file? {input_path.is_file()}")
    print(f"DEBUG: File name: {input_path.name}")

    # Get all contracts and check which ones need processing
    all_contracts = get_contracts_to_process(input_path)

    if not args.force:
        already_processed = [
            c for c in all_contracts if is_contract_processed(c, output_dir)]
        contracts_to_process = [
            c for c in all_contracts if not is_contract_processed(c, output_dir)]

        # Print summary and get user confirmation
        print_processing_summary(
            all_contracts, contracts_to_process, already_processed)
    else:
        contracts_to_process = all_contracts
        print(
            f"FORCE mode: Processing all {len(contracts_to_process)} contracts")

    if input_path.is_file() and input_path.name == "concatenated_clauses_validation.json":
        print("DEBUG: Processing concatenated format")
        # Handle concatenated format
        data = json.loads(input_path.read_text())

        for contract_id in tqdm.tqdm(contracts_to_process, desc="contracts"):
            clauses = data[contract_id]
            # Create minimal metadata for each contract
            meta = {
                "contract_id": contract_id,
                "source": "concatenated_clauses_validation.json"
            }

            out: Dict[str, Any] = {
                vid: run_variant(prompts, meta, clauses, cfg)
                for vid, cfg in enumerate(variants)
            }

            # Write output with sanitized filename
            output_file = get_output_filename(contract_id, output_dir)
            output_file.write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    elif input_path.is_file():
        print("DEBUG: Processing single file (not concatenated_clauses.json)")
        # Handle single file that's not the concatenated format
        data = json.loads(input_path.read_text())
        print(f"DEBUG: Loaded data with keys: {list(data.keys())}")

        for contract_id in tqdm.tqdm(contracts_to_process, desc="contracts"):
            clauses = data[contract_id]
            # Create minimal metadata for each contract
            meta = {
                "contract_id": contract_id,
                "source": input_path.name
            }

            out: Dict[str, Any] = {
                vid: run_variant(prompts, meta, clauses, cfg)
                for vid, cfg in enumerate(variants)
            }

            # Write output with sanitized filename
            output_file = get_output_filename(contract_id, output_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print("DEBUG: Processing directory of individual JSON files")
        # Handle original format (directory of individual JSON files)
        files = sorted(input_path.glob("*.json"))
        files_to_process = [f for f in files if f.stem in contracts_to_process]

        for f in tqdm.tqdm(files_to_process, desc="contracts"):
            data = json.loads(f.read_text())
            meta, clauses = data["contract_metadata"], data["clauses"]

            out: Dict[str, Any] = {
                vid: run_variant(prompts, meta, clauses, cfg)
                for vid, cfg in enumerate(variants)
            }
            output_file = get_output_filename(f.stem, output_dir)
            output_file.write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
