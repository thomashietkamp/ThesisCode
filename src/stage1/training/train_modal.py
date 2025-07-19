# src/training/stage1/train_modal.py
from pathlib import Path
import subprocess
import modal

ROOT = Path(__file__).parent          # ← your local repo root
REMOTE = "/workspace"                 # ← where it will live in the container
CKPT_DIR = "/ckpts"

# ------------------------------------------------ image -------------
image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
    .pip_install(
        "transformers==4.51.0",
        "datasets==3.5.0",
        "bitsandbytes==0.45.1",
        "peft==0.15.2",
        "accelerate",
        "sentencepiece",
        "wandb",
        "scikit-learn",
        "orjson",
        'hf_xet',

        # "torchvision"
    )
    .workdir(REMOTE)
    .add_local_dir(ROOT, remote_path=REMOTE)

)

vol = modal.Volume.from_name("gemma-stage1-ckpts", create_if_missing=True)

# “Stub” → “App” :contentReference[oaicite:1]{index=1}
app = modal.App("gemma-stage1-a100-80gb", image=image)

# ------------------------------------------------ remote function ---


@app.function(
    gpu="A100-40GB",                              # same SKU
    timeout=6 * 60 * 60,
    volumes={CKPT_DIR: vol},
    secrets=[modal.Secret.from_name(
        "wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train(
    train_jsonl: str,
    val_jsonl: str,
    agent_name: str = "Competition_Exclusivity",
    dry_run_steps: int = 0,
):
    cmd = [
        "python", "train_stage1.py",
        "--train_jsonl", train_jsonl,
        "--val_jsonl",   val_jsonl,
        "--agent_name",  agent_name,
        "--output_dir",  f"{CKPT_DIR}/{agent_name}",
    ]
    if dry_run_steps:
        cmd += ["--dry_run_steps", str(dry_run_steps)]
    subprocess.run(cmd, check=True)

# ------------------------------------------------ CLI entrypoint ----


@app.local_entrypoint()
def main(
    train_jsonl: str = "train_small.jsonl",
    val_jsonl:   str = "val_small.jsonl",
    agent_name:  str = "Competition_Exclusivity",
    dry_run_steps: int = 0,
):
    train.remote(train_jsonl, val_jsonl, agent_name, dry_run_steps)
