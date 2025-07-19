#!/usr/bin/env python3
"""
Modal Labs script to run DPO training on cloud GPUs.

This script sets up the environment, mounts necessary files, and runs the DPO training
script with proper GPU resources for 3x 14B models.
"""

import modal
from pathlib import Path
import subprocess

# Define paths - Follow same pattern as working stage1 script
ROOT = Path(__file__).parent  # Just the dpo directory
REMOTE = "/workspace"
DATA_DIR = "/data"

# Create Modal app
app = modal.App("dpo-training")

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
        "wandb",
        "scikit-learn",
        "trl",
        "fire",
    ])
    .workdir(REMOTE)
    .add_local_dir(ROOT, remote_path=REMOTE)
)

# Volume for dataset and checkpoints
volume = modal.Volume.from_name("dpo-data", create_if_missing=True)


@app.function(
    # Use 4x A100 80GB GPUs for the 3x 14B models + overhead
    gpu="A100-80GB:4",
    # High memory instance
    memory=256_000,  # 256GB RAM
    timeout=86400,   # 24 hours timeout
    image=image,
    volumes={DATA_DIR: volume},
    # Set environment variables
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_dpo_training(
    model_name: str = "Qwen/Qwen3-14B",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_grad_norm: float = 1.0,
    alpha: float = 1.0,
    save_steps: int = 500,
    eval_steps: int = 100,
    use_wandb: bool = True,
    experiment_name: str = "dpo-qwen3-14b",
):
    """
    Run DPO training with the specified parameters.

    Args:
        model_name: HuggingFace model name to use as base
        epochs: Number of training epochs
        batch_size: Global batch size (will be divided by 3 for each model)
        learning_rate: Learning rate for optimization
        max_grad_norm: Gradient clipping threshold
        alpha: DPO sharpness parameter
        save_steps: Steps between model saves
        eval_steps: Steps between evaluation runs
        use_wandb: Whether to log to Weights & Biases
        experiment_name: Name for the experiment
    """
    import os
    import shutil
    from pathlib import Path

    # Set up working directory
    os.chdir(REMOTE)

    # Set up data directory
    data_dir = Path(DATA_DIR)
    checkpoint_dir = data_dir / "checkpoints" / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset to data volume if it doesn't exist
    dataset_path = data_dir / "dpo_training_data.jsonl"
    if not dataset_path.exists():
        local_dataset = Path(f"{REMOTE}/dpo_training_data.jsonl")
        if local_dataset.exists():
            print("Copying dataset to data volume...")
            shutil.copy(local_dataset, dataset_path)
        else:
            raise FileNotFoundError("DPO training dataset not found!")

    # Set environment variables for Modal and the existing DPO script
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TRANSFORMERS_CACHE"] = str(data_dir / "hf_cache")
    os.environ["HF_HOME"] = str(data_dir / "hf_cache")
    if use_wandb:
        os.environ["WANDB_PROJECT"] = experiment_name
        os.environ["WANDB_RUN_NAME"] = f"{experiment_name}-{model_name.split('/')[-1]}"

    # Configure environment variables for the existing DPO script to override defaults
    os.environ["DPO_MODEL_NAME"] = model_name
    os.environ["DPO_NUM_EPOCHS"] = str(epochs)
    os.environ["DPO_GLOBAL_BATCH_SIZE"] = str(batch_size)
    os.environ["DPO_LEARNING_RATE"] = str(learning_rate)
    os.environ["DPO_MAX_GRAD_NORM"] = str(max_grad_norm)
    os.environ["DPO_ALPHA"] = str(alpha)
    os.environ["DPO_SAVE_STEPS"] = str(save_steps)
    os.environ["DPO_DATASET_PATH"] = str(dataset_path)
    os.environ["DPO_CHECKPOINT_DIR"] = str(checkpoint_dir)

    print(f"Starting DPO training with:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Checkpoint Dir: {checkpoint_dir}")

    # Run the existing DPO training script
    print("Starting DPO training script...")
    cmd = ["python", "dpo.py"]
    result = subprocess.run(cmd, cwd=REMOTE, check=True)

    print("DPO training completed successfully!")

    # Return status
    return {
        "status": "completed",
        "experiment_name": experiment_name,
        "checkpoint_dir": str(checkpoint_dir),
    }


@app.function(image=image, volumes={DATA_DIR: volume})
def download_dataset_to_volume():
    """Download and prepare the DPO dataset to Modal volume."""
    import shutil
    import os
    from pathlib import Path

    # This would run to upload the dataset
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    # Copy dataset if available in the workspace
    local_dataset = Path(f"{REMOTE}/dpo_training_data.jsonl")
    print(f"Looking for dataset at: {local_dataset}")
    if local_dataset.exists():
        print("Dataset found! Copying to volume...")
        shutil.copy(local_dataset, data_dir / "dpo_training_data.jsonl")
        return f"Dataset uploaded to volume: {len(open(data_dir / 'dpo_training_data.jsonl').readlines())} lines"
    else:
        return f"Dataset not found at: {local_dataset}"


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-14B",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    experiment_name: str = "dpo-qwen3-14b",
    use_wandb: bool = False,
):
    """
    Local entrypoint to run DPO training on Modal.

    Usage:
        modal run modal_dpo_runner.py --model-name "Qwen/Qwen3-14B" --epochs 3
    """
    print(f"Starting DPO training experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # First ensure dataset is available
    print("Checking dataset availability...")
    dataset_status = download_dataset_to_volume.remote()
    print(f"Dataset status: {dataset_status}")

    # Run the training
    print("Launching DPO training on Modal...")
    result = run_dpo_training.remote(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
    )

    print(f"Training completed with result: {result}")
    return result


if __name__ == "__main__":
    import fire
    fire.Fire(main)
