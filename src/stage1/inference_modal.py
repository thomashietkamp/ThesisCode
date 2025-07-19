# src/stage1/inference_modal.py
from pathlib import Path
import subprocess
import modal
import os

ROOT = Path(__file__).parent          # ← your local repo root
REMOTE = "/workspace"                 # ← where it will live in the container
CKPT_DIR = "/gemma-stage1-ckpts"
DATA_DIR = "/data"
OUTPUT_DIR = "/outputs"

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
        "scikit-learn",
        "orjson",
        "pyyaml",
        'hf_xet',
    )
    .workdir(REMOTE)
    .add_local_dir(ROOT, remote_path=REMOTE)
)

# Volumes for checkpoints, data, and outputs
ckpt_vol = modal.Volume.from_name("gemma-stage1-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("inference-data", create_if_missing=True)
output_vol = modal.Volume.from_name(
    "inference-outputs", create_if_missing=True)

# "Stub" → "App"
app = modal.App("inference-stage1", image=image)

# ------------------------------------------------ remote functions ---


@app.function(
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={
        CKPT_DIR: ckpt_vol,
        DATA_DIR: data_vol,
        OUTPUT_DIR: output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_inference_gpu(
    input_jsonl_filename: str,
    adapter_name: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    config_filename: str = "config.yaml",
):
    """Run inference on GPU (A100)"""
    input_jsonl_path = f"{DATA_DIR}/{input_jsonl_filename}"
    adapter_dir = f"{CKPT_DIR}/{adapter_name}/final"
    config_path = f"{REMOTE}/{config_filename}"

    cmd = [
        "python", "inference.py",
        "--input_jsonl_path", input_jsonl_path,
        "--output_dir", OUTPUT_DIR,
        "--adapter_dir", adapter_dir,
        "--base_model_name", base_model_name,
        "--config_path", config_path,
    ]

    print(f"Running inference on GPU with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit the output volume to persist results
    output_vol.commit()
    print(f"Inference complete. Results saved to {OUTPUT_DIR}")


@app.function(
    gpu=None,  # CPU only
    timeout=6 * 60 * 60,
    volumes={
        CKPT_DIR: ckpt_vol,
        DATA_DIR: data_vol,
        OUTPUT_DIR: output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_inference_cpu(
    input_jsonl_filename: str,
    adapter_name: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    config_filename: str = "config.yaml",
):
    """Run inference on CPU (slower but cheaper)"""
    input_jsonl_path = f"{DATA_DIR}/{input_jsonl_filename}"
    adapter_dir = f"{CKPT_DIR}/{adapter_name}"
    config_path = f"{REMOTE}/{config_filename}"

    cmd = [
        "python", "inference.py",
        "--input_jsonl_path", input_jsonl_path,
        "--output_dir", OUTPUT_DIR,
        "--adapter_dir", adapter_dir,
        "--base_model_name", base_model_name,
        "--config_path", config_path,
    ]

    print(f"Running inference on CPU with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit the output volume to persist results
    output_vol.commit()
    print(f"Inference complete. Results saved to {OUTPUT_DIR}")


@app.function(
    volumes={
        DATA_DIR: data_vol,
    },
)
def upload_data(local_jsonl_path: str):
    """Upload JSONL data file to Modal volume"""
    import shutil

    filename = os.path.basename(local_jsonl_path)
    remote_path = f"{DATA_DIR}/{filename}"

    print(f"Uploading {local_jsonl_path} to {remote_path}")
    shutil.copy2(local_jsonl_path, remote_path)

    # Commit the data volume
    data_vol.commit()
    print(f"Upload complete: {filename}")
    return filename


@app.function(
    volumes={
        OUTPUT_DIR: output_vol,
    },
)
def download_results(output_filename: str, local_output_path: str):
    """Download results from Modal volume to local filesystem"""
    import shutil

    remote_path = f"{OUTPUT_DIR}/{output_filename}"

    if not os.path.exists(remote_path):
        available_files = os.listdir(
            OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
        raise FileNotFoundError(
            f"Output file {output_filename} not found. Available files: {available_files}")

    print(f"Downloading {remote_path} to {local_output_path}")
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    shutil.copy2(remote_path, local_output_path)
    print(f"Download complete: {local_output_path}")


# ------------------------------------------------ CLI entrypoint ----

@app.local_entrypoint()
def main(
    input_jsonl_path: str,
    adapter_name: str,
    local_output_path: str = None,
    base_model_name: str = "Qwen/Qwen3-8B",
    config_filename: str = "config.yaml",
    use_gpu: bool = True,
):
    """
    Run inference using Modal Labs

    Args:
        input_jsonl_path: Local path to input JSONL file
        adapter_name: Name of the adapter directory in the checkpoints volume
        local_output_path: Local path to save results (optional, defaults to ./out/)
        base_model_name: Base model name
        config_filename: Config file name (should be in the repo)
        use_gpu: Whether to use GPU (A100) or CPU
    """

    # Validate input file exists
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    print(f"Starting inference with Modal Labs...")
    print(f"Input file: {input_jsonl_path}")
    print(f"Adapter: {adapter_name}")
    print(f"Base model: {base_model_name}")
    print(f"Using {'GPU (A100)' if use_gpu else 'CPU'}")

    # Step 1: Upload data
    print("\n1. Uploading data...")
    uploaded_filename = upload_data.remote(input_jsonl_path)

    # Step 2: Run inference
    print(f"\n2. Running inference...")
    if use_gpu:
        run_inference_gpu.remote(
            uploaded_filename,
            adapter_name,
            base_model_name,
            config_filename
        )
    else:
        run_inference_cpu.remote(
            uploaded_filename,
            adapter_name,
            base_model_name,
            config_filename
        )

    # Step 3: Download results
    print(f"\n3. Downloading results...")

    # Construct expected output filename based on input
    base_input_filename = os.path.splitext(
        os.path.basename(input_jsonl_path))[0]
    output_filename = f"{base_input_filename}_outputs.json"

    # Set default local output path if not provided
    if local_output_path is None:
        local_output_path = f"./out/{output_filename}"

    download_results.remote(output_filename, local_output_path)

    print(f"\n✅ Inference complete! Results saved to: {local_output_path}")


# ------------------------------------------------ Helper functions ----

@app.function(
    volumes={
        CKPT_DIR: ckpt_vol,
    },
)
def list_adapters():
    """List available adapters in the checkpoint volume"""
    if os.path.exists(CKPT_DIR):
        adapters = [d for d in os.listdir(
            CKPT_DIR) if os.path.isdir(os.path.join(CKPT_DIR, d))]
        print("Available adapters:")
        for adapter in adapters:
            print(f"  - {adapter}")
        return adapters
    else:
        print("No checkpoint directory found")
        return []


@app.function(
    volumes={
        OUTPUT_DIR: output_vol,
    },
)
def list_outputs():
    """List available output files"""
    if os.path.exists(OUTPUT_DIR):
        outputs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
        print("Available output files:")
        for output in outputs:
            print(f"  - {output}")
        return outputs
    else:
        print("No output directory found")
        return []
