# src/stage1/eval/base/evaluate_with_qwen_base_modal.py
from pathlib import Path
import subprocess
import modal
import os

ROOT = Path(__file__).parent.parent.parent  # Go up to src/stage1 level
REMOTE = "/workspace"
DATA_DIR = "/data"
OUTPUT_DIR = "/outputs"

# ------------------------------------------------ image -------------
image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
    .pip_install(
        "transformers==4.51.0",
        "datasets==3.5.0",
        "bitsandbytes==0.45.1",
        "accelerate",
        "sentencepiece",
        "scikit-learn",
        "orjson",
        "pyyaml",
        "tqdm",
        'hf_xet',
    )
    .workdir(REMOTE)
    .add_local_dir(ROOT, remote_path=REMOTE)
)

# Volumes for data and outputs
data_vol = modal.Volume.from_name("eval-data", create_if_missing=True)
output_vol = modal.Volume.from_name("eval-outputs", create_if_missing=True)

# "Stub" → "App"
app = modal.App("evaluate-base-qwen", image=image)

# ------------------------------------------------ remote functions ---


@app.function(
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={
        DATA_DIR: data_vol,
        OUTPUT_DIR: output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_evaluation_gpu(
    input_jsonl_filename: str,
    model_name: str = "Qwen/Qwen3-1.7B",
    max_samples: int = None,
    temperature: float = 0.7,
    max_new_tokens: int = 500,
    use_quantization: bool = True,
):
    """Run base model evaluation on GPU (A100)"""
    input_jsonl_path = f"{DATA_DIR}/{input_jsonl_filename}"

    # Construct output filename based on input
    base_input_filename = os.path.splitext(input_jsonl_filename)[0]
    output_filename = f"{base_input_filename}_outputs_base_{model_name.replace('/', '_')}.json"
    output_path = f"{OUTPUT_DIR}/{output_filename}"

    cmd = [
        "python", "eval/base/evaluate_base_local.py",
        "--input_jsonl", input_jsonl_path,
        "--output_json", output_path,
        "--model_name", model_name,
        "--temperature", str(temperature),
        "--max_new_tokens", str(max_new_tokens),
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    if use_quantization:
        cmd.append("--use_quantization")

    print(
        f"Running base model evaluation on GPU with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit the output volume to persist results
    output_vol.commit()
    print(f"Evaluation complete. Results saved to {output_path}")
    return output_filename


@app.function(
    gpu=None,  # CPU only
    timeout=6 * 60 * 60,
    volumes={
        DATA_DIR: data_vol,
        OUTPUT_DIR: output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_evaluation_cpu(
    input_jsonl_filename: str,
    model_name: str = "Qwen/Qwen3-1.7B",
    max_samples: int = None,
    temperature: float = 0.7,
    max_new_tokens: int = 500,
):
    """Run base model evaluation on CPU (slower but cheaper)"""
    input_jsonl_path = f"{DATA_DIR}/{input_jsonl_filename}"

    # Construct output filename based on input
    base_input_filename = os.path.splitext(input_jsonl_filename)[0]
    output_filename = f"{base_input_filename}_outputs_base_{model_name.replace('/', '_')}.json"
    output_path = f"{OUTPUT_DIR}/{output_filename}"

    cmd = [
        "python", "eval/base/evaluate_with_qwen_base_local.py",
        "--input_jsonl", input_jsonl_path,
        "--output_json", output_path,
        "--model_name", model_name,
        "--temperature", str(temperature),
        "--max_new_tokens", str(max_new_tokens),
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    print(
        f"Running base model evaluation on CPU with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit the output volume to persist results
    output_vol.commit()
    print(f"Evaluation complete. Results saved to {output_path}")
    return output_filename


@app.function(
    volumes={
        DATA_DIR: data_vol,
    },
)
def upload_data(file_content: bytes, filename: str):
    """Upload JSONL data file to Modal volume"""
    remote_path = f"{DATA_DIR}/{filename}"

    print(f"Uploading {filename} to {remote_path}")

    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Write the file content to the remote path
    with open(remote_path, 'wb') as f:
        f.write(file_content)

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
    local_output_path: str = None,
    model_name: str = "Qwen/Qwen3-1.7B",
    max_samples: int = None,
    temperature: float = 0.7,
    max_new_tokens: int = 500,
    use_gpu: bool = True,
    use_quantization: bool = True,
):
    """
    Run base model evaluation using Modal Labs

    Args:
        input_jsonl_path: Local path to input JSONL file
        local_output_path: Local path to save results (optional, defaults to ./out/)
        model_name: HuggingFace model name
        max_samples: Maximum number of samples to process (for testing)
        temperature: Temperature for generation
        max_new_tokens: Maximum number of new tokens to generate
        use_gpu: Whether to use GPU (A100) or CPU
        use_quantization: Whether to use 4-bit quantization (GPU only)
    """

    # Validate input file exists
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    print(f"Starting base model evaluation with Modal Labs...")
    print(f"Input file: {input_jsonl_path}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples or 'All'}")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Using {'GPU (A100)' if use_gpu else 'CPU'}")
    if use_gpu:
        print(f"Use quantization: {use_quantization}")

    # Step 1: Upload data
    print("\n1. Uploading data...")
    filename = os.path.basename(input_jsonl_path)

    # Read the file content locally
    with open(input_jsonl_path, 'rb') as f:
        file_content = f.read()

    uploaded_filename = upload_data.remote(file_content, filename)

    # Step 2: Run evaluation
    print(f"\n2. Running evaluation...")
    if use_gpu:
        output_filename = run_evaluation_gpu.remote(
            uploaded_filename,
            model_name,
            max_samples,
            temperature,
            max_new_tokens,
            use_quantization
        )
    else:
        output_filename = run_evaluation_cpu.remote(
            uploaded_filename,
            model_name,
            max_samples,
            temperature,
            max_new_tokens
        )

    # Step 3: Download results
    print(f"\n3. Downloading results...")

    # Set default local output path if not provided
    if local_output_path is None:
        base_input_filename = os.path.splitext(
            os.path.basename(input_jsonl_path))[0]
        model_safe_name = model_name.replace('/', '_')
        output_filename_local = f"{base_input_filename}_outputs_base_{model_safe_name}.json"
        local_output_path = f"./out/base/{output_filename_local}"

    download_results.remote(output_filename, local_output_path)

    print(
        f"\n✅ Base model evaluation complete! Results saved to: {local_output_path}")


# ------------------------------------------------ Helper functions ----

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


@app.function(
    volumes={
        DATA_DIR: data_vol,
    },
)
def list_data():
    """List available data files"""
    if os.path.exists(DATA_DIR):
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
        print("Available data files:")
        for data_file in data_files:
            print(f"  - {data_file}")
        return data_files
    else:
        print("No data directory found")
        return []
