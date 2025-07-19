# Base Model Evaluation Scripts

This directory contains scripts to evaluate the base Qwen model (without fine-tuning) on test datasets using local HuggingFace transformers instead of OpenRouter API.

## Files

- `evaluate_with_qwen_base.py` - Original script using OpenRouter API
- `evaluate_with_qwen_base_local.py` - New script using local HuggingFace transformers
- `evaluate_with_qwen_base_modal.py` - Modal Labs version for cloud execution

## Local Evaluation

### Prerequisites

```bash
pip install transformers torch bitsandbytes accelerate tqdm
```

### Usage

Basic usage:

```bash
python evaluate_with_qwen_base_local.py \
    --input_jsonl ../../training/Competition_&_Exclusivity_test.jsonl \
    --model_name Qwen/Qwen3-8B
```

With custom parameters:

```bash
python evaluate_with_qwen_base_local.py \
    --input_jsonl ../../training/Competition_&_Exclusivity_test.jsonl \
    --output_json ../../out/base/my_test_outputs.json \
    --model_name Qwen/Qwen3-4B \
    --temperature 0.1 \
    --max_new_tokens 256 \
    --max_samples 10 \
    --use_quantization
```

### Arguments

- `--input_jsonl`: Path to input JSONL test file
- `--output_json`: Path to output JSON file (default: based on input filename)
- `--model_name`: HuggingFace model name (default: Qwen/Qwen3-8B)
- `--temperature`: Generation temperature (default: 0.7)
- `--max_new_tokens`: Maximum tokens to generate (default: 500)
- `--max_samples`: Limit number of samples for testing (optional)
- `--use_quantization`: Use 4-bit quantization for memory efficiency (CUDA only)

## Modal Labs Evaluation

### Prerequisites

1. Install Modal:

```bash
pip install modal
```

2. Set up Modal authentication:

```bash
modal setup
```

3. Create HuggingFace secret in Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

### Usage

Basic usage:

```bash
modal run evaluate_with_qwen_base_modal.py \
    --input-jsonl-path ../../training/Competition_&_Exclusivity_test.jsonl
```

With custom parameters:

```bash
modal run evaluate_with_qwen_base_modal.py \
    --input-jsonl-path ../../training/Competition_&_Exclusivity_test.jsonl \
    --model-name Qwen/Qwen3-4B \
    --max-samples 100 \
    --temperature 0.1 \
    --max-new-tokens 256 \
    --use-gpu false \
    --local-output-path ./my_results.json
```

### Modal Arguments

- `--input-jsonl-path`: Local path to input JSONL file
- `--local-output-path`: Local path to save results (optional)
- `--model-name`: HuggingFace model name (default: Qwen/Qwen3-8B)
- `--max-samples`: Limit number of samples for testing (optional)
- `--temperature`: Generation temperature (default: 0.7)
- `--max-new-tokens`: Maximum tokens to generate (default: 500)
- `--use-gpu`: Use GPU (A100) or CPU (default: true)
- `--use-quantization`: Use 4-bit quantization on GPU (default: true)

### Helper Commands

List available data files:

```bash
modal run evaluate_with_qwen_base_modal.py::list_data
```

List available output files:

```bash
modal run evaluate_with_qwen_base_modal.py::list_outputs
```

## Device Support

### Local Script

- **CUDA**: Supports 4-bit quantization with BitsAndBytes
- **MPS (Apple Silicon)**: Uses bfloat16 precision
- **CPU**: Uses default precision (slower)

### Modal Script

- **GPU**: A100-80GB with optional 4-bit quantization
- **CPU**: For smaller models or cost optimization

## Output Format

Both scripts generate JSON files with the same format:

```json
[
  {
    "id": "sample_id",
    "input_contract_snippet": "Contract text snippet...",
    "expected_output": "Expected answer",
    "generated_output": "Model generated answer"
  }
]
```

## Memory Requirements

| Model    | CUDA (4-bit) | MPS/CPU |
| -------- | ------------ | ------- |
| Qwen3-4B | ~3GB         | ~8GB    |
| Qwen3-8B | ~5GB         | ~16GB   |

For larger models, use Modal Labs with A100 GPU or enable quantization locally.
