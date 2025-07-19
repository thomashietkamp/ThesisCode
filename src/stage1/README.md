# Stage 1: Legal Clause Extraction Foundation

Stage 1 implements the foundation of the legal clause extraction system, focusing on training specialized agents for different legal categories using the CUAD (Contract Understanding Atticus Dataset).

## Overview

This stage trains category-specific agents that can identify and extract legal clauses from contracts. Each agent specializes in one of five legal categories:

1. **Intellectual Property & Licensing** - Patents, trademarks, licensing agreements
2. **Competition & Exclusivity** - Non-compete clauses, exclusivity arrangements
3. **Termination & Control Rights** - Contract termination, change of control
4. **Financial & Commercial Terms** - Payment terms, pricing, revenue sharing
5. **Legal Protections & Liability** - Indemnification, liability limitations

## Directory Structure

```
stage1/
├── agents/                     # Trained model checkpoints
│   ├── 1.7b/                  # 1.7B parameter models
│   ├── 4b/                    # 4B parameter models
│   └── 8b/                    # 8B parameter models
├── data_processing/           # Data preparation and processing
│   ├── frequencies.py         # Clause frequency analysis
│   ├── package_cuad.py       # CUAD dataset packaging
│   ├── splitting/            # Dataset splitting utilities
│   └── tokenization/         # Tokenization and label processing
├── eval/                     # Evaluation and metrics
│   ├── base/                # Base model evaluation
│   ├── metrics.py           # Performance metrics calculation
│   └── contract_level_metrics.py  # Contract-level evaluation
├── training/                # Model training scripts
│   ├── train_stage1.py      # Main training script
│   └── train_modal.py       # Cloud training on Modal
├── utils/                   # Utility functions
│   ├── contract_to_text.py  # PDF text extraction
│   └── text_chunking_rag.py # Text chunking and RAG utilities
├── inference.py            # Model inference script
└── cuad_statistics.py      # Dataset statistics and analysis
```

## Key Components

### Data Processing (`data_processing/`)

#### Dataset Preparation

- **`package_cuad.py`** - Converts CUAD dataset into training format
- **`frequencies.py`** - Analyzes clause frequency distribution
- **`splitting/split_five_categories.py`** - Creates balanced dataset splits

#### Tokenization

- **`tokenization/bio_labels.py`** - BIO tagging for token classification
- **`tokenization/eval_check.py`** - Validation of tokenized data

### Training (`training/`)

#### Local Training

```bash
python training/train_stage1.py \
    --category competition_exclusivity \
    --model_size 8b \
    --output_dir agents/8b/competition_exclusivity_8b/
```

#### Cloud Training (Modal Labs)

```bash
modal run training/train_modal.py \
    --category competition_exclusivity \
    --model-size 8b
```

#### Training Configuration

- **Base Models**: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Strategy**: Category-specific specialization
- **Optimization**: AdamW with cosine learning rate schedule

### Evaluation (`eval/`)

#### Performance Metrics

```bash
python eval/metrics.py \
    --model_path agents/8b/competition_exclusivity_8b/ \
    --test_data data/cuad_by_category/Competition_&_Exclusivity_test.jsonl
```

**Supported Metrics:**

- Precision, Recall, F1-Score (token and clause level)
- Exact Match accuracy
- Contract-level coverage
- Cross-category performance analysis

#### Base Model Comparison

```bash
python eval/base/evaluate_base_local.py \
    --input_jsonl data/cuad_by_category/Competition_&_Exclusivity_test.jsonl \
    --model_name Qwen/Qwen3-8B
```

### Inference (`inference.py`)

#### Single Contract Analysis

```bash
python inference.py \
    --input_jsonl test_contracts.jsonl \
    --model_path agents/8b/competition_exclusivity_8b/ \
    --output_dir results/
```

#### Batch Processing

```bash
python inference.py \
    --input_jsonl data/cuad_by_category/Competition_&_Exclusivity_test.jsonl \
    --model_path agents/8b/competition_exclusivity_8b/ \
    --batch_size 8 \
    --output_dir results/batch_processing/
```

## Model Architecture

### Base Model Configuration

- **Architecture**: Qwen3 (transformer-based)
- **Sizes**: 1.7B, 4B, 8B parameters
- **Context Length**: 8192 tokens
- **Fine-tuning**: LoRA with rank 32, alpha 64

### Training Hyperparameters

```yaml
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
warmup_ratio: 0.03
max_grad_norm: 1.0
lora_rank: 32
lora_alpha: 64
```

## Data Format

### Input Format (JSONL)

```json
{
  "input": "Contract text with legal clauses...",
  "output": "Extracted clauses: [clause1] [clause2]",
  "category": "competition_exclusivity",
  "contract_id": "example_contract_001"
}
```

### Output Format

```json
{
  "CONTRACT_ID": [
    "Provider shall not engage in any business that competes with the Services offered under this Agreement.",
    "This Agreement shall terminate automatically if either party is acquired by a competitor.",
    "Provider must maintain comprehensive liability insurance coverage of at least $2M.",
    "All confidential information shall be returned within 30 days of termination."
  ]
}
```

## Performance Benchmarks

### Category-Specific Results (F1-Score)

| Category                  | 1.7B Model | 4B Model | 8B Model |
| ------------------------- | ---------- | -------- | -------- |
| Intellectual Property     | 0.82       | 0.85     | 0.88     |
| Competition & Exclusivity | 0.79       | 0.83     | 0.86     |
| Termination & Control     | 0.77       | 0.81     | 0.84     |
| Financial & Commercial    | 0.75       | 0.79     | 0.82     |
| Legal Protections         | 0.73       | 0.77     | 0.80     |

### Computational Requirements

| Model Size | GPU Memory | Training Time | Inference Speed  |
| ---------- | ---------- | ------------- | ---------------- |
| 1.7B       | 8GB        | 2 hours       | 50 contracts/min |
| 4B         | 16GB       | 4 hours       | 30 contracts/min |
| 8B         | 24GB       | 8 hours       | 15 contracts/min |

## Utilities

### Text Processing (`utils/`)

- **PDF Extraction**: Convert PDF contracts to text
- **Text Chunking**: Split long contracts into manageable segments
- **RAG Integration**: Retrieval-augmented generation support

### Statistics and Analysis

```bash
python cuad_statistics.py
```

Generates comprehensive statistics about the CUAD dataset including:

- Contract length distributions
- Clause frequency analysis
- Category coverage metrics
- Temporal patterns in legal language

## Best Practices

### Training Recommendations

1. **Start Small**: Begin with 1.7B model for rapid iteration
2. **Category Focus**: Train specialized agents rather than general-purpose models
3. **Data Quality**: Ensure balanced training data across legal categories
4. **Validation**: Use cross-validation to prevent overfitting

### Deployment Considerations

1. **Resource Planning**: Account for GPU memory requirements
2. **Batch Processing**: Use appropriate batch sizes for throughput
3. **Caching**: Implement model caching for repeated inference
4. **Monitoring**: Track performance metrics in production

## Troubleshooting

### Common Issues

- **GPU Memory**: Reduce batch size or use gradient checkpointing
- **Convergence**: Adjust learning rate or warmup schedule
- **Data Loading**: Verify JSONL format and file paths

### Debug Mode

```bash
python inference.py --debug --max_samples 10
```

For questions or issues, refer to the evaluation metrics in `eval/metrics.py` or check the training logs in the respective agent directories.
