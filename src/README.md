# Source Code Directory

This directory contains the core implementation of the legal clause extraction system organized into two main stages:

## Directory Structure

### Stage 1: Data Processing and Model Training

- **`stage1/`** - Foundation stage focusing on individual legal categories
  - **Data Processing** - Scripts for preparing and splitting CUAD dataset
  - **Model Training** - Fine-tuning scripts for specialized legal agents
  - **Evaluation** - Metrics and evaluation frameworks
  - **Inference** - Model inference and prediction utilities
  - **Agents** - Trained model checkpoints for different legal categories

### Stage 2: Committee Analysis and Human Evaluation

- **`stage2/`** - Advanced multi-agent analysis and preference learning
  - **Committee System** - Multi-agent reasoning for contract analysis
  - **DPO Training** - Direct Preference Optimization for model alignment
  - **Human Evaluation** - Web interface and statistical analysis tools
  - **Voting System** - Human preference collection and analysis

## Key Features

### Stage 1 Capabilities

- **Multi-category Training**: Specialized agents for 5 legal categories:

  - Intellectual Property & Licensing
  - Competition & Exclusivity
  - Termination & Control Rights
  - Financial & Commercial Terms
  - Legal Protections & Liability

- **Scalable Training Pipeline**: Support for different model sizes (1.7B, 4B, 8B parameters)
- **Comprehensive Evaluation**: Contract-level and clause-level metrics
- **Flexible Inference**: Both local and cloud-based deployment options

### Stage 2 Capabilities

- **Multi-agent Committee**: Coordinator, Validator, and Reasoner roles
- **Preference Learning**: DPO training for human-aligned outputs
- **Human Evaluation Platform**: Web-based voting system for preference collection
- **Statistical Analysis**: Comprehensive analysis of human preferences and model performance

## Quick Start

### Stage 1 Training

```bash
# Train a legal agent for a specific category
cd stage1
python training/train_stage1.py --category competition_exclusivity --model_size 8b

# Run evaluation
python eval/metrics.py --model_path agents/8b/competition_exclusivity_8b/
```

### Stage 2 Committee Analysis

```bash
# Run committee analysis
cd stage2/committee
python committe.py --in ../data/stage1_out/concatenated_clauses.json

# Launch human evaluation interface
cd ../website_vote
python app.py
```

## Dependencies

### Core Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT for parameter-efficient fine-tuning

### Stage 1 Specific

- Datasets library for data processing
- Tiktoken for tokenization
- Scikit-learn for evaluation metrics

### Stage 2 Specific

- Flask for web interface
- OpenRouter API integration
- Statistical analysis libraries (scipy, pandas)

## Model Architecture

The system uses Qwen-based models fine-tuned with LoRA (Low-Rank Adaptation) for efficient training:

- **Base Models**: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
- **Fine-tuning Method**: Parameter Efficient Fine-Tuning (PEFT) with LoRA
- **Training Strategy**: Category-specific specialization with cross-validation

## Output Formats

### Stage 1 Output

```json
{
  "CONTRACT_ID": [
    "Provider shall not engage in any business that competes with the Services offered under this Agreement.",
    "This Agreement shall terminate automatically if either party is acquired by a competitor.",
    "Provider must maintain comprehensive liability insurance coverage of at least $2M."
  ]
}
```

### Stage 2 Output

```json
{
  "0": {
    "cfg": {
      "persona": "neutral_legal",
      "model_id": "qwen/qwen3-14b:free",
      "self_critique": false,
      "debate": false
    },
    "coordinator_draft": "Analysis plan focusing on termination rights and assignment restrictions...",
    "reasoner_analysis": "### Legal Analysis Report...",
    "final_report": "# Legal Review Report\n\n## Summary of Key Clauses\n- Termination rights\n- Assignment restrictions\n- Liability limitations..."
  }
}
```

## Research Context

This codebase supports research in:

- Legal AI and contract understanding
- Multi-agent systems for document analysis
- Human preference learning in specialized domains
- Parameter-efficient fine-tuning for legal applications

For detailed documentation of specific components, see the README files in each subdirectory.
