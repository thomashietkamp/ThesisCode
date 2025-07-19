# Legal Contract Analysis with Multi-Agent Systems and Human Preference Learning

A comprehensive research system for legal contract analysis using fine-tuned language models, multi-agent reasoning, and human preference optimization. This project implements a two-stage approach combining specialized legal clause extraction with committee-based analysis and Direct Preference Optimization (DPO) for human-aligned outputs.

## 🏗️ System Architecture

The system consists of two main stages:

### Stage 1: Specialized Legal Agents

Fine-tuned models for extracting specific types of legal clauses:

1. **Intellectual Property & Licensing** - Patents, trademarks, licensing agreements
2. **Competition & Exclusivity** - Non-compete clauses, exclusivity arrangements
3. **Termination & Control Rights** - Contract termination, change of control
4. **Financial & Commercial Terms** - Payment terms, pricing, revenue sharing
5. **Legal Protections & Liability** - Indemnification, liability limitations

### Stage 2: Multi-Agent Committee & Preference Learning

- **Committee System**: Coordinator, Validator, and Reasoner agents for comprehensive analysis
- **DPO Training**: Direct Preference Optimization based on human feedback
- **Human Evaluation**: Web-based platform for collecting human preferences
- **Statistical Analysis**: Comprehensive evaluation of model performance

## 📁 Project Structure

```
Thesis/
├── data/                          # Legal contract datasets
│   ├── CUAD_v1/                  # Contract Understanding Atticus Dataset
│   │   ├── full_contract_pdf/    # Original PDF contracts (500+ files)
│   │   ├── full_contract_txt/    # Extracted text files
│   │   └── CUAD_v1.json         # Labeled dataset (41MB)
│   ├── cuad_by_category/         # Category-specific training data
│   └── split_filenames/          # Train/validation/test splits
├── src/                          # Source code (two-stage system)
│   ├── stage1/                   # Foundation: Specialized legal agents
│   │   ├── agents/              # Trained model checkpoints
│   │   ├── data_processing/     # Dataset preparation
│   │   ├── training/            # Model training scripts
│   │   ├── eval/               # Evaluation metrics
│   │   └── utils/              # Utility functions
│   └── stage2/                  # Advanced: Multi-agent analysis
│       ├── committee/           # Multi-agent committee system
│       ├── dpo/                # Direct Preference Optimization
│       ├── website_vote/       # Human evaluation platform
│       └── voting/             # Preference analysis
├── notebooks/                   # Jupyter analysis notebooks
├── output_plots/               # Generated visualizations
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM for 8B models

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd Thesis
```

2. **Set up virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download the CUAD dataset** (if not included):

```bash
# The CUAD dataset should be placed in data/CUAD_v1/
# Download from: https://www.atticusprojectai.org/cuad
```

## 🔧 Usage

### Stage 1: Train Specialized Legal Agents

1. **Prepare the dataset**:

```bash
cd src/stage1
python data_processing/package_cuad.py
```

2. **Train category-specific models**:

```bash
# Train a single category (e.g., Competition & Exclusivity)
python training/train_stage1.py \
    --train_jsonl ../../data/cuad_by_category/Competition_&_Exclusivity_train.jsonl \
    --val_jsonl ../../data/cuad_by_category/Competition_&_Exclusivity_validation.jsonl \
    --agent_name competition_exclusivity_8b \
    --output_dir agents/8b/competition_exclusivity_8b/

# Train all categories
./train_all_categories.sh
```

3. **Run evaluation**:

```bash
python eval/metrics.py \
    --model_path agents/8b/competition_exclusivity_8b/ \
    --test_data ../../data/cuad_by_category/Competition_&_Exclusivity_test.jsonl
```

### Stage 2: Multi-Agent Committee Analysis

1. **Run committee analysis**:

```bash
cd src/stage2/committee
python committe.py \
    --in ../data/stage1_out/concatenated_clauses.json \
    --cfg prompts.yaml \
    --out ../data/stage2_out/
```

2. **Launch human evaluation interface**:

```bash
cd ../website_vote
python app.py
# Visit http://localhost:7860 to start evaluating
```

3. **Create DPO training data**:

```bash
cd ../data
python create_dpo_dataset.py \
    --stage2_dir stage2_out/ \
    --voting_data ../website_vote/voting_results.csv \
    --output_file ../dpo/dpo_training_data.jsonl
```

4. **Train DPO models**:

```bash
cd ../dpo
python dpo.py \
    --dataset_path dpo_training_data.jsonl \
    --base_model Qwen/Qwen3-8B \
    --output_dir checkpoints/dpo_8b/
```

## 🔍 Model Inference

### Single Contract Analysis

```bash
# Stage 1: Extract clauses with specialized agents
cd src/stage1
python inference.py \
    --input_jsonl test_contracts.jsonl \
    --base_model_name Qwen/Qwen3-8B \
    --adapter_dir agents/8b/competition_exclusivity_8b/ \
    --output_dir results/

# Stage 2: Comprehensive committee analysis
cd ../stage2/committee
python generate_test_single.py \
    --contract_id example_contract \
    --output_dir ../data/test_reports/
```

### Batch Processing

```bash
# Process multiple contracts
python inference.py \
    --input_jsonl data/cuad_by_category/Competition_&_Exclusivity_test.jsonl \
    --batch_size 8 \
    --output_dir results/batch_processing/
```

## 📊 Performance & Results

### Stage 1 Performance (F1-Scores)

| Legal Category            | 1.7B Model | 4B Model | 8B Model |
| ------------------------- | ---------- | -------- | -------- |
| Intellectual Property     | 0.82       | 0.85     | **0.88** |
| Competition & Exclusivity | 0.79       | 0.83     | **0.86** |
| Termination & Control     | 0.77       | 0.81     | **0.84** |
| Financial & Commercial    | 0.75       | 0.79     | **0.82** |
| Legal Protections         | 0.73       | 0.77     | **0.80** |

### Stage 2 Human Evaluation Results

- **Committee vs Single-Stage**: 68% preference for committee analysis
- **Human Alignment Score**: 4.2/5.0 average rating
- **Inter-Annotator Agreement**: κ = 0.71 (substantial agreement)

### Example Output

**Stage 1 - Clause Extraction**:

```json
{
  "contract_id": "example_contract_001",
  "extracted_clauses": [
    {
      "text": "Vendor shall not engage in any business that competes...",
      "category": "competition_exclusivity",
      "confidence": 0.95,
      "start_position": 1234,
      "end_position": 1456
    }
  ]
}
```

**Stage 2 - Committee Analysis**:

```json
{
  "final_report": "## Summary of Key Clauses\n- Non-compete clause (Section 8.2)...",
  "legal_risks": ["Broad non-compete scope may be unenforceable..."],
  "recommendations": ["Consider narrowing geographic scope..."],
  "committee_consensus": 0.85
}
```

## 🔬 Research Applications

This system supports research in:

- **Legal AI**: Contract understanding and clause extraction
- **Multi-Agent Systems**: Collaborative reasoning for document analysis
- **Human Preference Learning**: Aligning AI outputs with human preferences
- **Parameter-Efficient Fine-Tuning**: LoRA for domain-specific adaptation

## 🐳 Docker Deployment

```bash
# Build the container
docker build -t legal-analysis .

# Run the web interface
docker run -p 7860:7860 legal-analysis
```

## 📈 Monitoring & Analysis

Generate comprehensive statistics and visualizations:

```bash
# Dataset statistics
python src/stage1/cuad_statistics.py

# Performance analysis
python improved_process_results.py

# Human evaluation analysis
python src/stage2/human_eval.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## 📚 Documentation

For detailed documentation on each component:

- [Stage 1 Documentation](src/stage1/README.md)
- [Stage 2 Documentation](src/stage2/README.md)
- [API Reference](docs/api_reference.md)

## 🙏 Acknowledgments

- **CUAD Dataset**: [Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad)
- **Base Models**: [Qwen3 Language Models](https://github.com/QwenLM/Qwen2)
- **Training Infrastructure**: [Modal Labs](https://modal.com/) for cloud training
- **Human Evaluation**: Web platform built with Flask

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the research team.
