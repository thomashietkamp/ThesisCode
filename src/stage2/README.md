# Stage 2: Multi-Agent Analysis and Human Preference Learning

Stage 2 builds upon the foundation models from Stage 1 to implement advanced multi-agent contract analysis and human preference learning. This stage focuses on generating comprehensive legal reports through committee-based reasoning and aligning model outputs with human preferences using Direct Preference Optimization (DPO).

## Overview

Stage 2 introduces several key innovations:

1. **Multi-Agent Committee System** - Coordinator, Validator, and Reasoner agents collaborate to produce comprehensive legal analysis
2. **Direct Preference Optimization (DPO)** - Train models based on human preference data
3. **Human Evaluation Platform** - Web-based system for collecting human judgments
4. **Statistical Analysis** - Comprehensive analysis of model performance and human preferences

## Directory Structure

```
stage2/
├── committee/                 # Multi-agent committee system
│   ├── committe.py           # Main committee orchestration
│   ├── prompts.yaml          # Agent prompts and configurations
│   ├── generate_test_single.py  # Single-contract processing
│   └── llm_wrapper_openrouter.py  # LLM API integration
├── dpo/                      # Direct Preference Optimization
│   ├── dpo.py               # DPO training implementation
│   ├── modal_dpo_runner.py  # Cloud-based DPO training
│   ├── inference_dpo_models.py  # DPO model inference
│   └── dpo_diagnostics.py   # DPO analysis and diagnostics
├── data/                    # Data processing and outputs
│   ├── create_dpo_dataset.py   # DPO dataset creation
│   ├── stage1_out/          # Stage 1 outputs
│   ├── stage2_out/          # Committee analysis results
│   └── test_reports/        # Test analysis outputs
├── voting/                  # Human preference analysis
│   ├── ai_voter.py          # Automated voting simulation
│   └── calculate_kappa.py   # Inter-annotator agreement
├── website_vote/            # Human evaluation web interface
│   ├── app.py              # Flask web application
│   ├── templates/          # HTML templates
│   ├── reset_database.py   # Database management
│   └── optimize_for_replit.py  # Deployment optimization
├── human_eval.py           # Human evaluation analysis
└── preference_analysis.py  # Preference pattern analysis
```

## Key Components

### Multi-Agent Committee System (`committee/`)

The committee system implements a sophisticated multi-agent approach to contract analysis:

#### Agent Roles

- **Coordinator**: Provides initial analysis and summarization
- **Validator**: Reviews and validates the coordinator's analysis
- **Reasoner**: Synthesizes findings into final comprehensive report

#### Running Committee Analysis

```bash
# Full dataset analysis
python committee/committe.py \
    --in data/stage1_out/concatenated_clauses.json \
    --cfg committee/prompts.yaml \
    --out data/stage2_out/

# Single contract analysis
python committee/generate_test_single.py \
    --contract_id example_contract \
    --output_dir data/test_reports/
```

#### Configuration (`prompts.yaml`)

```yaml
coordinator:
  neutral: "You are a neutral legal analyst..."
  risk_averse: "You are a cautious legal analyst..."
  aggressive: "You are a thorough legal analyst..."

validator:
  prompt: "Review the analysis for accuracy..."

reasoner:
  prompt: "Synthesize the findings into a final report..."

generation:
  coord_T: [0.2, 0.4, 0.7] # Temperature variations
  persona: [neutral, risk_averse, aggressive]
```

### Direct Preference Optimization (`dpo/`)

DPO training aligns models with human preferences by learning from comparative feedback.

#### Creating DPO Dataset

```bash
python data/create_dpo_dataset.py \
    --stage2_dir data/stage2_out/ \
    --voting_data voting_results.csv \
    --output_file dpo/dpo_training_data.jsonl
```

#### Training DPO Models

```bash
# Local training
python dpo/dpo.py \
    --dataset_path dpo/dpo_training_data.jsonl \
    --base_model Qwen/Qwen3-8B \
    --output_dir checkpoints/dpo_8b/

# Cloud training on Modal
modal run dpo/modal_dpo_runner.py \
    --dataset-path dpo/dpo_training_data.jsonl
```

#### DPO Model Inference

```bash
python dpo/inference_dpo_models.py \
    --checkpoint_dir checkpoints/dpo_8b/ \
    --input_file test_contracts.json \
    --output_dir dpo_results/
```

### Human Evaluation Platform (`website_vote/`)

A comprehensive web-based platform for collecting human preferences and judgments.

#### Features

- **Pairwise Comparison**: Compare outputs from different models/configurations
- **Three-way Voting**: Compare committee vs. single-stage vs. stage2 outputs
- **Multi-dimensional Rating**: Rate on clarity, legal soundness, reasoning depth, human alignment
- **Progress Tracking**: Monitor evaluation progress and statistics
- **Admin Dashboard**: View voting statistics and export data

#### Running the Web Interface

```bash
cd website_vote/
python app.py
```

#### Database Management

```bash
# Reset database for new evaluation round
python reset_database.py

# Backup current data
python reset_database.py --backup-only
```

#### Configuration for Different Environments

```bash
# Local development (SQLite)
export DATABASE_URL=""
python app.py

# Production with PostgreSQL
export DATABASE_URL="postgresql://user:pass@host:port/db"
python app.py

# Replit optimization
python optimize_for_replit.py
```

### Analysis and Evaluation

#### Human Evaluation Analysis

```bash
python human_eval.py
```

Generates comprehensive statistical analysis including:

- Preference distributions across models
- Inter-annotator agreement (Cohen's Kappa)
- Performance rankings with confidence intervals
- Correlation analysis between different rating dimensions

#### Preference Pattern Analysis

```bash
python preference_analysis.py \
    --csv_path voting_results.csv \
    --json_dir data/stage2_out/
```

Analyzes voting patterns to understand:

- Agent type preferences (human vs. AI voters)
- Model complexity preferences
- Persona-based voting patterns
- Length bias in human judgments

#### Automated Voting Simulation

```bash
python voting/ai_voter.py \
    --model_id gemini-pro \
    --contracts_dir data/stage2_out/ \
    --output_csv ai_voting_results.csv
```

## Data Formats

### Committee Output Format

```json
{
  "0": {
    "config": {
      "persona": "neutral",
      "coord_T": 0.2,
      "variant_id": 0
    },
    "coordinator_draft": "Initial analysis...",
    "validator_feedback": "Validation comments...",
    "final_report": "Comprehensive legal analysis...",
    "reasoning_trace": "Step-by-step thinking..."
  }
}
```

### DPO Training Data Format

```json
{
  "prompt": "Analyze this contract...",
  "chosen": "High-quality legal analysis...",
  "rejected": "Lower-quality analysis...",
  "contract_id": "example_contract",
  "metadata": {
    "winner_source": "committee",
    "loser_source": "single_stage"
  }
}
```

### Voting Data Format

```json
{
  "user_session": "session_123",
  "contract_id": "example_contract",
  "option1_source": "committee",
  "option2_source": "single_stage",
  "winner_source": "committee",
  "clarity_rating": 4,
  "legal_soundness_rating": 5,
  "reasoning_depth_rating": 4,
  "human_alignment_rating": 4,
  "voted_at": "2024-01-15T10:30:00Z"
}
```

## Performance Metrics

### Committee System Performance

- **Consensus Score**: Agreement level between agent roles
- **Coverage**: Completeness of legal issue identification
- **Accuracy**: Correctness of legal interpretations
- **Human Alignment**: Correlation with human preferences

### DPO Training Metrics

- **Preference Accuracy**: Ability to predict human preferences
- **Reward Model Score**: Learned preference alignment
- **KL Divergence**: Distance from base model
- **Win Rate**: Performance in head-to-head comparisons

### Human Evaluation Metrics

- **Inter-Annotator Agreement**: Cohen's Kappa scores
- **Rating Consistency**: Variance in multi-dimensional ratings
- **Preference Stability**: Consistency of choices over time
- **Coverage**: Percentage of contract types evaluated

## Advanced Features

### Multi-Persona Analysis

The committee system generates multiple perspectives using different personas:

- **Neutral**: Balanced, objective analysis
- **Risk Averse**: Conservative, cautious interpretation
- **Aggressive**: Thorough, detailed examination

### Temperature Variation

Different temperature settings provide diversity in analysis:

- **Low (0.2)**: Focused, consistent outputs
- **Medium (0.4)**: Balanced creativity and consistency
- **High (0.7)**: Creative, diverse perspectives

### Cloud Deployment

```bash
# Deploy on Modal Labs
modal deploy dpo/modal_dpo_runner.py

# Deploy web interface on Heroku
git push heroku main
```

## Best Practices

### Committee Configuration

1. **Persona Balance**: Use all three personas for comprehensive coverage
2. **Temperature Tuning**: Adjust based on desired output diversity
3. **Prompt Engineering**: Customize prompts for specific legal domains
4. **Quality Control**: Validate outputs before using for DPO training

### DPO Training

1. **Data Quality**: Ensure high-quality preference pairs
2. **Base Model Selection**: Choose appropriate foundation model
3. **Hyperparameter Tuning**: Optimize learning rate and regularization
4. **Evaluation**: Regular validation on held-out preference data

### Human Evaluation

1. **Instruction Clarity**: Provide clear evaluation guidelines
2. **Bias Mitigation**: Randomize presentation order
3. **Quality Assurance**: Include attention checks
4. **Statistical Power**: Collect sufficient data for reliable conclusions

## Troubleshooting

### Committee System Issues

- **API Rate Limits**: Implement exponential backoff
- **Inconsistent Outputs**: Adjust temperature or prompts
- **Memory Issues**: Process contracts in batches

### DPO Training Issues

- **Convergence Problems**: Reduce learning rate or add regularization
- **Data Imbalance**: Ensure balanced preference pairs
- **Overfitting**: Use early stopping and validation monitoring

### Web Interface Issues

- **Database Connections**: Check DATABASE_URL configuration
- **Session Management**: Clear cookies and restart session
- **Performance**: Optimize database queries and caching

For detailed troubleshooting guides, see the README files in each subdirectory.
