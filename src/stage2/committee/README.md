# Stage 2 Committee System

This directory contains the Stage 2 Joint Reasoning Committee that analyzes contract clauses using a multi-agent approach with Coordinator, Validator, and Reasoner roles.

## Overview

The committee system:

- Takes extracted clauses as input (from Stage 1)
- Generates exactly 9 variants per contract (3 personas × 3 temperatures)
- Uses a multi-step reasoning process: Draft → Validate → Final Report
- Supports both individual JSON files and concatenated format

## Files

- `committe.py` - Main committee script
- `prompts.yaml` - Configuration with prompts and generation parameters
- `llm_wrapper_openrouter.py` - OpenRouter API integration
- `test_committee.py` - Test script to verify functionality
- `test_with_sample.py` - Creates small sample datasets for testing

## Usage

### With concatenated_clauses.json (Recommended)

```bash
# Full dataset (57 contracts)
python committe.py --in ../data/stage1_out/concatenated_clauses.json --cfg prompts.yaml

# Create a small sample first (3 contracts)
python test_with_sample.py
python committe.py --in sample_contracts.json --cfg prompts.yaml --out ../data/stage2_out_sample
```

### With individual JSON files

```bash
python committe.py --in ../data/stage1_out/ --cfg prompts.yaml
```

## Configuration

The `prompts.yaml` file contains:

1. **Prompt templates** for each role:

   - `coordinator` (3 personas: neutral, risk_averse, aggressive)
   - `validator`
   - `reasoner`

2. **Generation parameters**:
   - `coord_T`: [0.2, 0.4, 0.7] (3 temperature values)
   - `persona`: [neutral, risk_averse, aggressive] (3 personas)
   - Model IDs for each component

## Output

Each contract generates a JSON file with 9 variants (numbered 0-8), containing:

- Configuration used
- Generated prompts
- Draft reports, validation feedback, final reports
- "Thinking" traces from the LLM

## Input Format

### Concatenated Format (concatenated_clauses.json)

```json
{
  "contract_id_1": ["clause1", "clause2", ...],
  "contract_id_2": ["clause1", "clause2", ...],
  ...
}
```

### Individual Files Format

```json
{
  "contract_metadata": {...},
  "clauses": ["clause1", "clause2", ...]
}
```

## Testing

```bash
# Test basic functionality
python test_committee.py

# Create small sample for testing
python test_with_sample.py
```

## Requirements

- OpenRouter API key (set as `OPENROUTER_API_KEY` environment variable)
- Python packages: `pyyaml`, `tqdm`, `requests`, `python-dotenv`

## API Costs

Each contract processes 9 variants × 3 LLM calls = 27 API calls. With 57 contracts, that's ~1,539 API calls total. Consider testing with the sample first.
