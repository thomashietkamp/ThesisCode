# Legal Clause Extraction with Gemma 3

This project provides a system for extracting and categorizing legal clauses from contracts using fine-tuned Gemma 3 models. It supports five specialized agents, each focusing on a different legal category:

1. **Intellectual Property & Licensing Agent**
2. **Competition & Exclusivity Agent**
3. **Termination & Control Rights Agent**
4. **Financial & Commercial Terms Agent**
5. **Legal Protections & Liability Agent**

## Project Structure

```
.
├── data/
│   ├── CUAD_v1/                   # Original CUAD dataset
│   │   ├── full_contract_pdf/     # PDF contracts
│   │   └── subcategories/         # CSV files with labeled clauses
│   └── clause_extraction/         # Generated training data
│       ├── intellectual_property_licensing/
│       ├── competition_exclusivity/
│       ├── termination_control/
│       ├── financial_commercial/
│       └── legal_protections_liability/
├── models/                        # Fine-tuned models will be saved here
├── src/
│   ├── data_processing/
│   │   └── prepare_clause_extraction_data.py
│   ├── training/
│   │   └── train_clause_extraction_models.py
│   └── inference/
│       └── clause_extraction_inference.py
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

To prepare the training data for the models:

```bash
python src/data_processing/prepare_clause_extraction_data.py
```

This script will:

- Process the CSV files in the `data/CUAD_v1/subcategories/` directory
- Extract clause texts and their labels
- Format the data for training
- Save the processed data as JSONL files in the `data/clause_extraction/` directory

## Training Models

To train the models:

```bash
python src/training/train_clause_extraction_models.py --category all
```

Options:

- `--category`: Specify which category to train ("all" or one of the five categories)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 4)
- `--eval_steps`: Steps between evaluations (default: 500)
- `--save_steps`: Steps between model saves (default: 1000)
- `--warmup_steps`: Warmup steps (default: 500)
- `--weight_decay`: Weight decay (default: 0.01)
- `--logging_steps`: Steps between logs (default: 100)

The trained models will be saved in the `models/` directory.

## Inference

To extract clauses from a contract:

```bash
python src/inference/clause_extraction_inference.py --file path/to/contract.txt --output results.json
```

Options:

- `--file`: Path to the input contract file (required)
- `--category`: Specific category to extract (optional, extracts all categories if not specified)
- `--output`: Path to the output JSON file (default: extracted_clauses.json)

The extracted clauses will be saved in the specified output file in JSON format.

## Example Output

```json
{
  "intellectual_property_licensing": [
    {
      "label": "License Grant",
      "clause_text": "The Licensee is granted a license to use the software for internal business..."
    },
    {
      "label": "IP Ownership Assignment",
      "clause_text": "All intellectual property created during the engagement shall be assigned to the Client..."
    }
  ],
  "competition_exclusivity": [
    {
      "label": "Non-Compete",
      "clause_text": "Vendor shall not engage in any business that competes with Client for a period of 2 years..."
    }
  ]
}
```

## License

[Specify the license under which this project is released]

## Acknowledgments

This project uses the [CUAD dataset](https://www.atticusprojectai.org/cuad) and [Google's Gemma 3 models](https://ai.google.dev/gemma).
