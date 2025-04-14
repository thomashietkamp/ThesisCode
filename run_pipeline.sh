#!/bin/bash
set -e

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting contract extraction model fine-tuning pipeline${NC}"

# Create required directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p data/CUAD/{part1,part2,part3} data/processed models/gemma-3-4b-contract-extractor logs

# Create and activate virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv .venv
fi

# Determine the activate script based on OS
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install --upgrade pip
pip install torch transformers datasets accelerate bitsandbytes peft evaluate tqdm pandas numpy sentencepiece

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())"; then
    echo -e "${GREEN}CUDA is available.${NC}"
else
    echo -e "${YELLOW}CUDA is not available. Training will be slower.${NC}"
fi

# Prepare the dataset
echo -e "${YELLOW}Preparing the dataset...${NC}"
python src/data_processing/prepare_dataset.py 2>&1 | tee logs/data_preparation.log

# Check if dataset preparation was successful
if [ -f "data/processed/train.json" ] && [ -f "data/processed/eval.json" ]; then
    echo -e "${GREEN}Dataset preparation completed successfully.${NC}"
else
    echo -e "${RED}Dataset preparation failed. Check logs/data_preparation.log for details.${NC}"
    exit 1
fi

# Train the model
echo -e "${YELLOW}Starting model training...${NC}"
python src/model/train.py 2>&1 | tee logs/training.log

# Check if model training was successful
if [ -d "models/gemma-3-4b-contract-extractor" ]; then
    echo -e "${GREEN}Model training completed successfully.${NC}"
else
    echo -e "${RED}Model training failed. Check logs/training.log for details.${NC}"
    exit 1
fi

# Run inference on the test set
echo -e "${YELLOW}Running inference on test set...${NC}"
python src/model/inference.py --dir data/CUAD/part3 --output results/test_results.json 2>&1 | tee logs/inference.log

echo -e "${GREEN}Pipeline completed!${NC}"
echo -e "${GREEN}Results saved to results/test_results.json${NC}" 