#!/usr/bin/env python3
"""
Test script to verify that the generated DPO data is in the correct format
for our DPO training script.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader


class DPODataset(Dataset):
    """Dataset class for DPO training data"""

    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dpo_collate_fn(batch):
    """
    Collate function for DPO training data.
    This should match the collator in our DPO training script.
    """
    # Extract all fields from the batch
    prompts = [item['prompt'] for item in batch]

    # Winner trajectory (preferred)
    coord_think_plus = [item['coord_think_plus'] for item in batch]
    coord_out_plus = [item['coord_out_plus'] for item in batch]
    reason_think_plus = [item['reason_think_plus'] for item in batch]
    reason_out_plus = [item['reason_out_plus'] for item in batch]
    valid_think_plus = [item['valid_think_plus'] for item in batch]
    valid_out_plus = [item['valid_out_plus'] for item in batch]

    # Loser trajectory (rejected)
    coord_think_minus = [item['coord_think_minus'] for item in batch]
    coord_out_minus = [item['coord_out_minus'] for item in batch]
    reason_think_minus = [item['reason_think_minus'] for item in batch]
    reason_out_minus = [item['reason_out_minus'] for item in batch]
    valid_think_minus = [item['valid_think_minus'] for item in batch]
    valid_out_minus = [item['valid_out_minus'] for item in batch]

    return {
        'prompt': prompts,
        'coord_think_plus': coord_think_plus,
        'coord_out_plus': coord_out_plus,
        'reason_think_plus': reason_think_plus,
        'reason_out_plus': reason_out_plus,
        'valid_think_plus': valid_think_plus,
        'valid_out_plus': valid_out_plus,
        'coord_think_minus': coord_think_minus,
        'coord_out_minus': coord_out_minus,
        'reason_think_minus': reason_think_minus,
        'reason_out_minus': reason_out_minus,
        'valid_think_minus': valid_think_minus,
        'valid_out_minus': valid_out_minus,
    }


def test_data_format():
    """Test that the data format is correct"""

    # Load dataset
    print("Loading DPO dataset...")
    dataset = DPODataset('src/stage2/data/dpo_training_data.jsonl')
    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=dpo_collate_fn
    )

    # Test first batch
    print("\nTesting first batch...")
    batch = next(iter(dataloader))

    # Check that all required fields are present
    required_fields = [
        'prompt', 'coord_think_plus', 'coord_out_plus', 'reason_think_plus',
        'reason_out_plus', 'valid_think_plus', 'valid_out_plus',
        'coord_think_minus', 'coord_out_minus', 'reason_think_minus',
        'reason_out_minus', 'valid_think_minus', 'valid_out_minus'
    ]

    print("✅ Checking required fields...")
    for field in required_fields:
        assert field in batch, f"Missing field: {field}"
        assert len(
            batch[field]) == 4, f"Wrong batch size for {field}: {len(batch[field])}"
        print(f"  ✅ {field}: {len(batch[field])} items")

    # Check data quality
    print("\n✅ Checking data quality...")
    for i in range(4):
        # Check that prompts are not empty
        assert len(batch['prompt'][i].strip()
                   ) > 0, f"Empty prompt at index {i}"

        # Check that trajectories are not empty
        for field in required_fields[1:]:  # Skip prompt
            assert len(batch[field][i].strip()
                       ) > 0, f"Empty {field} at index {i}"

    print("  ✅ All prompts and trajectories are non-empty")

    # Sample data preview
    print("\n📊 Sample data preview:")
    print(f"Prompt length: {len(batch['prompt'][0])}")
    print(f"Coord think plus length: {len(batch['coord_think_plus'][0])}")
    print(f"Valid out plus length: {len(batch['valid_out_plus'][0])}")

    print(f"\nFirst prompt (first 200 chars):")
    print(f"{batch['prompt'][0][:200]}...")

    print("\n✅ All tests passed! Data format is correct for DPO training.")

    return dataset, dataloader


def analyze_dataset():
    """Analyze the dataset statistics"""
    print("\n📊 Dataset Analysis:")

    dataset = DPODataset('src/stage2/data/dpo_training_data.jsonl')

    # Collect statistics
    prompt_lengths = []
    trajectory_lengths = {
        'coord_think_plus': [], 'coord_out_plus': [], 'reason_think_plus': [],
        'reason_out_plus': [], 'valid_think_plus': [], 'valid_out_plus': [],
        'coord_think_minus': [], 'coord_out_minus': [], 'reason_think_minus': [],
        'reason_out_minus': [], 'valid_think_minus': [], 'valid_out_minus': []
    }

    for item in dataset:
        prompt_lengths.append(len(item['prompt']))
        for key in trajectory_lengths.keys():
            trajectory_lengths[key].append(len(item[key]))

    # Print statistics
    print(f"Number of examples: {len(dataset)}")
    print(
        f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.0f} chars")
    print(f"Min prompt length: {min(prompt_lengths)} chars")
    print(f"Max prompt length: {max(prompt_lengths)} chars")

    print("\nTrajectory lengths (characters):")
    for key, lengths in trajectory_lengths.items():
        avg_len = sum(lengths) / len(lengths)
        print(f"  {key}: avg={avg_len:.0f}, min={min(lengths)}, max={max(lengths)}")

    # Count unique contracts
    contracts = set(item['contract_id'] for item in dataset)
    print(f"\nUnique contracts: {len(contracts)}")

    # Count winners
    winners = {}
    for item in dataset:
        winner = item['winner_key']
        winners[winner] = winners.get(winner, 0) + 1

    print(f"Winner distribution: {winners}")


if __name__ == "__main__":
    # Test data format
    dataset, dataloader = test_data_format()

    # Analyze dataset
    analyze_dataset()
