#!/usr/bin/env python3
"""
Simple wrapper to run DPO model testing locally with default parameters.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run DPO model testing")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="./checkpoints/final",
                        help="Path to model checkpoint directory")
    parser.add_argument("--dataset-path", type=str,
                        default="dpo_training_data.jsonl",
                        help="Path to DPO training dataset")
    parser.add_argument("--test-samples", type=int, default=10,
                        help="Number of samples to test")
    parser.add_argument("--output-file", type=str, default="dpo_test_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Check if files exist
    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_path = Path(args.dataset_path)

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("Available directories:")
        for p in Path(".").glob("**/checkpoints"):
            print(f"  {p}")
        return 1

    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        # Try to find it
        for p in Path(".").rglob("dpo_training_data.jsonl"):
            print(f"Found dataset at: {p}")
            args.dataset_path = str(p)
            break
        else:
            return 1

    # Run the test script
    cmd = [
        sys.executable, "test_dpo_models.py",
        "--checkpoint-dir", args.checkpoint_dir,
        "--dataset-path", args.dataset_path,
        "--test-samples", str(args.test_samples),
        "--output-file", args.output_file
    ]

    print(f"🚀 Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Testing completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Testing failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        return 1


if __name__ == "__main__":
    exit(main())
