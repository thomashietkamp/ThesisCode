#!/usr/bin/env python3
"""
Setup script for Modal Labs DPO training.

This script helps you set up Modal and configure the necessary secrets.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def check_modal_installation():
    """Check if Modal is installed."""
    try:
        import modal
        print("✅ Modal is already installed")
        return True
    except ImportError:
        print("❌ Modal is not installed")
        return False


def install_modal():
    """Install Modal package."""
    print("Installing Modal...")
    run_command("pip install modal")
    print("✅ Modal installed successfully")


def setup_modal_token():
    """Set up Modal authentication token."""
    print("\n🔐 Setting up Modal authentication...")
    print("You need to create a Modal account and get your token.")
    print("1. Go to https://modal.com")
    print("2. Sign up or log in")
    print("3. Go to your settings and get your token")

    token = input(
        "\nEnter your Modal token (or press Enter to skip): ").strip()
    if token:
        # Set up Modal token
        result = run_command(f"modal token set {token}", check=False)
        if result.returncode == 0:
            print("✅ Modal token set successfully")
        else:
            print("❌ Failed to set Modal token")
            print("You can set it manually later with: modal token set YOUR_TOKEN")
    else:
        print("⏭️  Skipping token setup. You can set it later with: modal token set YOUR_TOKEN")


def setup_wandb_secret():
    """Set up Weights & Biases secret for logging."""
    print("\n📊 Setting up Weights & Biases (optional)...")
    print("If you want to use W&B for logging, you need to set up a secret.")

    use_wandb = input(
        "Do you want to set up W&B logging? (y/N): ").strip().lower()
    if use_wandb in ['y', 'yes']:
        wandb_key = input(
            "Enter your W&B API key (from https://wandb.ai/authorize): ").strip()
        if wandb_key:
            # Create W&B secret in Modal
            result = run_command(
                f'modal secret create wandb-secret WANDB_API_KEY="{wandb_key}"', check=False)
            if result.returncode == 0:
                print("✅ W&B secret created successfully")
            else:
                print("❌ Failed to create W&B secret")
                print(
                    "You can create it manually later with: modal secret create wandb-secret WANDB_API_KEY=your_key")
        else:
            print("⏭️  Skipping W&B setup")
    else:
        print("⏭️  Skipping W&B setup")


def check_dataset():
    """Check if the DPO dataset exists."""
    dataset_path = Path("src/stage2/data/dpo_training_data.jsonl")
    if dataset_path.exists():
        print(f"✅ DPO dataset found at {dataset_path}")
        print(
            f"   Dataset size: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    else:
        print(f"❌ DPO dataset not found at {dataset_path}")
        print("   Make sure you have generated the DPO training data first")
        return False


def create_modal_config():
    """Create a simple config file for Modal runs."""
    config_content = '''# Modal DPO Training Configuration

## Quick Start Commands:

# Basic training run (2 epochs, default settings):
modal run modal_dpo_runner.py

# Custom training run:
modal run modal_dpo_runner.py --epochs 3 --batch-size 12 --learning-rate 3e-5

# Training with W&B logging:
modal run modal_dpo_runner.py --use-wandb true --experiment-name "my-dpo-experiment"

# Different model:
modal run modal_dpo_runner.py --model-name "Qwen/Qwen2.5-14B"

## GPU Configuration:
- Uses 4x A100 80GB GPUs
- Each model gets its own GPU (coordinator, reasoner, validator)
- 4th GPU used for aggregation

## Cost Estimation:
- A100 80GB: ~$4/hour per GPU
- 4 GPUs: ~$16/hour
- 2 epochs typically take 4-8 hours
- Estimated cost: $64-128 per training run

## Checkpoints:
- Saved to Modal volume: /data/checkpoints/
- Includes final models + intermediate checkpoints
- Persistent across runs

## Monitoring:
- Check Modal dashboard for job status
- Use W&B for training metrics (if configured)
- Logs printed to Modal console
'''

    config_path = Path("modal_config.md")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"✅ Created configuration guide at {config_path}")


def main():
    """Main setup function."""
    print("🚀 Modal Labs DPO Training Setup")
    print("=" * 40)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(
        f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Check/install Modal
    if not check_modal_installation():
        install_modal()

    # Setup Modal authentication
    setup_modal_token()

    # Setup W&B secret
    setup_wandb_secret()

    # Check dataset
    dataset_exists = check_dataset()

    # Create config
    create_modal_config()

    print("\n" + "=" * 40)
    print("🎉 Setup Complete!")
    print("=" * 40)

    if dataset_exists:
        print("\n✅ You're ready to run DPO training!")
        print("\nNext steps:")
        print("1. Review the configuration in modal_config.md")
        print("2. Run your first training:")
        print("   modal run modal_dpo_runner.py")
        print("\n3. Monitor progress in the Modal dashboard")
    else:
        print("\n⚠️  Next steps:")
        print("1. Generate your DPO training dataset first")
        print("2. Then run: python setup_modal.py")
        print("3. Finally: modal run modal_dpo_runner.py")

    print(f"\n📖 For more details, see: modal_config.md")


if __name__ == "__main__":
    main()
