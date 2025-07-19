#!/usr/bin/env python3
"""
run_test_demo.py - Demo script for running the end-to-end contract processing

This script demonstrates how to use generate_test_single.py to process test contracts.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run a demo of the end-to-end contract processing."""

    # Path to the main script
    script_path = Path(__file__).parent / "generate_test_single.py"

    print("Running End-to-End Contract Processing Demo")
    print("Using real CUAD contract data for direct analysis")
    print("With intelligent skipping of already processed contracts")
    print("=" * 50)
    print()

    # Run with a limit of 3 contracts for demo purposes
    cmd = [
        sys.executable, str(script_path),
        "--limit", "3",
        "--model", "qwen/qwen3-32b:free"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return 1

    print("\nDemo completed successfully!")
    print("Check the output directory for generated reports.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
