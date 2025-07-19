#!/usr/bin/env python3
"""
Optimization script for Replit deployment
Run this before deploying to Replit for better performance
"""

import os
import sys
from pathlib import Path


def optimize_for_replit():
    """Apply optimizations for Replit deployment"""

    print("🚀 Optimizing for Replit deployment...")

    # Set environment variables for production
    env_vars = {
        'DEBUG': 'False',
        'LOG_LEVEL': 'WARNING',
        'PYTHONOPTIMIZE': '1',
        'PYTHONDONTWRITEBYTECODE': '1'
    }

    # Create a simple .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# Performance optimized for Replit\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        print("✅ Created optimized .env file")

    # Create Replit configuration
    replit_config = {
        'run': 'python app.py',
        'language': 'python3',
        'onBoot': 'pip install -r requirements.txt'
    }

    # Check if we're in a Replit environment
    if 'REPL_ID' in os.environ:
        print("🔧 Detected Replit environment")

        # Set environment variables directly
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"   Set {key}={value}")

    # Optimize Python files (remove __pycache__ and .pyc files)
    for cache_dir in Path('.').rglob('__pycache__'):
        try:
            for pyc_file in cache_dir.rglob('*.pyc'):
                pyc_file.unlink()
            cache_dir.rmdir()
            print(f"   Cleaned {cache_dir}")
        except:
            pass

    print("✅ Optimization complete!")
    print("\n📋 Performance Tips for Replit:")
    print("   • Set DEBUG=False in environment variables")
    print("   • Use the 'Boost' option for better performance")
    print("   • Monitor memory usage in the Replit console")
    print("   • Consider upgrading to Replit Core for better resources")


if __name__ == "__main__":
    optimize_for_replit()
