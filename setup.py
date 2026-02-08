#!/usr/bin/env python3
"""Setup script for the RL continuous control project."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up RL Continuous Control Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["checkpoints", "logs", "assets", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run basic tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Some tests failed, but setup can continue")
    
    # Quick training test
    if not run_command("python scripts/train_simple.py --episodes 5 --algorithm ddpg", "Quick training test"):
        print("Quick training test failed, but setup can continue")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Train an agent: python -m src.train.train --env Pendulum-v1 --algorithm ddpg")
    print("2. Evaluate: python -m src.eval.eval --env Pendulum-v1 --algorithm ddpg --model checkpoints/best_model.pth")
    print("3. Launch demo: streamlit run demo/app.py")
    print("\nRemember: This is for research/educational purposes only!")


if __name__ == "__main__":
    main()
