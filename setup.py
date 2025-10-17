#!/usr/bin/env python3
"""
Setup script for the fine-tune-gpt-oss project.
Run this script to verify your environment and setup.
"""

import sys
import subprocess
import os


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available - training will be very slow on CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet - will check after installation")
        return None


def create_directories():
    """Create necessary directories."""
    dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Created necessary directories")


def verify_config():
    """Verify configuration file exists."""
    if os.path.exists("config/training_config.yaml"):
        print("✓ Configuration file found")
        return True
    else:
        print("❌ Configuration file missing")
        return False


def main():
    print("=" * 80)
    print("Fine-Tune GPT OSS 20B Setup")
    print("=" * 80)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify config
    verify_config()
    
    # Check CUDA
    check_cuda()
    
    print()
    print("=" * 80)
    print("Setup completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Prepare your training data: python scripts/prepare_data.py --help")
    print("2. Configure training: edit config/training_config.yaml")
    print("3. Start training: python scripts/finetune.py")
    print()


if __name__ == "__main__":
    main()
