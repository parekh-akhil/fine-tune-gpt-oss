#!/bin/bash
# Quick start script for fine-tuning GPT OSS 20B

set -e

echo "======================================================================"
echo "Fine-Tune GPT OSS 20B - Quick Start"
echo "======================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "Installing requirements..."
echo "This may take several minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

# Run setup
echo ""
echo "Running setup verification..."
python setup.py

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Prepare your data: python scripts/prepare_data.py --help"
echo "  3. Start training: python scripts/finetune.py"
echo ""
echo "For more information, see README.md"
echo ""
