#!/bin/bash

# ActiveScanLab Virtual Environment Setup Script
# Run this script once after cloning the repository

echo "==================================================================="
echo "Setting up ActiveScanLab Virtual Environment"
echo "==================================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==================================================================="
echo "✓ Virtual environment setup complete!"
echo "==================================================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To run experiments:"
echo "  python main_runner.py --mode full --config run1"
echo ""