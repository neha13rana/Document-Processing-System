#!/bin/bash

# Exit on error
set -e

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system dependencies for OpenCV and Tesseract
sudo apt-get update
sudo apt-get install -y libgl1 tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete. Activate your environment with: source venv/bin/activate"