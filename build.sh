#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate Prisma Client
python -m prisma generate

# Any other build steps (like downloading NLTK data or models if needed)
echo "Build process completed successfully!"
