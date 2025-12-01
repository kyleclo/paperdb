#!/bin/bash

# Script to generate synthetic query datasets with parameter sweep
# Sweeps over title_dropout and metadata_dropout values: 0.0, 0.2, 0.4, 0.6, 0.8
# Creates 25 train files (5x5 grid)

set -e  # Exit on error

echo "=================================================="
echo "Generating synthetic query datasets (parameter sweep)"
echo "=================================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Run the dataset creation script
python data/synth/metadata_as_query/create_data_with_difficulty.py

echo ""
echo "=================================================="
echo "Done! Generated 25 train files in:"
echo "  data/synth/metadata_as_query/train_td*_md*.jsonl"
echo ""
echo "Dropout values: 0.0, 0.2, 0.4, 0.6, 0.8"
echo "Format: train_td{title_dropout}_md{metadata_dropout}.jsonl"
echo "=================================================="
