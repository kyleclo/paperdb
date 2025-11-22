#!/bin/bash

# Script to generate synthetic query datasets with different difficulty levels
# Creates train_easy.jsonl and train_hard.jsonl from papers_100.jsonl

set -e  # Exit on error

echo "=================================================="
echo "Generating synthetic query datasets"
echo "=================================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Run the dataset creation script
python data/synth/metadata_as_query/create_data_with_difficulty.py

echo ""
echo "=================================================="
echo "Done! Generated files:"
echo "  - data/synth/metadata_as_query/train_td0.0_md0.0.jsonl (easy)"
echo "  - data/synth/metadata_as_query/train_td0.4_md0.3.jsonl (medium)"
echo "  - data/synth/metadata_as_query/train_td0.7_md0.3.jsonl (hard)"
echo "  - data/synth/metadata_as_query/train_td0.4_md0.7.jsonl (hardest)"
echo "  - data/synth/metadata_as_query/train_td0.7_md0.7.jsonl (extreme)"
echo "=================================================="
