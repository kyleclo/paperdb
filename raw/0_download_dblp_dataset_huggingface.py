#!/usr/bin/env python3
"""Download DBLP discovery dataset from HuggingFace to raw/ directory."""

from datasets import load_dataset
import os
import sys
from datetime import datetime

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

log("Starting download of DBLP discovery dataset from HuggingFace...")
log("Dataset: jpwahle/dblp-discovery-dataset")

try:
    log("Loading dataset... (this will download files)")
    # Load the dataset with streaming to see progress
    dataset = load_dataset("jpwahle/dblp-discovery-dataset")

    log("Dataset loaded successfully!")
    log(f"Splits found: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        log(f"  {split_name}: {len(split_data)} examples")

    # Save to raw/ directory
    output_dir = "raw/dblp-discovery-dataset"
    os.makedirs(output_dir, exist_ok=True)

    log(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)

    log("SUCCESS! Dataset downloaded and saved")
    log(f"Location: {output_dir}")

except Exception as e:
    log(f"ERROR: {str(e)}")
    sys.exit(1)
