#!/usr/bin/env python3
"""
Convert HuggingFace dataset to JSONLines format.
Takes the full text dataset and writes it as a flat .jsonl file.
"""

import json
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from tqdm import tqdm


def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    print("=" * 80)
    print("CONVERTING DATASET TO JSONLINES")
    print("=" * 80)
    print()

    # Paths
    input_path = 'raw/dblp-nlp-ml-ai-oa-recent-with-fulltext'
    output_path = Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext.jsonl')

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"✓ Loaded {len(dataset):,} papers\n")

    # Show columns
    print("Dataset columns:")
    for col in dataset.column_names:
        print(f"  - {col}")
    print()

    # Convert to JSONLines
    print(f"Converting to JSONLines...")
    print(f"  Output: {output_path}")
    print()

    with open(output_path, 'w') as f:
        for record in tqdm(dataset, desc="Writing records"):
            # Convert record to dict and write as JSON line
            json_line = json.dumps(record, ensure_ascii=False, default=json_serializer)
            f.write(json_line + '\n')

    print(f"\n✓ Conversion complete!")
    print(f"  Total records: {len(dataset):,}")
    print(f"  Output file: {output_path}")

    # Show file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == '__main__':
    main()
