#!/usr/bin/env python3
"""
Upload dataset to HuggingFace Hub.
Requires: huggingface-cli login (run this first to configure credentials)
"""

import json
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo
from datasets import Dataset


def upload_dataset(repo_id: str, private: bool = False):
    """
    Upload the final dataset and mappings to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        private: Whether to make the repository private
    """
    print("=" * 80)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("=" * 80)
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()

    # Check if logged in
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print("✗ Not logged in to HuggingFace!")
        print("\nPlease run: huggingface-cli login")
        print("Then re-run this script.")
        return

    # Files to upload
    files = {
        'dataset': Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl'),
        'author_map': Path('raw/author_affiliation_map.json'),
        'venue_map': Path('raw/venue_to_id.json'),
        'venue_stats': Path('raw/venue_stats.json'),
        'readme': Path('raw/README.md')
    }

    # Verify files exist
    print("\nVerifying files...")
    for name, path in files.items():
        if not path.exists():
            print(f"✗ Missing file: {path}")
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {path.name} ({size_mb:.1f} MB)")

    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return

    # Upload files
    print("\nUploading files...")

    # Upload main dataset
    print("  Uploading dataset (this may take a while)...")
    try:
        upload_file(
            path_or_fileobj=str(files['dataset']),
            path_in_repo=files['dataset'].name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"  ✓ Uploaded {files['dataset'].name}")
    except Exception as e:
        print(f"  ✗ Error uploading dataset: {e}")
        return

    # Upload mappings
    for name in ['author_map', 'venue_map', 'venue_stats']:
        print(f"  Uploading {files[name].name}...")
        try:
            upload_file(
                path_or_fileobj=str(files[name]),
                path_in_repo=files[name].name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"  ✓ Uploaded {files[name].name}")
        except Exception as e:
            print(f"  ✗ Error uploading {name}: {e}")
            return

    # Upload README
    print(f"  Uploading README...")
    try:
        upload_file(
            path_or_fileobj=str(files['readme']),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"  ✓ Uploaded README.md")
    except Exception as e:
        print(f"  ✗ Error uploading README: {e}")
        return

    # Create dataset card metadata
    print("\nCreating dataset card...")
    dataset_card = f"""---
language:
- en
license: cc-by-4.0
task_categories:
- text-retrieval
- question-answering
size_categories:
- 10K<n<100K
---

# DBLP Papers with Full Text + Enrichments

This dataset contains 22,371 recent open access NLP/ML/AI papers from DBLP, enriched with:
- Full text from Vespa API (96.8% coverage)
- Author affiliations from OpenAlex (13,830 authors)
- Venue normalization (workshops mapped to main conferences)
- Venue type tags (main/workshop)

## Files

- `{files['dataset'].name}` - Main dataset (589 MB)
- `{files['author_map'].name}` - Author affiliation mappings (4.4 MB)
- `{files['venue_map'].name}` - Venue normalization mappings (38 KB)
- `{files['venue_stats'].name}` - Venue statistics (54 KB)

## Quick Start

```python
import json

# Load papers
papers = []
with open('{files['dataset'].name}', 'r') as f:
    for line in f:
        papers.append(json.loads(line))

# Filter by venue type
main_papers = [p for p in papers if p['venue_type'] == 'main']
print(f"Main conference papers: {{len(main_papers):,}}")
```

See README.md for full documentation.
"""

    try:
        with open('dataset_card.md', 'w') as f:
            f.write(dataset_card)

        upload_file(
            path_or_fileobj='dataset_card.md',
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✓ Uploaded dataset card")
    except Exception as e:
        print(f"Note: Could not upload dataset card: {e}")

    print("\n" + "=" * 80)
    print("✓ UPLOAD COMPLETE!")
    print("=" * 80)
    print(f"\nView your dataset at:")
    print(f"https://huggingface.co/datasets/{repo_id}")


def main():
    import sys

    print("=" * 80)
    print("HUGGINGFACE DATASET UPLOADER")
    print("=" * 80)
    print()

    # Check if logged in
    print("Step 1: Login to HuggingFace")
    print("If you haven't logged in yet, run:")
    print("  huggingface-cli login")
    print()

    # Get repository info
    if len(sys.argv) < 2:
        print("Usage: python 8_upload_to_huggingface.py <repo_id> [--private]")
        print()
        print("Examples:")
        print("  python 8_upload_to_huggingface.py username/dblp-papers-fulltext")
        print("  python 8_upload_to_huggingface.py username/my-dataset --private")
        print()
        return

    repo_id = sys.argv[1]
    private = '--private' in sys.argv

    # Confirm upload
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()
    response = input("Continue with upload? (y/n): ")

    if response.lower() != 'y':
        print("Upload cancelled.")
        return

    # Upload
    upload_dataset(repo_id, private)


if __name__ == '__main__':
    main()
