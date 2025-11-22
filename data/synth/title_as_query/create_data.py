import json
import random
import sys
from pathlib import Path


def create_title_queries(input_path, output_dir, train_ratio=0.8, seed=42):
    """
    Create synthetic query-document pairs where query = lowercased paper title.

    Args:
        input_path: Path to papers JSONL file
        output_dir: Directory to save train.jsonl and test.jsonl
        train_ratio: Fraction of data to use for training (default 0.8)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading papers from {input_path}...")

    # Read all papers
    papers = []
    with open(input_path) as f:
        for line in f:
            papers.append(json.loads(line))

    print(f"  Found {len(papers)} papers")

    # Shuffle papers with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(papers)

    # Split into train and test
    train_size = int(len(papers) * train_ratio)
    train_papers = papers[:train_size]
    test_papers = papers[train_size:]

    print(f"  Train: {len(train_papers)} papers")
    print(f"  Test: {len(test_papers)} papers")

    # Create train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for paper in train_papers:
            query = paper['title'].lower()
            entry = {
                'query': query,
                'paperId': paper['paperId'],
                'relevance': 1
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Created {train_path}")

    # Create test.jsonl
    test_path = output_dir / "test.jsonl"
    with open(test_path, 'w') as f:
        for paper in test_papers:
            query = paper['title'].lower()
            entry = {
                'query': query,
                'paperId': paper['paperId'],
                'relevance': 1
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Created {test_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_data.py <input_papers.jsonl> [output_dir] [train_ratio]")
        print("\nExample:")
        print("  python create_data.py data/raw/papers_100.jsonl data/synth 0.8")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/synth"
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    create_title_queries(input_path, output_dir, train_ratio)
