import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import clean_query


def create_title_queries(input_path, output_dir, seed=42):
    """
    Create synthetic query-document pairs where query = lowercased paper title.

    Args:
        input_path: Path to papers JSONL file
        output_dir: Directory to save train.jsonl
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

    # Create train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for paper in papers:
            query = clean_query(paper['title'])
            entry = {
                'query': query,
                'corpus_id': paper['corpusid'],
                'relevance': 1
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Created {train_path} with {len(papers)} queries")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_data.py <input_papers.jsonl> [output_dir]")
        print("\nExample:")
        print("  python create_data.py raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl data/synth/title_as_query")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/synth/title_as_query"

    create_title_queries(input_path, output_dir)
