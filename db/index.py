import json
from pathlib import Path
from retrieval.dense import DenseRetriever
from sql.query import SQLRetriever


def load_papers(papers_path: str) -> list[dict]:
    papers = []
    with open(papers_path) as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def main():
    papers_path = Path(__file__).parent.parent / "data" / "raw" / "papers.jsonl"
    papers = load_papers(papers_path)

    print(f"Loaded {len(papers)} papers")

    dense = DenseRetriever()
    dense.index(papers)
    print("✓ Dense retrieval indexed")

    sql = SQLRetriever()
    sql.index(papers)
    print("✓ SQL database indexed")


if __name__ == "__main__":
    main()
