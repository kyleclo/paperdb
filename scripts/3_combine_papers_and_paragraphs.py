import json
from pathlib import Path


def combine_papers_and_paragraphs():
    """Combine papers.jsonl with paper_id_to_paragraphs.jsonl."""

    papers_file = Path("data/raw/papers.jsonl")
    paragraphs_file = Path("data/raw/paper_id_to_paragraphs.jsonl")
    output_file = Path("data/raw/papers_with_passages.jsonl")

    print(f"Loading {papers_file}...")
    papers = {}
    with open(papers_file) as f:
        for line in f:
            paper = json.loads(line)
            papers[paper['corpusId']] = paper

    print(f"  Loaded {len(papers)} papers")

    print(f"Loading {paragraphs_file}...")
    paragraphs_by_id = {}
    with open(paragraphs_file) as f:
        for line in f:
            paper_paragraphs = json.loads(line)
            corpus_id = paper_paragraphs['corpusId']
            paragraphs_by_id[corpus_id] = paper_paragraphs['paragraphs']

    print(f"  Loaded paragraphs for {len(paragraphs_by_id)} papers")

    print(f"Combining datasets...")
    combined_count = 0
    papers_without_paragraphs = 0
    papers_without_metadata = 0

    with open(output_file, 'w') as f:
        # Iterate through papers from papers.jsonl (prioritize this)
        for corpus_id, paper in papers.items():
            # Add paragraphs if available
            if corpus_id in paragraphs_by_id:
                paper['paragraphs'] = paragraphs_by_id[corpus_id]
                paper['paragraphCount'] = len(paragraphs_by_id[corpus_id])
                combined_count += 1
            else:
                # No paragraphs found for this paper
                paper['paragraphs'] = []
                paper['paragraphCount'] = 0
                papers_without_paragraphs += 1

            f.write(json.dumps(paper) + '\n')

        # Check if there are papers in paragraphs that aren't in papers.jsonl
        for corpus_id in paragraphs_by_id:
            if corpus_id not in papers:
                papers_without_metadata += 1

    print(f"\n✓ Created {output_file}")
    print(f"  Papers with both metadata and paragraphs: {combined_count}")
    print(f"  Papers with metadata but no paragraphs: {papers_without_paragraphs}")
    print(f"  Papers with paragraphs but no metadata: {papers_without_metadata}")
    print(f"  Total papers in output: {len(papers)}")


if __name__ == "__main__":
    combine_papers_and_paragraphs()
