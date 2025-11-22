import csv
import json
from pathlib import Path
from collections import defaultdict


def organize_paragraphs():
    """Group paragraphs by corpusId and save to JSONL."""

    csv_file = Path("data/raw/paragraphs.csv")
    output_file = Path("data/raw/paper_id_to_paragraphs.jsonl")

    print(f"Reading {csv_file}...")

    # Group paragraphs by corpusId
    papers = defaultdict(list)

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus_id = row['corpusId']

            # Create paragraph object with all fields except corpusId
            paragraph = {
                'paragraphId': row['paragraphId'],
                'title': row['title'],
                'sectionTitle': row['sectionTitle'],
                'text': row['text'],
                'spans': row['spans'],  # Citations as JSON string
                'conference': row['conference'],
                'year': int(row['year']),
                'likelyRelatedWorkSection': row['likelyRelatedWorkSection'] == 'True',
                'refCount': int(row['refCount'])
            }

            papers[corpus_id].append(paragraph)

    print(f"Found {len(papers)} unique papers")
    print(f"Writing to {output_file}...")

    # Write to JSONL
    with open(output_file, 'w') as f:
        for corpus_id in sorted(papers.keys()):
            paper_obj = {
                'corpusId': corpus_id,
                'title': papers[corpus_id][0]['title'],  # All paragraphs have same title
                'conference': papers[corpus_id][0]['conference'],
                'year': papers[corpus_id][0]['year'],
                'paragraphCount': len(papers[corpus_id]),
                'paragraphs': papers[corpus_id]
            }
            f.write(json.dumps(paper_obj) + '\n')

    # Print statistics
    paragraph_counts = [len(paragraphs) for paragraphs in papers.values()]
    print(f"\n✓ Created {output_file}")
    print(f"  Total papers: {len(papers)}")
    print(f"  Total paragraphs: {sum(paragraph_counts)}")
    print(f"  Avg paragraphs per paper: {sum(paragraph_counts)/len(papers):.1f}")
    print(f"  Min paragraphs: {min(paragraph_counts)}")
    print(f"  Max paragraphs: {max(paragraph_counts)}")


if __name__ == "__main__":
    organize_paragraphs()
