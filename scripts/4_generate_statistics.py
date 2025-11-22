#!/usr/bin/env python3
"""
Generate aggregate statistics from papers.jsonl and output a markdown report.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def generate_statistics(input_path, output_dir):
    """
    Generate statistics from papers.jsonl and save to markdown report.

    Args:
        input_path: Path to papers.jsonl file
        output_dir: Directory to save the statistics report
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading papers from {input_path}...")

    # Data structures for statistics
    paper_count = 0
    unique_authors = set()
    publication_years = []
    venues = []
    author_counts = []

    # Read and process papers
    with open(input_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            paper_count += 1

            # Collect unique authors
            if paper.get('authors'):
                author_counts.append(len(paper['authors']))
                for author in paper['authors']:
                    if author.get('authorId'):
                        unique_authors.add(author['authorId'])

            # Collect publication years
            if paper.get('year'):
                publication_years.append(paper['year'])

            # Collect venues
            if paper.get('venue'):
                venues.append(paper['venue'])

    print(f"Processed {paper_count} papers")

    # Generate report
    output_path = output_dir / 'paper_statistics.md'

    with open(output_path, 'w') as f:
        f.write("# Paper Statistics\n\n")

        # 1. Total number of papers
        f.write("## Total Papers\n\n")
        f.write(f"{paper_count:,}\n\n")

        # 2. Unique authors
        f.write("## Unique Authors\n\n")
        f.write(f"{len(unique_authors):,}\n\n")

        # 3. Publication years histogram
        f.write("## Publication Years\n\n")
        year_counts = Counter(publication_years)
        f.write("| Year | Count |\n")
        f.write("|------|-------|\n")
        for year in sorted(year_counts.keys()):
            f.write(f"| {year} | {year_counts[year]:,} |\n")
        f.write("\n")

        # 4. Venue counts
        f.write("## Venues\n\n")
        venue_counts = Counter(venues)
        f.write("| Venue | Count |\n")
        f.write("|-------|-------|\n")
        # Sort by count descending
        for venue, count in venue_counts.most_common():
            f.write(f"| {venue} | {count:,} |\n")
        f.write("\n")

        # 5. Number of authors histogram
        f.write("## Authors per Paper\n\n")
        author_count_hist = Counter(author_counts)
        f.write("| Number of Authors | Count |\n")
        f.write("|-------------------|-------|\n")
        for num_authors in sorted(author_count_hist.keys()):
            f.write(f"| {num_authors} | {author_count_hist[num_authors]:,} |\n")
        f.write("\n")

    print(f"✓ Statistics report saved to {output_path}")

    # Print summary to console
    print("\n=== Summary ===")
    print(f"Total papers: {paper_count:,}")
    print(f"Unique authors: {len(unique_authors):,}")
    print(f"Year range: {min(publication_years) if publication_years else 'N/A'} - {max(publication_years) if publication_years else 'N/A'}")
    print(f"Number of venues: {len(venue_counts)}")
    print(f"Top venue: {venue_counts.most_common(1)[0][0] if venue_counts else 'N/A'} ({venue_counts.most_common(1)[0][1]} papers)")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 4_generate_statistics.py <input_papers.jsonl> [output_dir]")
        print("\nExample:")
        print("  python 4_generate_statistics.py data/raw/papers.jsonl stats")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'stats'

    generate_statistics(input_path, output_dir)
