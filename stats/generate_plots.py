#!/usr/bin/env python3
"""
Generate histogram plots for publication statistics.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


def generate_plots(papers_path, output_dir):
    """
    Generate histogram plots from papers data.

    Args:
        papers_path: Path to papers.jsonl file
        output_dir: Directory to save the plots
    """
    papers_path = Path(papers_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading papers from {papers_path}...")

    # Collect data
    publication_years = []
    author_counts = []
    paper_lengths = []

    with open(papers_path, 'r') as f:
        for line in f:
            paper = json.loads(line)

            # Collect publication years
            if paper.get('year'):
                publication_years.append(paper['year'])

            # Collect author counts
            if paper.get('authors'):
                author_counts.append(len(paper['authors']))

            # Collect paper length (word count from paragraphs)
            if paper.get('paragraphs'):
                word_count = 0
                for paragraph in paper['paragraphs']:
                    if paragraph.get('text'):
                        # Count whitespace-separated words
                        word_count += len(paragraph['text'].split())
                if word_count > 0:
                    paper_lengths.append(word_count)

    print(f"Collected data from {len(publication_years)} papers")
    print(f"  Papers with paragraph data: {len(paper_lengths)}")

    # Set up square figure size
    fig_size = (6, 6)

    # 1. Publication Years Histogram
    plt.figure(figsize=fig_size)
    year_counts = Counter(publication_years)
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]

    plt.bar(years, counts, width=0.8, color='#2E86AB', edgecolor='black', linewidth=0.5)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()

    output_path = output_dir / 'publication_years.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved publication years histogram to {output_path}")

    # 2. Authors per Paper Histogram
    plt.figure(figsize=fig_size)
    author_count_hist = Counter(author_counts)
    num_authors = sorted(author_count_hist.keys())
    counts = [author_count_hist[n] for n in num_authors]

    plt.bar(num_authors, counts, width=0.8, color='#A23B72', edgecolor='black', linewidth=0.5)
    plt.xlabel('Number of Authors')
    plt.ylabel('Count')
    plt.tight_layout()

    output_path = output_dir / 'authors_per_paper.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved authors per paper histogram to {output_path}")

    # 3. Paper Length Histogram (word count)
    if paper_lengths:
        plt.figure(figsize=fig_size)

        # Use bins for better visualization of word count distribution
        plt.hist(paper_lengths, bins=30, color='#F18F01', edgecolor='black', linewidth=0.5)
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        plt.tight_layout()

        output_path = output_dir / 'paper_lengths.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved paper lengths histogram to {output_path}")
        print(f"  Word count range: {min(paper_lengths):,} - {max(paper_lengths):,}")
        print(f"  Mean: {sum(paper_lengths)/len(paper_lengths):.0f} words")
    else:
        print("⚠ No paragraph data found, skipping paper lengths histogram")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <input_papers.jsonl> [output_dir]")
        print("\nExample:")
        print("  python generate_plots.py data/raw/papers.jsonl stats")
        sys.exit(1)

    papers_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'stats'

    generate_plots(papers_path, output_dir)
