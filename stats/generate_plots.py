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

    # Load venue mappings
    venue_to_id_path = Path('raw/venue_to_id.json')
    venue_mapping = {}
    if venue_to_id_path.exists():
        with open(venue_to_id_path, 'r') as f:
            venue_mapping = json.load(f)
        print(f"Loaded {len(venue_mapping)} venue mappings")

    # Load author affiliation mappings
    author_affiliation_path = Path('raw/author_affiliation_map.json')
    author_affiliation_map = {}
    if author_affiliation_path.exists():
        with open(author_affiliation_path, 'r') as f:
            author_affiliation_map = json.load(f)
        print(f"Loaded {len(author_affiliation_map)} author affiliations")

    # Collect data
    publication_years = []
    author_counts = []
    paper_lengths = []
    canonical_venues = []
    affiliations = []

    with open(papers_path, 'r') as f:
        for line in f:
            paper = json.loads(line)

            # Collect publication years
            if paper.get('year'):
                publication_years.append(paper['year'])

            # Collect author counts
            if paper.get('authors'):
                author_counts.append(len(paper['authors']))

            # Collect paper length (word count from fulltext_body)
            if paper.get('fulltext_body'):
                word_count = len(paper['fulltext_body'].split())
                if word_count > 0:
                    paper_lengths.append(word_count)
            elif paper.get('paragraphs'):
                # Fallback for older data format
                word_count = 0
                for paragraph in paper['paragraphs']:
                    if paragraph.get('text'):
                        word_count += len(paragraph['text'].split())
                if word_count > 0:
                    paper_lengths.append(word_count)

            # Collect canonical venues
            if paper.get('venue'):
                venue = paper['venue']
                # Map to canonical venue if available
                canonical_venue = venue_mapping.get(venue, venue)
                canonical_venues.append(canonical_venue)

            # Collect affiliations (from first author)
            if paper.get('authors') and len(paper['authors']) > 0:
                author_id = paper['authors'][0].get('authorId')
                if author_id and author_id in author_affiliation_map:
                    affil_data = author_affiliation_map[author_id]
                    if affil_data.get('affiliations') and len(affil_data['affiliations']) > 0:
                        affil_name = affil_data['affiliations'][0].get('display_name')
                        if affil_name:
                            affiliations.append(affil_name)

    print(f"Collected data from {len(publication_years)} papers")
    print(f"  Papers with fulltext body: {len(paper_lengths)}")

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
        print("⚠ No fulltext body data found, skipping paper lengths histogram")

    # 4. Canonical Venues Bar Plot (Top 15)
    if canonical_venues:
        plt.figure(figsize=(10, 6))
        venue_counts = Counter(canonical_venues)
        top_venues = venue_counts.most_common(15)
        venues = [v[0] for v in top_venues]
        counts = [v[1] for v in top_venues]

        plt.barh(range(len(venues)), counts, color='#06A77D', edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(venues)), venues)
        plt.xlabel('Number of Papers')
        plt.ylabel('Canonical Venue')
        plt.gca().invert_yaxis()  # Highest count at top
        plt.tight_layout()

        output_path = output_dir / 'venues.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved venues bar plot to {output_path}")
        print(f"  Total unique venues: {len(venue_counts)}")
        print(f"  Showing top 15 venues")
    else:
        print("⚠ No venue data found, skipping venues bar plot")

    # 5. Affiliations Bar Plot (Top 20)
    if affiliations:
        plt.figure(figsize=(10, 8))
        affil_counts = Counter(affiliations)
        top_affils = affil_counts.most_common(20)
        affils = [a[0][:60] + '...' if len(a[0]) > 60 else a[0] for a in top_affils]  # Truncate long names
        counts = [a[1] for a in top_affils]

        plt.barh(range(len(affils)), counts, color='#D62828', edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(affils)), affils, fontsize=9)
        plt.xlabel('Number of Papers (First Author)')
        plt.ylabel('Institution/Organization')
        plt.gca().invert_yaxis()  # Highest count at top
        plt.tight_layout()

        output_path = output_dir / 'affiliations.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved affiliations bar plot to {output_path}")
        print(f"  Total unique affiliations: {len(affil_counts)}")
        print(f"  Papers with affiliation data: {len(affiliations)}")
        print(f"  Showing top 20 affiliations")
    else:
        print("⚠ No affiliation data found, skipping affiliations bar plot")


if __name__ == '__main__':
    import sys

    # Default to the new dataset
    default_papers_path = 'raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl'

    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py [input_papers.jsonl] [output_dir]")
        print("\nExample:")
        print(f"  python generate_plots.py {default_papers_path} stats")
        print("\nNo arguments provided, using default dataset:")
        print(f"  {default_papers_path}")
        papers_path = default_papers_path
    else:
        papers_path = sys.argv[1]

    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'stats'

    generate_plots(papers_path, output_dir)
