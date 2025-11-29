#!/usr/bin/env python3
"""
Filter NLP/ML/AI subset to only open access papers from year 2000 onwards.
Creates a smaller, more accessible subset for research.
"""

from datasets import load_from_disk
from pathlib import Path
from collections import Counter


def is_oa_and_recent(example):
    """
    Filter function - returns True if paper is open access and published since 2000.

    Args:
        example: Dataset record

    Returns:
        bool: True if both conditions met
    """
    # Check year (>= 2000)
    year = example.get('year')
    if not year or year < 2000:
        return False

    # Check open access
    is_oa = example.get('isopenaccess', False)
    if not is_oa:
        return False

    return True


def main():
    print("=" * 80)
    print("CREATING OPEN ACCESS + RECENT SUBSET")
    print("Filters: isopenaccess=True AND year>=2000")
    print("=" * 80)
    print()

    input_path = 'raw/dblp-nlp-ml-ai-subset'
    output_path = Path('raw/dblp-nlp-ml-ai-oa-recent-subset')

    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"✓ Loaded {len(dataset):,} papers\n")

    print("Filtering with multiprocessing...")
    print("  Condition 1: isopenaccess = True")
    print("  Condition 2: year >= 2000")
    print()

    # Use HuggingFace's built-in filter with multiprocessing
    filtered_dataset = dataset.filter(
        is_oa_and_recent,
        num_proc=8,
        desc="Filtering OA + recent"
    )

    print(f"\n✓ Filtering complete!")
    print(f"  Original: {len(dataset):,} papers")
    print(f"  Filtered: {len(filtered_dataset):,} papers")
    print(f"  Kept: {len(filtered_dataset)/len(dataset)*100:.2f}%")

    # Analyze the filtered dataset
    print(f"\nAnalyzing filtered dataset...")

    # Year distribution
    years = [row['year'] for row in filtered_dataset if row.get('year')]
    print(f"\nYear range: {min(years)} - {max(years)}")

    year_counts = Counter(years)
    print(f"\nPapers by decade:")
    for decade in [2000, 2010, 2020]:
        decade_count = sum(count for year, count in year_counts.items() if decade <= year < decade + 10)
        print(f"  {decade}s: {decade_count:,}")

    # Venue distribution
    venues = [row['venue'] for row in filtered_dataset if row.get('venue')]
    venue_counts = Counter(venues)

    print(f"\nTop 15 venues:")
    for venue, count in venue_counts.most_common(15):
        print(f"  {count:5d}  {venue}")

    print(f"\nTotal unique venues: {len(venue_counts):,}")

    # Abstract coverage
    with_abstract = sum(1 for row in filtered_dataset if row.get('abstract'))
    print(f"\nAbstract coverage: {with_abstract:,}/{len(filtered_dataset):,} ({with_abstract/len(filtered_dataset)*100:.1f}%)")

    # Save filtered dataset
    print(f"\nSaving to {output_path}...")
    filtered_dataset.save_to_disk(str(output_path))

    print(f"✓ Saved {len(filtered_dataset):,} papers")
    print(f"✓ Location: {output_path}")


if __name__ == '__main__':
    main()
