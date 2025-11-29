#!/usr/bin/env python3
"""
Create NLP/ML/AI subset using HuggingFace datasets' built-in parallelization.
Much faster than iterating through records.
"""

from datasets import load_from_disk
from pathlib import Path


# Venue keywords for each category
NLP_KEYWORDS = ['acl', 'emnlp', 'naacl', 'eacl', 'coling', 'lrec', 'conll', 'semeval',
                'computational linguistics', 'natural language']
ML_KEYWORDS = ['neurips', 'nips', 'icml', 'iclr', 'machine learning']
AI_KEYWORDS = ['aaai', 'ijcai', 'uai', 'artificial intelligence']


def is_nlp_ml_ai_venue(example):
    """Filter function - returns True if venue matches NLP/ML/AI."""
    venue = example.get('venue', '')
    if not venue:
        return False

    venue_lower = venue.lower()

    # Check if venue contains any of the keywords
    all_keywords = NLP_KEYWORDS + ML_KEYWORDS + AI_KEYWORDS
    return any(keyword in venue_lower for keyword in all_keywords)


def main():
    print("=" * 80)
    print("CREATING NLP/ML/AI SUBSET (FAST VERSION)")
    print("=" * 80)
    print()

    input_path = 'raw/dblp-discovery-dataset'
    output_path = Path('raw/dblp-nlp-ml-ai-subset')

    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    data = dataset['train']
    print(f"✓ Loaded {len(data):,} papers\n")

    print("Filtering with multiprocessing...")
    print(f"  Venue patterns: {len(NLP_KEYWORDS)} NLP + {len(ML_KEYWORDS)} ML + {len(AI_KEYWORDS)} AI")
    print()

    # Use HuggingFace's built-in filter with multiprocessing
    filtered_dataset = data.filter(
        is_nlp_ml_ai_venue,
        num_proc=8,  # Use 8 processes
        desc="Filtering venues"
    )

    print(f"\n✓ Filtering complete!")
    print(f"  Original: {len(data):,} papers")
    print(f"  Filtered: {len(filtered_dataset):,} papers")
    print(f"  Kept: {len(filtered_dataset)/len(data)*100:.2f}%")

    # Analyze venues in filtered set
    print(f"\nAnalyzing filtered dataset...")
    from collections import Counter

    venues = [row['venue'] for row in filtered_dataset if row.get('venue')]
    venue_counts = Counter(venues)

    print(f"\nTop 20 venues:")
    for venue, count in venue_counts.most_common(20):
        print(f"  {count:5d}  {venue}")

    print(f"\nTotal unique venues: {len(venue_counts)}")

    # Save filtered dataset
    print(f"\nSaving to {output_path}...")
    filtered_dataset.save_to_disk(str(output_path))

    print(f"✓ Saved {len(filtered_dataset):,} papers")
    print(f"✓ Location: {output_path}")


if __name__ == '__main__':
    main()
