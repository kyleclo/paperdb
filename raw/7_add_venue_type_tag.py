#!/usr/bin/env python3
"""
Add venue_type metadata tag (main/workshop) to all papers.
Detects workshops using @ pattern and adds venue_type field.
"""

import json
from pathlib import Path
from tqdm import tqdm


def is_workshop_venue(venue):
    """
    Determine if a venue is a workshop.
    Workshops typically have @ symbol (e.g., BioNLP@ACL).
    """
    if not venue:
        return False
    return '@' in venue


def main():
    print("=" * 80)
    print("ADDING VENUE TYPE TAGS (MAIN/WORKSHOP)")
    print("=" * 80)
    print()

    # Paths
    input_path = Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext.jsonl')
    output_path = Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl')

    # Process papers
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print()

    main_count = 0
    workshop_count = 0
    no_venue_count = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc="Tagging papers"):
            try:
                paper = json.loads(line)

                # Get venue
                venue = paper.get('venue', '')

                # Determine venue type
                if not venue:
                    venue_type = None
                    no_venue_count += 1
                elif is_workshop_venue(venue):
                    venue_type = 'workshop'
                    workshop_count += 1
                else:
                    venue_type = 'main'
                    main_count += 1

                # Add venue_type field
                paper['venue_type'] = venue_type

                # Write updated paper
                fout.write(json.dumps(paper, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                # Skip malformed lines
                continue

    print(f"\n✓ Tagging complete!")
    print(f"  Main conference papers: {main_count:,} ({main_count/(main_count+workshop_count+no_venue_count)*100:.1f}%)")
    print(f"  Workshop papers: {workshop_count:,} ({workshop_count/(main_count+workshop_count+no_venue_count)*100:.1f}%)")
    print(f"  No venue: {no_venue_count:,}")

    # Show file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Show examples
    print(f"\n" + "=" * 80)
    print("EXAMPLE TAGGED PAPERS")
    print("=" * 80)

    examples_shown = {'main': 0, 'workshop': 0}
    max_examples = 2

    with open(output_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            venue_type = paper.get('venue_type')

            if venue_type and examples_shown.get(venue_type, 0) < max_examples:
                print(f"\nVenue type: {venue_type}")
                print(f"  Title: {paper['title'][:60]}...")
                print(f"  Venue: {paper.get('venue', 'N/A')}")
                examples_shown[venue_type] = examples_shown.get(venue_type, 0) + 1

            if all(count >= max_examples for count in examples_shown.values()):
                break


if __name__ == '__main__':
    main()
