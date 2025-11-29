#!/usr/bin/env python3
"""
Build venue string to normalized venue ID mapping.
Maps workshops to their main conferences and normalizes venue variants.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def extract_main_conference(venue):
    """
    Extract main conference from workshop venue.
    E.g., "BioNLP@ACL" → "ACL", "WMT@EMNLP" → "EMNLP"
    """
    if not venue:
        return None

    # Check for workshop pattern (something@conference)
    match = re.search(r'@([A-Z]+)', venue)
    if match:
        return match.group(1)

    return None


def normalize_venue_key(venue):
    """
    Normalize venue string for grouping similar venues.
    Removes years, special characters, and standardizes format.
    """
    if not venue:
        return ""

    # Convert to lowercase
    normalized = venue.lower().strip()

    # Remove years (4 digits)
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)

    # Remove volume/issue numbers
    normalized = re.sub(r'\bvol\.\s*\d+\b', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bvolume\s*\d+\b', '', normalized, flags=re.IGNORECASE)

    # Remove common separators and extra spaces
    normalized = re.sub(r'[:\-_/|]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()

    # Remove parenthetical content
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def extract_canonical_name(venue_variants):
    """
    Extract the best canonical name from a list of venue variants.
    Prefers shorter, cleaner names.
    """
    if not venue_variants:
        return ""

    # Sort by length and frequency (if available)
    sorted_variants = sorted(venue_variants, key=lambda x: (len(x[0]), -x[1]))

    # Return the shortest variant (usually the cleanest)
    return sorted_variants[0][0]


def create_venue_mapping(dataset_path, map_workshops_to_main=True):
    """
    Create venue mapping from dataset with intelligent normalization.

    Args:
        dataset_path: Path to JSONLines dataset
        map_workshops_to_main: If True, map workshops to main conferences

    Returns:
        tuple: (venue_to_id, venue_stats)
    """
    # Collect all venue strings from dataset
    print(f"Scanning dataset: {dataset_path}")
    venue_counts = defaultdict(int)

    with open(dataset_path, 'r') as f:
        for line in tqdm(f, desc="Reading papers"):
            try:
                paper = json.loads(line)
                venue = paper.get('venue')
                if venue and isinstance(venue, str) and venue.strip():
                    venue_counts[venue.strip()] += 1
            except json.JSONDecodeError:
                continue

    print(f"\n✓ Found {len(venue_counts):,} unique venue strings")
    print(f"  Total venue occurrences: {sum(venue_counts.values()):,}")

    # Group venues by normalized key or main conference
    print("\nNormalizing and grouping venues...")
    grouped_venues = defaultdict(list)

    workshops_mapped = 0
    for venue_string, count in venue_counts.items():
        canonical = None

        # First, check if it's a workshop and map to main conference
        if map_workshops_to_main:
            main_conf = extract_main_conference(venue_string)
            if main_conf:
                canonical = main_conf
                workshops_mapped += 1

        # If not a workshop, use normalized key
        if not canonical:
            normalized_key = normalize_venue_key(venue_string)
            canonical = normalized_key if normalized_key else venue_string

        grouped_venues[canonical].append((venue_string, count))

    print(f"✓ Grouped into {len(grouped_venues):,} canonical venues")
    if map_workshops_to_main:
        print(f"  Workshops mapped to main conferences: {workshops_mapped:,}")

    # Create mappings
    venue_to_id = {}
    venue_details = {}

    for canonical_key, variants in grouped_venues.items():
        # For main conferences (like ACL, EMNLP), use the canonical_key
        # For others, choose canonical name from shortest variant
        if canonical_key.isupper() and len(canonical_key) <= 10 and ' ' not in canonical_key:
            # This looks like a main conference abbreviation
            canonical_id = canonical_key
        else:
            # Use shortest clean variant
            canonical_id = extract_canonical_name(variants)

        total_count = sum(count for _, count in variants)

        # Map all variants to canonical ID
        variant_list = []
        for variant_string, count in variants:
            venue_to_id[variant_string] = canonical_id
            variant_list.append({
                'name': variant_string,
                'count': count
            })

        # Store details
        venue_details[canonical_id] = {
            'total_count': total_count,
            'num_variants': len(variant_list),
            'variants': sorted(variant_list, key=lambda x: x['count'], reverse=True)
        }

    # Create sorted stats
    venue_stats = []
    for canonical_id, info in sorted(venue_details.items(), key=lambda x: x[1]['total_count'], reverse=True):
        venue_stats.append({
            'venue': canonical_id,
            'count': info['total_count'],
            'num_variants': info['num_variants'],
            'variants': [v['name'] for v in info['variants']]
        })

    return venue_to_id, venue_stats


def main():
    print("=" * 80)
    print("BUILDING VENUE MAPPING (WORKSHOPS → MAIN CONFERENCES)")
    print("=" * 80)
    print()

    # Paths
    dataset_path = Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext.jsonl')
    output_venue_to_id = Path('raw/venue_to_id.json')
    output_venue_stats = Path('raw/venue_stats.json')

    # Create venue mapping
    venue_to_id, venue_stats = create_venue_mapping(dataset_path, map_workshops_to_main=True)

    # Save venue_to_id mapping (for normalization)
    print(f"\nSaving venue-to-id mapping...")
    with open(output_venue_to_id, 'w') as f:
        json.dump(venue_to_id, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(venue_to_id):,} venue mappings to {output_venue_to_id}")

    # Save venue statistics
    print(f"\nSaving venue statistics...")
    with open(output_venue_stats, 'w') as f:
        json.dump(venue_stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved venue statistics to {output_venue_stats}")

    # Show top venues
    print(f"\n" + "=" * 80)
    print("TOP 20 CANONICAL VENUES BY PAPER COUNT")
    print("=" * 80)
    for i, stat in enumerate(venue_stats[:20], 1):
        venue = stat['venue']
        count = stat['count']
        num_variants = stat['num_variants']
        variant_info = f" ({num_variants} variants)" if num_variants > 1 else ""
        print(f"{i:2d}. {venue[:65]:<65} {count:>5d} papers{variant_info}")

    # Show examples of workshop mappings
    print(f"\n" + "=" * 80)
    print("EXAMPLE WORKSHOP → MAIN CONFERENCE MAPPINGS")
    print("=" * 80)

    # Find venues with workshops (those with @ in variant names)
    workshop_venues = []
    for stat in venue_stats:
        has_workshops = any('@' in v for v in stat['variants'])
        if has_workshops and stat['num_variants'] > 1:
            workshop_venues.append(stat)

    for stat in workshop_venues[:10]:
        print(f"\n{stat['venue']} ({stat['count']} total papers, {stat['num_variants']} variants)")
        workshops = [v for v in stat['variants'] if '@' in v]
        main_venues = [v for v in stat['variants'] if '@' not in v]

        if main_venues:
            print(f"  Main conference variants:")
            for v in main_venues[:3]:
                print(f"    - {v}")

        if workshops:
            print(f"  Workshops ({len(workshops)}):")
            for v in workshops[:5]:
                print(f"    - {v}")
            if len(workshops) > 5:
                print(f"    ... and {len(workshops) - 5} more workshops")

    # Show summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_papers = sum(s['count'] for s in venue_stats)
    total_variants = len(venue_to_id)
    total_canonical = len(venue_stats)

    print(f"Total venue strings: {total_variants:,}")
    print(f"Canonical venues: {total_canonical:,}")
    print(f"Reduction: {(1 - total_canonical/total_variants)*100:.1f}%")
    print(f"Total papers: {total_papers:,}")
    print(f"Average papers per canonical venue: {total_papers/total_canonical:.1f}")

    # Count workshops
    workshop_strings = sum(1 for v in venue_to_id.keys() if '@' in v)
    print(f"\nWorkshop venue strings: {workshop_strings:,}")

    # Venues with multiple variants
    multi_variant = sum(1 for s in venue_stats if s['num_variants'] > 1)
    print(f"Canonical venues with multiple variants: {multi_variant:,} ({multi_variant/total_canonical*100:.1f}%)")

    print(f"\nOutput files:")
    print(f"  - {output_venue_to_id} (all venue strings → canonical venue)")
    print(f"  - {output_venue_stats} (sorted stats with variants)")


if __name__ == '__main__':
    main()
