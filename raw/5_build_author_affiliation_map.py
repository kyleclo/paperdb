#!/usr/bin/env python3
"""
Build author ID to affiliation mapping using OpenAlex API with batch queries.
Queries OpenAlex by DOI to get affiliation data for authors.
"""

import requests
import json
import time
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
from collections import defaultdict


def query_openalex_batch(dois, email=None):
    """
    Fetch multiple papers from OpenAlex by DOI in a single batch query.

    Args:
        dois: List of DOI strings
        email: Email for polite API usage

    Returns:
        Dict mapping DOI -> OpenAlex work data
    """
    if not dois:
        return {}

    # Build filter with pipe-separated DOIs
    doi_filter = '|'.join(dois)
    url = f"https://api.openalex.org/works?filter=doi:{doi_filter}&per-page=50"

    headers = {}
    if email:
        headers['User-Agent'] = f'mailto:{email}'

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            # Build map of DOI -> work
            results = {}
            for work in data.get('results', []):
                work_doi = work.get('doi', '').replace('https://doi.org/', '')
                if work_doi:
                    results[work_doi.lower()] = work
            return results
    except Exception as e:
        return {}

    return {}


def extract_author_affiliations(openalex_work, s2_authors):
    """
    Extract author ID to affiliation mappings from OpenAlex work.

    Args:
        openalex_work: OpenAlex work response
        s2_authors: List of S2 author dicts with authorId and name

    Returns:
        Dict mapping S2 author IDs to affiliation data
    """
    if not openalex_work or 'authorships' not in openalex_work:
        return {}

    author_affiliations = {}
    authorships = openalex_work['authorships']

    # Try to match OpenAlex authors to S2 authors by name
    # This is approximate matching since IDs may not align perfectly
    for i, authorship in enumerate(authorships):
        institutions = authorship.get('institutions', [])

        if not institutions:
            continue

        # Get OpenAlex author name
        oa_author = authorship.get('author', {})
        oa_name = oa_author.get('display_name', '')

        # Try to match to S2 author by position or name similarity
        matched_s2_author = None

        # Strategy 1: Match by position (if same order)
        if i < len(s2_authors):
            matched_s2_author = s2_authors[i]

        # Strategy 2: Match by name similarity (simple exact match on last name)
        if not matched_s2_author and oa_name:
            oa_last_name = oa_name.split()[-1].lower() if oa_name else ''
            for s2_author in s2_authors:
                s2_name = s2_author.get('name', '')
                s2_last_name = s2_name.split()[-1].lower() if s2_name else ''
                if oa_last_name and s2_last_name and oa_last_name == s2_last_name:
                    matched_s2_author = s2_author
                    break

        if matched_s2_author:
            author_id = matched_s2_author.get('authorId')
            if author_id:
                # Extract institution info
                affiliation_list = []
                for inst in institutions:
                    affiliation_list.append({
                        'id': inst.get('id'),
                        'display_name': inst.get('display_name'),
                        'ror': inst.get('ror'),
                        'country_code': inst.get('country_code'),
                        'type': inst.get('type')
                    })

                author_affiliations[author_id] = {
                    'name': matched_s2_author.get('name'),
                    'affiliations': affiliation_list
                }

    return author_affiliations


def main():
    print("=" * 80)
    print("BUILDING AUTHOR-AFFILIATION MAP FROM OPENALEX (BATCH MODE)")
    print("=" * 80)
    print()

    # Configuration
    email = "your.email@example.com"  # Replace with your email for better rate limits
    batch_size = 50  # Query 50 papers at a time
    rate_limit_delay = 0.1  # 100ms between batch requests (10 batches/sec)

    print(f"Email for API: {email}")
    print(f"Batch size: {batch_size} papers per request")
    print(f"Rate limit: {1/rate_limit_delay:.0f} batches/second = {batch_size/rate_limit_delay:.0f} papers/second")
    print("(Set email to your real address for better rate limits)")
    print()

    # Paths
    input_path = 'raw/dblp-nlp-ml-ai-oa-recent-with-fulltext'
    output_path = Path('raw/author_affiliation_map.json')

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"✓ Loaded {len(dataset):,} papers\n")

    # Build author-affiliation map
    author_map = {}  # Maps author_id -> affiliation data

    papers_processed = 0
    papers_with_doi = 0
    papers_found_in_oa = 0
    authors_mapped = 0

    print("Processing papers in batches...")
    print(f"  Batch size: {batch_size}")
    print(f"  Rate limiting: {rate_limit_delay*1000:.0f}ms between batches")
    print()

    # Prepare batches
    batches = []
    current_batch = []
    batch_papers = []

    for record in dataset:
        external_ids = record.get('externalids', {})
        doi = external_ids.get('DOI') if external_ids else None
        authors = record.get('authors', [])

        if doi and authors:
            current_batch.append(doi)
            batch_papers.append((doi, authors))

            if len(current_batch) >= batch_size:
                batches.append((current_batch[:], batch_papers[:]))
                current_batch = []
                batch_papers = []

    # Add remaining papers
    if current_batch:
        batches.append((current_batch, batch_papers))

    print(f"Created {len(batches):,} batches")
    print()

    # Process batches
    for batch_idx, (dois, papers) in enumerate(tqdm(batches, desc="Processing batches")):
        # Query OpenAlex for this batch
        oa_results = query_openalex_batch(dois, email)

        # Process each paper in the batch
        for doi, s2_authors in papers:
            papers_processed += 1
            papers_with_doi += 1

            # Look up this paper's results
            doi_normalized = doi.lower()
            if doi_normalized in oa_results:
                papers_found_in_oa += 1
                oa_work = oa_results[doi_normalized]

                # Extract author-affiliation mappings
                paper_author_map = extract_author_affiliations(oa_work, s2_authors)

                # Merge into main map (keep first occurrence)
                for author_id, affil_data in paper_author_map.items():
                    if author_id not in author_map:
                        author_map[author_id] = affil_data
                        authors_mapped += 1

        # Rate limiting between batches
        time.sleep(rate_limit_delay)

        # Progress update every 20 batches
        if (batch_idx + 1) % 20 == 0:
            tqdm.write(f"  Progress: {papers_processed:,} papers | "
                      f"DOIs: {papers_with_doi:,} | "
                      f"Found in OA: {papers_found_in_oa:,} | "
                      f"Authors mapped: {authors_mapped:,}")

    print(f"\n✓ Processing complete!")
    print(f"  Papers processed: {papers_processed:,}")
    print(f"  Papers with DOI: {papers_with_doi:,} ({papers_with_doi/papers_processed*100:.1f}%)")
    print(f"  Found in OpenAlex: {papers_found_in_oa:,} ({papers_found_in_oa/papers_with_doi*100:.1f}% of DOI papers)")
    print(f"  Unique authors mapped: {authors_mapped:,}")

    # Save author map
    print(f"\nSaving author-affiliation map...")
    with open(output_path, 'w') as f:
        json.dump(author_map, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Total authors: {len(author_map):,}")

    # Show sample
    if author_map:
        print(f"\nSample author entry:")
        sample_id = list(author_map.keys())[0]
        sample_data = author_map[sample_id]
        print(f"  Author ID: {sample_id}")
        print(f"  Name: {sample_data.get('name')}")
        print(f"  Affiliations: {len(sample_data.get('affiliations', []))}")
        if sample_data.get('affiliations'):
            first_affil = sample_data['affiliations'][0]
            print(f"    - {first_affil.get('display_name')}")


if __name__ == '__main__':
    main()
