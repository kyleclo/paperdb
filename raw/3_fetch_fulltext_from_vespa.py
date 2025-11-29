#!/usr/bin/env python3
"""
Fetch full text for papers using internal Vespa API snippets.
Reconstructs full text from snippets (title, abstract, body).
"""

import requests
import json
import time
from datasets import load_from_disk, Dataset
from pathlib import Path
from collections import defaultdict


VESPA_ENDPOINT = "http://internal-vespa--state-acj0iq9ltcdn-958141466.us-west-2.elb.amazonaws.com:8080/search/"


def get_snippets(corpus_id, max_snippets=400):
    """
    Query Vespa for all snippets of a paper.

    Args:
        corpus_id: Semantic Scholar corpus ID
        max_snippets: Maximum snippets to retrieve

    Returns:
        List of snippet dictionaries
    """
    epoch_time = int(time.time())

    query = {
        "yql": f'SELECT paper_corpus_id, snippet_idx, text, snippet_kind, snippet_start_offset, snippet_end_offset, section from snippet where paper_corpus_id contains "{corpus_id}" and paper_searchable_after_long < {epoch_time}',
        "timeout": 60,
        "hits": max_snippets,
        "ranking": "unranked",
        "ranking.sorting": "+snippet_idx"  # Sort by snippet index
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(VESPA_ENDPOINT, headers=headers, json=query, timeout=65)

        if response.status_code == 200:
            data = response.json()
            if 'root' in data and 'children' in data['root']:
                return [child.get('fields', {}) for child in data['root']['children']]
            return []
        else:
            return []

    except Exception as e:
        return []


def reconstruct_fulltext(snippets):
    """
    Reconstruct full text from snippets.

    Args:
        snippets: List of snippet dictionaries from Vespa

    Returns:
        Dictionary with title, abstract, body
    """
    # Group snippets by kind
    by_kind = defaultdict(list)

    for snippet in snippets:
        kind = snippet.get('snippet_kind', 'unknown')
        by_kind[kind].append(snippet)

    # Sort each kind by snippet_idx to ensure proper order
    for kind in by_kind:
        by_kind[kind].sort(key=lambda x: x.get('snippet_idx', 0))

    # Reconstruct text for each section
    result = {}

    for kind in ['title', 'abstract', 'body']:
        if kind in by_kind:
            # Join snippets with space
            text = ' '.join(s.get('text', '') for s in by_kind[kind])
            result[kind] = text.strip()

    return result


def main():
    print("=" * 80)
    print("FETCHING FULL TEXT FROM VESPA SNIPPETS")
    print("=" * 80)
    print()

    # Paths
    input_path = 'raw/dblp-nlp-ml-ai-oa-recent-subset'
    output_path = Path('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext')

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"✓ Loaded {len(dataset):,} papers\n")

    print("Starting full text retrieval...")
    print(f"  Output: {output_path}")
    print(f"  Rate limiting: 100ms between requests")
    print()

    # Process papers
    papers_with_fulltext = []
    success_count = 0
    fail_count = 0

    for i, paper in enumerate(dataset):
        corpus_id = str(paper['corpusid'])

        # Query for snippets
        snippets = get_snippets(corpus_id)

        if snippets:
            # Reconstruct full text
            fulltext = reconstruct_fulltext(snippets)

            if fulltext:
                # Add fulltext to paper record
                paper_with_text = dict(paper)
                paper_with_text['fulltext_title'] = fulltext.get('title', '')
                paper_with_text['fulltext_abstract'] = fulltext.get('abstract', '')
                paper_with_text['fulltext_body'] = fulltext.get('body', '')
                paper_with_text['snippet_count'] = len(snippets)

                papers_with_fulltext.append(paper_with_text)
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

        # Progress updates
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1:,}/{len(dataset):,} papers | Success: {success_count:,} | Failed: {fail_count:,}")

        # Rate limiting (10 requests per second max)
        time.sleep(0.1)

    print(f"\n✓ Fetch complete!")
    print(f"  Total processed: {len(dataset):,}")
    print(f"  Successful: {success_count:,} ({success_count/len(dataset)*100:.1f}%)")
    print(f"  Failed: {fail_count:,}")

    # Show sample
    if papers_with_fulltext:
        print(f"\nSample paper with full text:")
        sample = papers_with_fulltext[0]
        print(f"  Title: {sample['title'][:60]}...")
        print(f"  Snippets retrieved: {sample['snippet_count']}")
        print(f"  Full text title length: {len(sample['fulltext_title'])} chars")
        print(f"  Full text abstract length: {len(sample['fulltext_abstract'])} chars")
        print(f"  Full text body length: {len(sample['fulltext_body'])} chars")

    # Save to new dataset
    if papers_with_fulltext:
        print(f"\nSaving dataset with full text...")
        output_dataset = Dataset.from_list(papers_with_fulltext)
        output_dataset.save_to_disk(str(output_path))

        print(f"✓ Saved {len(papers_with_fulltext):,} papers with full text")
        print(f"✓ Location: {output_path}")
    else:
        print("\n⚠ No papers with full text retrieved")


if __name__ == '__main__':
    main()
