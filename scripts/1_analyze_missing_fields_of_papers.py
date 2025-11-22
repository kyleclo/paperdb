import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_missing_fields(jsonl_file):
    """Analyze which fields are missing or null in papers."""

    papers = []
    with open(jsonl_file) as f:
        for line in f:
            papers.append(json.loads(line))

    total_papers = len(papers)

    # Track missing/null counts for each field
    missing_counts = defaultdict(int)
    null_counts = defaultdict(int)
    empty_counts = defaultdict(int)

    # Get all possible fields from all papers
    all_fields = set()
    for paper in papers:
        all_fields.update(paper.keys())

    # Count missing/null/empty for each field
    for paper in papers:
        for field in all_fields:
            if field not in paper:
                missing_counts[field] += 1
            elif paper[field] is None:
                null_counts[field] += 1
            elif isinstance(paper[field], (list, str, dict)) and len(paper[field]) == 0:
                empty_counts[field] += 1

    # Print results
    print(f"\n{'='*70}")
    print(f"Missing Field Analysis for {jsonl_file}")
    print(f"{'='*70}")
    print(f"Total papers: {total_papers}\n")

    print(f"{'Field':<30} {'Missing':<10} {'Null':<10} {'Empty':<10} {'Valid':<10}")
    print(f"{'-'*70}")

    for field in sorted(all_fields):
        missing = missing_counts[field]
        null = null_counts[field]
        empty = empty_counts[field]
        valid = total_papers - missing - null - empty

        missing_pct = f"{missing} ({missing/total_papers*100:.1f}%)"
        null_pct = f"{null} ({null/total_papers*100:.1f}%)"
        empty_pct = f"{empty} ({empty/total_papers*100:.1f}%)"
        valid_pct = f"{valid} ({valid/total_papers*100:.1f}%)"

        print(f"{field:<30} {missing_pct:<10} {null_pct:<10} {empty_pct:<10} {valid_pct:<10}")

    # Print summary of most problematic fields
    print(f"\n{'='*70}")
    print("Most Problematic Fields (>10% missing/null):")
    print(f"{'='*70}")

    problematic = []
    for field in all_fields:
        total_issues = missing_counts[field] + null_counts[field]
        if total_issues > total_papers * 0.1:
            problematic.append((field, total_issues, total_issues/total_papers*100))

    for field, count, pct in sorted(problematic, key=lambda x: x[1], reverse=True):
        print(f"  {field:<30} {count}/{total_papers} ({pct:.1f}%)")

    if not problematic:
        print("  None! All fields are >90% complete.")


if __name__ == "__main__":
    jsonl_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/papers_100.jsonl"
    analyze_missing_fields(jsonl_file)
