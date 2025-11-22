import json
import sys
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_metrics(results: list[dict]) -> dict:
    total = len(results)
    hits_at_1 = sum(1 for r in results if r["expected"] == r["retrieved"][0])
    hits_at_5 = sum(1 for r in results if r["expected"] in r["retrieved"][:5])

    mrr = sum(
        1 / (r["retrieved"].index(r["expected"]) + 1)
        for r in results
        if r["expected"] in r["retrieved"]
    ) / total

    return {
        "hits@1": hits_at_1 / total,
        "hits@5": hits_at_5 / total,
        "mrr": mrr,
        "total_queries": total
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python score.py <results_file>")
        sys.exit(1)

    results_path = sys.argv[1]
    results = load_jsonl(results_path)

    metrics = calculate_metrics(results)

    print(f"\nEvaluation Metrics for: {results_path}")
    print(f"  Hits@1: {metrics['hits@1']:.2%}")
    print(f"  Hits@5: {metrics['hits@5']:.2%}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  Total queries: {metrics['total_queries']}")


if __name__ == "__main__":
    main()
