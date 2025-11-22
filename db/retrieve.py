import json
from pathlib import Path
from retrieval.dense import DenseRetriever
from sql.query import SQLRetriever


def load_queries(queries_path: str) -> list[dict]:
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def main(method: str = "dense", query_file: str = "test"):
    queries_path = Path(__file__).parent.parent / "data" / "synth" / f"{query_file}.jsonl"
    queries = load_queries(queries_path)

    if method == "dense":
        retriever = DenseRetriever()
        retriever.load()
    else:
        retriever = SQLRetriever()
        retriever.load()

    results = []
    for query_data in queries:
        query = query_data["query"]
        retrieved = retriever.retrieve(query, k=5)
        results.append({
            "query": query,
            "expected": query_data["document_id"],
            "retrieved": retrieved
        })

    results_path = Path(__file__).parent.parent / "results.jsonl"
    with open(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Retrieved {len(results)} queries, saved to {results_path}")


if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "dense"
    query_file = sys.argv[2] if len(sys.argv) > 2 else "test"
    main(method, query_file)
