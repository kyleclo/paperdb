import json
import argparse
from pathlib import Path
from retrieval.dense import DenseRetriever
from sql.query import SQLRetriever


def load_queries(queries_path: str) -> list[dict]:
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def main():
    parser = argparse.ArgumentParser(description="Retrieve papers using FAISS index")
    parser.add_argument('--index_path', type=str, required=True,
                       help='Path to the FAISS index directory')
    parser.add_argument('--query_file', type=str, required=True,
                       help='Path to the query file (JSONL format)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to the output results file (JSONL format)')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of units to retrieve per query (default: 100)')
    
    args = parser.parse_args()
    
    # Load queries
    queries_path = Path(args.query_file)
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path}")

    # Initialize retriever
    retriever = DenseRetriever(index_dir=args.index_path)
    retriever.load()

    # Retrieve for each query
    results = []
    for query_data in queries:
        query = query_data["query"]
        
        # Retrieve top-k units
        retrieved_data = retriever.retrieve(query, k=args.k)
        
        # Build result entry
        result = {
            "query": query,
            "expected": query_data["paperId"],
            "retrieved": retrieved_data["paper_ids"],  # Deduplicated paper IDs for scoring
            "units": {
                k: v for k, v in 
                zip(retrieved_data["unit_ids"], retrieved_data["unit_texts"])
            }, # unit_ids and unit_texts are used for manual inspection
        }
        results.append(result)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Retrieved {len(results)} queries with k={args.k}")
    print(f"  Output saved to: {output_path}")
    print(f"  Average unique papers per query: {sum(len(r['retrieved']) for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
