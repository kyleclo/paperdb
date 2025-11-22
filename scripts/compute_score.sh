#!/bin/bash

# Get the project root and activate virtual environment
PROJECT_ROOT=$(git rev-parse --show-toplevel)
source $PROJECT_ROOT/.venv/bin/activate

# Set up directories
RESULTS_DIR="$PROJECT_ROOT/results"
EVAL_DIR="$PROJECT_ROOT/eval"

echo "Computing scores for all results files in $RESULTS_DIR"
echo "================================================"

# Find all .jsonl files in the results directory
find "$RESULTS_DIR" -name "*.jsonl" | while read -r results_file; do
    # Get the directory and base name
    dir=$(dirname "$results_file")
    base=$(basename "$results_file" .jsonl)
    
    # Create metrics file name
    metrics_file="${dir}/${base}.metrics.json"
    
    echo ""
    echo "Processing: $results_file"
    echo "Output: $metrics_file"
    
    # Run the score computation
    python "$EVAL_DIR/score.py" "$results_file" "$metrics_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Success"
    else
        echo "✗ Failed"
    fi
done

echo ""
echo "================================================"
echo "All scores computed!"

