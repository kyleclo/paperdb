#!/bin/bash

set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate

# List of results files to compute scores for
# results_list=(
#     "results/title_as_query/train.dense-all-units.results.jsonl"
#     "results/metadata_as_query/train.dense-all-units.results.jsonl"
#     "results/title_as_query/train.dense-para-abs.results.jsonl"
#     "results/metadata_as_query/train.dense-para-abs.results.jsonl"
# )

# # Compute scores for each results file
# for results_file in "${results_list[@]}"; do
#     metrics_file="${results_file%.jsonl}.metrics.json"
    
#     python $PROJECT_ROOT/eval/score.py \
#         $PROJECT_ROOT/$results_file \
#         $PROJECT_ROOT/$metrics_file
# done

# find all files in results/** ends with results.jsonl and compute scores

# find $PROJECT_ROOT/results -name "*.results.jsonl" -exec python $PROJECT_ROOT/eval/score.py {} {}.metrics.json \;
find $PROJECT_ROOT/results -name "*.results.jsonl" | while read results_file; do
    metrics_file=$(echo $results_file | sed 's/.results.jsonl$/.metrics.json/')
    echo "Computing scores for $results_file -> $metrics_file"
    python $PROJECT_ROOT/eval/score.py \
        "$results_file" \
        "$metrics_file"
done
