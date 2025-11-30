#!/bin/bash

set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate

# Database credentials
DB_NAME="paperdb"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# Model configuration
MODEL="${MODEL:-gpt-4o}"
MAX_TURNS="${MAX_TURNS:-5}"

# Index path
INDEX_PATH="$PROJECT_ROOT/data/index-all-units"

# Query-output pairs
# Format: "query_file output_file"
query_output_list=(
    # Example: uncomment and modify these lines
    # "data/synth/title_as_query/train.jsonl results/title_as_query/train.agent-${MODEL}-t${MAX_TURNS}.results.jsonl"
    # "data/synth/metadata_as_query/train.jsonl results/metadata_as_query/train.agent-${MODEL}-t${MAX_TURNS}.results.jsonl"
    
    # metadata_as_query/train_td0.0_md0.0
    "data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.agent-${MODEL}-t${MAX_TURNS}.results.jsonl"
)

# Retrieve papers using agent-based retrieval
for args in "${query_output_list[@]}"; do
    query_file=$(echo $args | awk '{print $1}')
    output_file=$(echo $args | awk '{print $2}')
    
    echo "========================================"
    echo "Processing: $query_file"
    echo "Output: $output_file"
    echo "Model: $MODEL, Max turns: $MAX_TURNS"
    echo "========================================"
    
    python $PROJECT_ROOT/db/retrieve_agent.py \
        --db_name $DB_NAME \
        --db_user $DB_USER \
        --db_password $DB_PASSWORD \
        --db_host $DB_HOST \
        --db_port $DB_PORT \
        --index_path $INDEX_PATH \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --model $MODEL \
        --max_turns $MAX_TURNS
    
    echo ""
done

echo "All queries processed successfully!"

