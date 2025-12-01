#!/bin/bash
set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
source $PROJECT_ROOT/.venv/bin/activate

[ -z "$OPENAI_API_KEY" ] && echo "Error: OPENAI_API_KEY not set" && exit 1

DB_NAME=${PAPERDB_NAME:-"paperdb"}
DB_USER=${PAPERDB_USER:-$USER}
DB_PASSWORD=${PAPERDB_PASSWORD:-"your_password"}
DB_HOST=${PAPERDB_HOST:-"localhost"}
DB_PORT=${PAPERDB_PORT:-5432}
MODEL=${OPENAI_MODEL:-"gpt-4o"}
MAX_TURNS=${MAX_TURNS:-5}

index_query_output_list=(
    # title_as_query
    # "data/index-all-units-v2 data/synth/title_as_query/train.jsonl results/title_as_query/train.agent-all-units.results.jsonl"
    "data/index-para-abs-v2 data/synth/title_as_query/train.jsonl results/title_as_query/train.agent.para-abs.results.jsonl"
)

echo "Agent Retrieval | DB: $DB_NAME | Model: $MODEL | Max Turns: $MAX_TURNS"

for args in "${index_query_output_list[@]}"; do
    index_dir=$(echo $args | awk '{print $1}')  
    query_file=$(echo $args | awk '{print $2}')
    output_file=$(echo $args | awk '{print $3}')
    
    echo "Processing: $index_dir | $query_file → $output_file"
    
    python $PROJECT_ROOT/db/retrieve_agent.py \
        --db_name $DB_NAME \
        --db_user $DB_USER \
        --db_password $DB_PASSWORD \
        --db_host $DB_HOST \
        --db_port $DB_PORT \
        --index_path $PROJECT_ROOT/$index_dir \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --model $MODEL \
        --max_turns $MAX_TURNS
done
