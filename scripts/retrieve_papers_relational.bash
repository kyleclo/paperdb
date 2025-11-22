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
MODEL=${OPENAI_MODEL:-"gpt-5.1"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"detailed"}

style_query_output_list=(
    # "detailed data/synth/title_as_query/train.jsonl results/title_as_query/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train.jsonl results/metadata_as_query/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.4_md0.3.jsonl results/metadata_as_query_td0.4_md0.3/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.4_md0.7.jsonl results/metadata_as_query_td0.4_md0.7/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.7_md0.3.jsonl results/metadata_as_query_td0.7_md0.3/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.7_md0.7.jsonl results/metadata_as_query_td0.7_md0.7/train.relational-detailed.results.jsonl"

    "minimal data/synth/title_as_query/train.jsonl results/title_as_query/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train.jsonl results/metadata_as_query/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train_td0.4_md0.3.jsonl results/metadata_as_query_td0.4_md0.3/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train_td0.4_md0.7.jsonl results/metadata_as_query_td0.4_md0.7/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train_td0.7_md0.3.jsonl results/metadata_as_query_td0.7_md0.3/train.relational-minimal.results.jsonl"
    "minimal data/synth/metadata_as_query/train_td0.7_md0.7.jsonl results/metadata_as_query_td0.7_md0.7/train.relational-minimal.results.jsonl"
)

echo "Text-to-SQL Retrieval | DB: $DB_NAME | Model: $MODEL | Prompt: $SYSTEM_PROMPT"

for args in "${style_query_output_list[@]}"; do
    read system_prompt query_file output_file <<< "$args"
    echo "Processing: $system_prompt | $query_file → $output_file"
    
    python $PROJECT_ROOT/db/retrieve_relational.py \
        --db_name $DB_NAME \
        --db_user $DB_USER \
        --db_password $DB_PASSWORD \
        --db_host $DB_HOST \
        --db_port $DB_PORT \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --model $MODEL \
        --system_prompt $system_prompt
done
