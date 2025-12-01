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
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"minimal"}

export CUDA_VISIBLE_DEVICES=0
style_query_output_list=(
    # "detailed data/synth/title_as_query/train.jsonl results/title_as_query/train.relational.detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.4_md0.0.jsonl results/metadata_as_query_td0.4_md0.0/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.8_md0.0.jsonl results/metadata_as_query_td0.8_md0.0/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.0_md0.4.jsonl results/metadata_as_query_td0.0_md0.4/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.4_md0.4.jsonl results/metadata_as_query_td0.4_md0.4/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.8_md0.4.jsonl results/metadata_as_query_td0.8_md0.4/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.0_md0.8.jsonl results/metadata_as_query_td0.0_md0.8/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.4_md0.8.jsonl results/metadata_as_query_td0.4_md0.8/train.relational-detailed.results.jsonl"
    # "detailed data/synth/metadata_as_query/train_td0.8_md0.8.jsonl results/metadata_as_query_td0.8_md0.8/train.relational-detailed.results.jsonl"
    # "detailed data/synth/content_as_query/train_claude_key_passages.jsonl results/content_as_query-claude_key_passages/train.relational-detailed.results.jsonl"
    # "detailed data/synth/content_as_query/train_claude_keywords.jsonl results/content_as_query-claude_keywords/train.relational-detailed.results.jsonl"
    # "detailed data/synth/content_as_query/train_gpt_key_passages.jsonl results/content_as_query-gpt_key_passages/train.relational-detailed.results.jsonl"
    # "detailed data/synth/content_as_query/train_gpt_keywords.jsonl results/content_as_query-gpt_keywords/train.relational-detailed.results.jsonl"

    # "gpt-4o minimal data/synth/title_as_query/train.jsonl results/title_as_query/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.4_md0.0.jsonl results/metadata_as_query_td0.4_md0.0/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.8_md0.0.jsonl results/metadata_as_query_td0.8_md0.0/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.0_md0.4.jsonl results/metadata_as_query_td0.0_md0.4/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.4_md0.4.jsonl results/metadata_as_query_td0.4_md0.4/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.8_md0.4.jsonl results/metadata_as_query_td0.8_md0.4/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.0_md0.8.jsonl results/metadata_as_query_td0.0_md0.8/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.4_md0.8.jsonl results/metadata_as_query_td0.4_md0.8/train.relational.gpt-4o.minimal.results.jsonl"
    # "gpt-4o minimal data/synth/metadata_as_query/train_td0.8_md0.8.jsonl results/metadata_as_query_td0.8_md0.8/train.relational.gpt-4o.minimal.results.jsonl"

    "gpt-5.1 minimal data/synth/title_as_query/train.jsonl results/title_as_query/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.4_md0.0.jsonl results/metadata_as_query_td0.4_md0.0/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.8_md0.0.jsonl results/metadata_as_query_td0.8_md0.0/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.0_md0.4.jsonl results/metadata_as_query_td0.0_md0.4/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.4_md0.4.jsonl results/metadata_as_query_td0.4_md0.4/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.8_md0.4.jsonl results/metadata_as_query_td0.8_md0.4/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.0_md0.8.jsonl results/metadata_as_query_td0.0_md0.8/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.4_md0.8.jsonl results/metadata_as_query_td0.4_md0.8/train.relational.gpt-51.minimal.results.jsonl"
    "gpt-5.1 minimal data/synth/metadata_as_query/train_td0.8_md0.8.jsonl results/metadata_as_query_td0.8_md0.8/train.relational.gpt-51.minimal.results.jsonl"
    # "minimal data/synth/content_as_query/train_claude_key_passages.jsonl results/content_as_query-claude_key_passages/train.relational-minimal.results.jsonl"
    # "minimal data/synth/content_as_query/train_claude_keywords.jsonl results/content_as_query-claude_keywords/train.relational-minimal.results.jsonl"
    # "minimal data/synth/content_as_query/train_gpt_key_passages.jsonl results/content_as_query-gpt_key_passages/train.relational-minimal.results.jsonl"
    # "minimal data/synth/content_as_query/train_gpt_keywords.jsonl results/content_as_query-gpt_keywords/train.relational-minimal.results.jsonl"

    # "minimal db/test_retrieve_relational_v2.jsonl results/test_relational_v2/test_results_v2.jsonl"
)

echo "Text-to-SQL Retrieval | DB: $DB_NAME"

for args in "${style_query_output_list[@]}"; do
    read model system_prompt query_file output_file <<< "$args"
    echo "Processing: $model | $system_prompt | $query_file → $output_file"
    
    python $PROJECT_ROOT/db/retrieve_relational_v2.py \
        --db_name $DB_NAME \
        --db_user $DB_USER \
        --db_password $DB_PASSWORD \
        --db_host $DB_HOST \
        --db_port $DB_PORT \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --model $model \
        --system_prompt $system_prompt
done
