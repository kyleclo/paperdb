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
MAX_TURNS=${MAX_TURNS:-3}

export CUDA_VISIBLE_DEVICES=1
index_query_output_list=(
    # title_as_query
    # "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/title_as_query/train.jsonl results/title_as_query/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"

    # "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/metadata_as_query/train_td0.8_md0.8.jsonl results/metadata_as_query_td0.8_md0.8/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"
    # "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"
    # "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/metadata_as_query/train_td0.0_md0.8.jsonl results/metadata_as_query_td0.0_md0.8/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"
    # "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/metadata_as_query/train_td0.8_md0.0.jsonl results/metadata_as_query_td0.8_md0.0/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"

    "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/content_as_query/train_gpt_key_passages.jsonl results/content_as_query_gpt_key_passages/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"
    "gpt-5.1 max-turns-3 data/index-para-abs-v2 data/synth/content_as_query/train_gpt_keywords.jsonl results/content_as_query_gpt_keywords/train.agent.gpt-51.max-turns-3.para-abs.results.jsonl"
)

echo "Agent Retrieval | DB: $DB_NAME"

for args in "${index_query_output_list[@]}"; do
    read model max_turns index_dir query_file output_file <<< "$args"
    # remove the "max-turns-" prefix
    max_turns=${max_turns#"max-turns-"}
    echo "Processing: $model | max turns: $max_turns | index: $index_dir | query: $query_file → $output_file"
    
    python $PROJECT_ROOT/db/retrieve_agent.py \
        --db_name $DB_NAME \
        --db_user $DB_USER \
        --db_password $DB_PASSWORD \
        --db_host $DB_HOST \
        --db_port $DB_PORT \
        --index_path $PROJECT_ROOT/$index_dir \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --model $model \
        --max_turns $max_turns
done
