#!/bin/bash

set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate


index_query_output_list=(
    # # metadata_as_query 
    # "data/index-all-units data/synth/title_as_query/train.jsonl results/title_as_query/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/title_as_query/train.jsonl results/title_as_query/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query 
    # "data/index-all-units data/synth/metadata_as_query/train.jsonl results/metadata_as_query/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train.jsonl results/metadata_as_query/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query/train_td0.0_md0.0
    # "data/index-all-units data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train_td0.0_md0.0.jsonl results/metadata_as_query_td0.0_md0.0/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query/train_td0.4_md0.3
    # "data/index-all-units data/synth/metadata_as_query/train_td0.4_md0.3.jsonl results/metadata_as_query_td0.4_md0.3/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train_td0.4_md0.3.jsonl results/metadata_as_query_td0.4_md0.3/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query/train_td0.4_md0.7
    # "data/index-all-units data/synth/metadata_as_query/train_td0.4_md0.7.jsonl results/metadata_as_query_td0.4_md0.7/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train_td0.4_md0.7.jsonl results/metadata_as_query_td0.4_md0.7/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query/train_td0.7_md0.3
    # "data/index-all-units data/synth/metadata_as_query/train_td0.7_md0.3.jsonl results/metadata_as_query_td0.7_md0.3/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train_td0.7_md0.3.jsonl results/metadata_as_query_td0.7_md0.3/train.dense-para-abs.results.jsonl"
    
    # # metadata_as_query/train_td0.7_md0.7
    # "data/index-all-units data/synth/metadata_as_query/train_td0.7_md0.7.jsonl results/metadata_as_query_td0.7_md0.7/train.dense-all-units.results.jsonl"
    # "data/index-para-abs data/synth/metadata_as_query/train_td0.7_md0.7.jsonl results/metadata_as_query_td0.7_md0.7/train.dense-para-abs.results.jsonl"

    # /mmfs1/home/chentong/ws/cse544/project/paperdb/data/synth/content_as_query/train_claude_key_passages.jsonl
    "data/index-all-units data/synth/content_as_query/train_claude_key_passages.jsonl results/content_as_query-claude_key_passages/train.dense-all-units.results.jsonl"
    "data/index-para-abs data/synth/content_as_query/train_claude_key_passages.jsonl results/content_as_query-claude_key_passages/train.dense-para-abs.results.jsonl"

    # /mmfs1/home/chentong/ws/cse544/project/paperdb/data/synth/content_as_query/train_claude_keywords.jsonl
    "data/index-all-units data/synth/content_as_query/train_claude_keywords.jsonl results/content_as_query-claude_keywords/train.dense-all-units.results.jsonl"
    "data/index-para-abs data/synth/content_as_query/train_claude_keywords.jsonl results/content_as_query-claude_keywords/train.dense-para-abs.results.jsonl"

    # /mmfs1/home/chentong/ws/cse544/project/paperdb/data/synth/content_as_query/train_gpt_key_passages.jsonl
    "data/index-all-units data/synth/content_as_query/train_gpt_key_passages.jsonl results/content_as_query-gpt_key_passages/train.dense-all-units.results.jsonl"
    "data/index-para-abs data/synth/content_as_query/train_gpt_key_passages.jsonl results/content_as_query-gpt_key_passages/train.dense-para-abs.results.jsonl"

    # /mmfs1/home/chentong/ws/cse544/project/paperdb/data/synth/content_as_query/train_gpt_keywords.jsonl
    "data/index-all-units data/synth/content_as_query/train_gpt_keywords.jsonl results/content_as_query-gpt_keywords/train.dense-all-units.results.jsonl"
    "data/index-para-abs data/synth/content_as_query/train_gpt_keywords.jsonl results/content_as_query-gpt_keywords/train.dense-para-abs.results.jsonl"
)

# retrieve papers using dense retrieval
# for loop and parse query_file and output_file

for args in "${index_query_output_list[@]}"; do
    index_dir=$(echo $args | awk '{print $1}')  
    query_file=$(echo $args | awk '{print $2}')
    output_file=$(echo $args | awk '{print $3}')
    
    python $PROJECT_ROOT/db/retrieve_dense.py \
        --index_path $PROJECT_ROOT/$index_dir \
        --query_file $PROJECT_ROOT/$query_file \
        --output_file $PROJECT_ROOT/$output_file \
        --k 100
done
