#!/bin/bash

# Create synthetic query-document pairs from papers_100.jsonl
# where query = lowercased paper title

python data/synth/title_as_query/create_data.py \
    data/raw/papers_100.jsonl \
    data/synth \
    0.8
