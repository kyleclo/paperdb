#!/bin/bash

# Create synthetic query-document pairs from 100-paper subsample
# where query = lowercased paper title

python data/synth/title_as_query/create_data.py \
    raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged-100.jsonl \
    data/synth/title_as_query
