PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate

# retrieve papers using dense retrieval
python $PROJECT_ROOT/db/retrieve_dense.py \
    --index_path $PROJECT_ROOT/data/index-all-units \
    --query_file $PROJECT_ROOT/data/synth/title_as_query/test.jsonl \
    --output_file $PROJECT_ROOT/results/title_as_query/dense.jsonl \
    --k 100

