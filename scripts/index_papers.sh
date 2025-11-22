PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate

# index paper
python $PROJECT_ROOT/db/index_dense.py \
    --paper_file $PROJECT_ROOT/data/raw/papers_100.jsonl \
    --retrieval_units paragraphs abstracts title metadata \
    --output_dir $PROJECT_ROOT/data/index-all-units \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --batch_size 32
