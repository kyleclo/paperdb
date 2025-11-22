PROJECT_ROOT=$(git rev-parse --show-toplevel)

# uv virtual environment
source $PROJECT_ROOT/.venv/bin/activate

# Database credentials (set these as environment variables or modify here)
DB_NAME=${PAPERDB_NAME:-"paperdb"}
DB_USER=${PAPERDB_USER:-$USER}
DB_PASSWORD=${PAPERDB_PASSWORD:-"your_password"}
DB_HOST=${PAPERDB_HOST:-"localhost"}
DB_PORT=${PAPERDB_PORT:-5432}

# # Index dense retrieval (all units)
# echo "Building dense index with all units..."
# python $PROJECT_ROOT/db/index_dense.py \
#     --paper_file $PROJECT_ROOT/data/raw/papers_100.jsonl \
#     --retrieval_units paragraphs abstracts title metadata \
#     --output_dir $PROJECT_ROOT/data/index-all-units \
#     --model_name Qwen/Qwen3-Embedding-0.6B \
#     --batch_size 32

# # Index dense retrieval (paragraphs and abstracts only)
# echo "Building dense index with paragraphs and abstracts..."
# python $PROJECT_ROOT/db/index_dense.py \
#     --paper_file $PROJECT_ROOT/data/raw/papers_100.jsonl \
#     --retrieval_units paragraphs abstracts \
#     --output_dir $PROJECT_ROOT/data/index-para-abs \
#     --model_name Qwen/Qwen3-Embedding-0.6B \
#     --batch_size 32

# Index relational database
echo "Building relational database index..."
python $PROJECT_ROOT/db/index_relational.py \
    --paper_file $PROJECT_ROOT/data/raw/papers_100.jsonl \
    --db_name $DB_NAME \
    --db_user $DB_USER \
    --db_password $DB_PASSWORD \
    --db_host $DB_HOST \
    --db_port $DB_PORT
