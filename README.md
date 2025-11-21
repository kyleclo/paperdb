# paperdb

A simple, lightweight project using LMs + SQL dbs for re-finding research papers

### structure

```
|-- data/
    |-- raw/
        |-- papers.jsonl    # dump of 1000 papers
    |-- synth/
        |-- train.jsonl     # gold (query, document) pairs
        |-- test.jsonl      # same as `train.jsonl` but a held-out test set
|-- db/
    |-- retrieval/          # dense retrieval method, indexing and retrieving papers given query
    |-- sql/                # sql retrieval method, indexing and retrieving papers given query
    |-- index.py            # main script used for indexing all the papers using both methods
    |-- retrieve.py         # main script used for retrieving papers using the indexes. outputs a `results.jsonl` file
|-- eval/
    |-- score.py            # implements eval metrics. takes as input `train|test.jsonl` and `results.jsonl` file to produce a score.
```