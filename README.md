# paperdb

A simple, lightweight project using LMs + SQL dbs for re-finding research papers

### setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
export ANTHROPIC_API_KEY=your_key  # only needed for SQL method
python db/index.py
python db/retrieve.py dense test
python eval/score.py
```

### data format

Each paper in `data/raw/papers.jsonl` contains:

```json
{
  "corpusId": "256697200",
  "paperId": "8e80efd4c95caa65404b9c273d179f0769c7350c",
  "title": "MetaExplorer : Facilitating Reasoning with Epistemic Uncertainty in Meta-analysis",
  "authors": [
    {
      "authorId": "144946284",
      "name": "Alex Kale"
    },
    {
      "authorId": "2205162540",
      "name": "Sarah Lee"
    }
  ],
  "abstract": "Scientists often use meta-analysis to characterize the impact of an intervention...",
  "year": 2023,
  "venue": "International Conference on Human Factors in Computing Systems",
  "citationCount": 4,
  "publicationDate": "2023-04-23",
  "publicationTypes": ["JournalArticle", "Conference"],
  "fieldsOfStudy": ["Computer Science"]
}
```

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