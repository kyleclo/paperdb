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

Each paper in `data/raw/papers_100.jsonl` contains S2 metadata + full paragraph text:

```json
{
  "corpusId": "211560654",
  "paperId": "421eddd6a2c771e87e80eb64fb1328de2db51479",
  "title": "3D-Auth: Two-Factor Authentication with Personalized 3D-Printed Items",
  "authors": [
    {"authorId": "3399279", "name": "Karola Marky"},
    {"authorId": "2057761601", "name": "Martin Schmitz"}
  ],
  "abstract": "Two-factor authentication is a widely recommended security mechanism...",
  "year": 2020,
  "venue": "International Conference on Human Factors in Computing Systems",
  "citationCount": 36,
  "publicationDate": "2020-04-21",
  "publicationTypes": ["JournalArticle", "Book", "Conference"],
  "fieldsOfStudy": ["Computer Science"],
  "paragraphCount": 8,
  "paragraphs": [
    {
      "paragraphId": "34028",
      "sectionTitle": "INTRODUCTION",
      "text": "In two-factor authentication, two of the following authentication factors...",
      "spans": "[{\"corpusId\": 23796197, \"span\": \"[48,\"...}]",
      "conference": "chi",
      "year": 2020,
      "likelyRelatedWorkSection": false,
      "refCount": 4
    }
  ]
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