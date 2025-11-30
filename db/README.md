# Relational Database Schema (v2)

PostgreSQL database schema for DBLP papers with metadata and author affiliations.

**Note:** Full text is NOT stored in the database - only metadata. Full text is available in the source JSONLines file.

## Schema Overview

**5 tables:**
1. **Papers** - Core paper metadata (no full text)
2. **Authors** - Author information
3. **PaperAuthors** - Many-to-many: papers ↔ authors
4. **Institutions** - Institution/affiliation data from OpenAlex
5. **AuthorInstitutions** - Many-to-many: authors ↔ institutions

## Table Structures

### Papers
```sql
CREATE TABLE Papers (
    corpus_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,

    -- Venue information
    venue VARCHAR(500),
    venue_type VARCHAR(20),      -- 'main' or 'workshop'
    canonical_venue VARCHAR(500), -- Normalized (e.g., BioNLP@ACL → ACL)

    -- Temporal
    year INTEGER,
    month INTEGER,  -- 1-12

    -- Citation metrics
    citation_count INTEGER DEFAULT 0
);
```

**Field Coverage (22,371 papers):**
- `corpus_id`: 100% (primary key)
- `title`: 100%
- `abstract`: 97.2%
- `venue`: 100%
- `venue_type`: 100%
- `canonical_venue`: 100%
- `year`: 100%
- `month`: 95.7%
- `citation_count`: 100%

### Authors
```sql
CREATE TABLE Authors (
    author_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL
);
```

### PaperAuthors
```sql
CREATE TABLE PaperAuthors (
    corpus_id INTEGER REFERENCES Papers(corpus_id),
    author_id VARCHAR(255) REFERENCES Authors(author_id),
    author_position INTEGER NOT NULL,
    PRIMARY KEY (corpus_id, author_id)
);
```

### Institutions
```sql
CREATE TABLE Institutions (
    institution_id VARCHAR(500) PRIMARY KEY,
    display_name TEXT NOT NULL,
    ror VARCHAR(255),            -- Research Organization Registry ID
    country_code VARCHAR(10),
    institution_type VARCHAR(50) -- 'education', 'government', 'nonprofit', etc.
);
```

### AuthorInstitutions
```sql
CREATE TABLE AuthorInstitutions (
    author_id VARCHAR(255) REFERENCES Authors(author_id),
    institution_id VARCHAR(500) REFERENCES Institutions(institution_id),
    PRIMARY KEY (author_id, institution_id)
);
```

## Indexes

**Metadata indexes only (no full text search):**
- `year`, `month`, `canonical_venue`, `venue_type`, `citation_count`
- `author.name`, `institution.country_code`, `institution.institution_type`
- All foreign keys in junction tables

## Setup

### 1. Download Data from HuggingFace

```bash
# Install HuggingFace Hub
pip install huggingface-hub

# Download dataset files
huggingface-cli download kylel/dblp-papers-fulltext --repo-type dataset --local-dir raw/

# Or download individual files
wget https://huggingface.co/datasets/kylel/dblp-papers-fulltext/resolve/main/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl -O raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl
wget https://huggingface.co/datasets/kylel/dblp-papers-fulltext/resolve/main/author_affiliation_map.json -O raw/author_affiliation_map.json
wget https://huggingface.co/datasets/kylel/dblp-papers-fulltext/resolve/main/venue_to_id.json -O raw/venue_to_id.json
```

**Required files:**
- `dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl` (589 MB)
- `author_affiliation_map.json` (4.4 MB)
- `venue_to_id.json` (38 KB)

### 2. Install PostgreSQL
```bash
# macOS
brew install postgresql

# Start PostgreSQL
brew services start postgresql
```

### 3. Create Database
```bash
createdb paperdb
```

### 4. Run Indexing Script
```bash
python db/index_relational_v2.py \
  --paper_file raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl \
  --author_map raw/author_affiliation_map.json \
  --venue_map raw/venue_to_id.json \
  --db_name paperdb \
  --db_user $USER \
  --db_password your_password
```

**Time estimate:** ~2-3 minutes for 22,371 papers

## Example SQL Queries

### Basic Queries

#### Get papers from a specific conference
```sql
SELECT corpus_id, title, year, venue_type
FROM Papers
WHERE canonical_venue = 'ACL'
ORDER BY year DESC
LIMIT 10;
```

#### Count papers by venue type
```sql
SELECT venue_type, COUNT(*) as count
FROM Papers
GROUP BY venue_type;
```

#### Find papers from 2020
```sql
SELECT corpus_id, title, canonical_venue, month
FROM Papers
WHERE year = 2020
ORDER BY month, title;
```

### Author Queries

#### Find papers by author
```sql
SELECT p.title, p.year, p.canonical_venue
FROM Papers p
JOIN PaperAuthors pa ON p.corpus_id = pa.corpus_id
JOIN Authors a ON pa.author_id = a.author_id
WHERE a.name = 'Lillian Lee'
ORDER BY p.year DESC;
```

#### Most prolific authors
```sql
SELECT a.name, COUNT(*) as paper_count
FROM Authors a
JOIN PaperAuthors pa ON a.author_id = pa.author_id
GROUP BY a.author_id, a.name
ORDER BY paper_count DESC
LIMIT 20;
```

#### Find authors with their institutions
```sql
SELECT a.name, i.display_name, i.country_code
FROM Authors a
JOIN AuthorInstitutions ai ON a.author_id = ai.author_id
JOIN Institutions i ON ai.institution_id = i.institution_id
WHERE a.name LIKE '%Turney%';
```

### Institution Queries

#### Papers by institution country
```sql
SELECT p.title, p.year, i.display_name, i.country_code
FROM Papers p
JOIN PaperAuthors pa ON p.corpus_id = pa.corpus_id
JOIN AuthorInstitutions ai ON pa.author_id = ai.author_id
JOIN Institutions i ON ai.institution_id = i.institution_id
WHERE i.country_code = 'FR'
LIMIT 10;
```

#### Top institutions by paper count
```sql
SELECT i.display_name, i.country_code, COUNT(DISTINCT p.corpus_id) as paper_count
FROM Institutions i
JOIN AuthorInstitutions ai ON i.institution_id = ai.institution_id
JOIN PaperAuthors pa ON ai.author_id = pa.author_id
JOIN Papers p ON pa.corpus_id = p.corpus_id
GROUP BY i.institution_id, i.display_name, i.country_code
ORDER BY paper_count DESC
LIMIT 20;
```

#### Universities vs government labs
```sql
SELECT institution_type, COUNT(*) as count
FROM Institutions
GROUP BY institution_type
ORDER BY count DESC;
```

### Venue Queries

#### Main conference vs workshop papers by year
```sql
SELECT year, venue_type, COUNT(*) as count
FROM Papers
WHERE year >= 2015
GROUP BY year, venue_type
ORDER BY year DESC, venue_type;
```

#### Workshop distribution for ACL
```sql
SELECT venue, COUNT(*) as count
FROM Papers
WHERE canonical_venue = 'ACL' AND venue_type = 'workshop'
GROUP BY venue
ORDER BY count DESC
LIMIT 10;
```

### Citation Analysis

#### Highly cited papers from 2020
```sql
SELECT title, canonical_venue, citation_count
FROM Papers
WHERE year = 2020
ORDER BY citation_count DESC
LIMIT 20;
```

#### Average citations by venue
```sql
SELECT canonical_venue,
       COUNT(*) as paper_count,
       AVG(citation_count) as avg_citations,
       MAX(citation_count) as max_citations
FROM Papers
WHERE canonical_venue IN ('ACL', 'EMNLP', 'NAACL', 'COLING')
GROUP BY canonical_venue
ORDER BY avg_citations DESC;
```

### Collaboration Analysis

#### Co-authorship network
```sql
SELECT a1.name as author1, a2.name as author2, COUNT(*) as papers_together
FROM PaperAuthors pa1
JOIN PaperAuthors pa2 ON pa1.corpus_id = pa2.corpus_id
JOIN Authors a1 ON pa1.author_id = a1.author_id
JOIN Authors a2 ON pa2.author_id = a2.author_id
WHERE pa1.author_id < pa2.author_id  -- Avoid duplicates
GROUP BY a1.author_id, a1.name, a2.author_id, a2.name
HAVING COUNT(*) >= 3
ORDER BY papers_together DESC
LIMIT 20;
```

#### International collaborations
```sql
SELECT i1.country_code as country1,
       i2.country_code as country2,
       COUNT(DISTINCT p.corpus_id) as papers
FROM Papers p
JOIN PaperAuthors pa1 ON p.corpus_id = pa1.corpus_id
JOIN PaperAuthors pa2 ON p.corpus_id = pa2.corpus_id
JOIN AuthorInstitutions ai1 ON pa1.author_id = ai1.author_id
JOIN AuthorInstitutions ai2 ON pa2.author_id = ai2.author_id
JOIN Institutions i1 ON ai1.institution_id = i1.institution_id
JOIN Institutions i2 ON ai2.institution_id = i2.institution_id
WHERE i1.country_code < i2.country_code  -- Avoid duplicates
  AND i1.country_code IS NOT NULL
  AND i2.country_code IS NOT NULL
GROUP BY i1.country_code, i2.country_code
ORDER BY papers DESC
LIMIT 20;
```

### Abstract Coverage

#### Papers with abstracts by year
```sql
SELECT year,
       COUNT(*) as total_papers,
       SUM(CASE WHEN abstract IS NOT NULL THEN 1 ELSE 0 END) as with_abstract,
       ROUND(100.0 * SUM(CASE WHEN abstract IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as coverage_pct
FROM Papers
WHERE year >= 2015
GROUP BY year
ORDER BY year DESC;
```

## Connection Info

After indexing, you can connect with:

```bash
psql -d paperdb
```

Or in Python:
```python
import psycopg2

conn = psycopg2.connect(
    dbname="paperdb",
    user="your_user",
    password="your_password"
)

cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM Papers;")
print(f"Total papers: {cursor.fetchone()[0]:,}")
```

## Notes

- **Full text NOT stored in DB** - only metadata (title, abstract, venue, etc.)
- For full text, query the source JSONLines file by `corpus_id`
- Workshops automatically mapped to main conferences via `canonical_venue`
- Only 35.9% of papers have affiliation data (OpenAlex coverage limitation)
- Author-institution matching is approximate (by position and name)
