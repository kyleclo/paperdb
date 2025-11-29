# DBLP Papers with Full Text + Enrichments

Recent open access NLP/ML/AI papers from DBLP, enriched with full text from Vespa and metadata from OpenAlex.

## Datasets

### 1. Papers with Full Text (Tagged)
**File**: `dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl` (589 MB)

- **Total papers**: 22,371
- **Main conference papers**: 18,804 (84.1%)
- **Workshop papers**: 3,567 (15.9%)
- **With abstract**: 21,743 (97.2%)
- **With full text body**: 21,662 (96.8%)
- **Format**: JSONLines (one JSON object per line)
- **New field**: `venue_type` ("main" or "workshop")

### 2. Author Affiliation Mapping
**File**: `author_affiliation_map.json` (4.4 MB)

- **Total authors**: 13,830
- **Coverage**: 35.9% of papers have affiliation data
- **Source**: OpenAlex API

### 3. Venue Normalization Mapping
**Files**: `venue_to_id.json`, `venue_stats.json`

- **Unique venue strings**: 428
- **Canonical venues**: 223 (workshops mapped to main conferences)
- **Workshops mapped**: 177
- **Top venue**: ACL (1,615 papers including 59 workshops)

---

## Creation Pipeline

### Step 1: Filter DBLP for Recent Open Access Papers
**Script**: `2_create_oa_recent_subset.py`

Started with DBLP NLP/ML/AI subset (140,976 papers), filtered to:
- Open access only (`isopenaccess = true`)
- Recent papers (2005+)
- Result: 31,442 papers

### Step 2: Fetch Full Text from Vespa
**Script**: `3_fetch_fulltext_from_vespa.py`

- Queried internal Vespa API for paper snippets
- Reconstructed full text from snippets (title, abstract, body)
- **Success rate**: 71.2% (22,371 papers with full text)
- **Body text coverage**: 96.8% of successful papers
- **Time**: ~52 minutes (10 requests/sec, 100ms rate limit)

### Step 3: Convert to JSONLines
**Script**: `4_convert_to_jsonlines.py`

Converted HuggingFace dataset format to flat JSONLines for easier processing.

### Step 4: Build Author Affiliation Map
**Script**: `5_build_author_affiliation_map.py`

- Queried OpenAlex API using DOIs from papers
- Matched OpenAlex authors to S2 authors by position and name
- **Batch mode**: 50 papers per request
- **Success rate**: 35.9% of papers found in OpenAlex
- **Authors mapped**: 13,830 unique authors
- **Time**: ~5 minutes (batch queries)

### Step 5: Build Venue Mapping
**Script**: `6_build_venue_mapping.py`

- Extracted all unique venue strings
- Mapped workshops to main conferences (e.g., BioNLP@ACL → ACL)
- Normalized variants (removed years, merged similar names)
- **Reduction**: 428 strings → 223 canonical venues (47.9%)
- **Workshops mapped**: 177 (e.g., 59 ACL workshops, 28 EMNLP workshops)

### Step 6: Add Venue Type Tags
**Script**: `7_add_venue_type_tag.py`

- Added `venue_type` field to all papers ("main" or "workshop")
- Detected workshops using `@` pattern in venue string
- **Main papers**: 18,804 (84.1%)
- **Workshop papers**: 3,567 (15.9%)

---

## Example JSONs

### Paper with Full Text
```json
{
  "corpusid": 294,
  "title": "Using interval particle filtering for marker less 3D human motion capture",
  "abstract": "In this paper we present a new approach for marker less human motion capture...",

  "fulltext_title": "Using interval particle filtering for marker less 3D human...",
  "fulltext_abstract": "In this paper we present a new approach...",
  "fulltext_body": "1. Introduction\nHuman motion capture has been...",
  "snippet_count": 2,

  "authors": [
    {"authorId": "1716686", "name": "J. Saboune"},
    {"authorId": "1731714", "name": "F. Charpillet"}
  ],

  "venue": "17th IEEE International Conference on Tools with Artificial Intelligence (ICTAI'05)",
  "venue_type": "main",
  "year": 2005,

  "externalids": {
    "DOI": "10.1109/ICTAI.2005.127",
    "ArXiv": "cs/0510062",
    "CorpusId": "294"
  },

  "citationcount": 29,
  "referencecount": 11,
  "isopenaccess": true,

  "s2fieldsofstudy": [
    {"category": "Computer Science", "source": "s2-fos-model"}
  ]
}
```

### Author Affiliation Entry
```json
{
  "1716686": {
    "name": "J. Saboune",
    "affiliations": [
      {
        "id": "https://openalex.org/I1326498283",
        "display_name": "Institut national de recherche en sciences et technologies du numérique",
        "ror": "https://ror.org/02kvxyf05",
        "country_code": "FR",
        "type": "government"
      },
      {
        "id": "https://openalex.org/I4210121838",
        "display_name": "Laboratoire Lorrain de Recherche en Informatique et ses Applications",
        "ror": "https://ror.org/02vnf0c38",
        "country_code": "FR",
        "type": "facility"
      }
    ]
  }
}
```

### Venue Mapping Entry
```json
{
  "NAACL": "NAACL",
  "NAACL 2016": "NAACL",
  "BioNLP@ACL": "ACL",
  "SemEval@ACL": "ACL",
  "WMT@EMNLP": "EMNLP"
}
```

---

## Usage

### Load Papers with Full Text
```python
import json

# Load JSONLines
papers = []
with open('raw/dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl', 'r') as f:
    for line in f:
        papers.append(json.loads(line))

# Filter by venue type
main_papers = [p for p in papers if p.get('venue_type') == 'main']
workshop_papers = [p for p in papers if p.get('venue_type') == 'workshop']

print(f"Main conference papers: {len(main_papers):,}")
print(f"Workshop papers: {len(workshop_papers):,}")

# Access full text
for paper in papers[:5]:
    print(f"Title: {paper['title']}")
    print(f"Venue type: {paper['venue_type']}")
    print(f"Body length: {len(paper['fulltext_body'])} chars")
    print()
```

### Look Up Author Affiliations
```python
import json

# Load author mapping
with open('raw/author_affiliation_map.json', 'r') as f:
    author_map = json.load(f)

# Look up author
author_id = "1716686"
if author_id in author_map:
    affil = author_map[author_id]
    print(f"Author: {affil['name']}")
    for inst in affil['affiliations']:
        print(f"  - {inst['display_name']} ({inst['country_code']})")
```

### Normalize Venue Names
```python
import json

# Load venue mapping
with open('raw/venue_to_id.json', 'r') as f:
    venue_map = json.load(f)

# Normalize venue strings
print(venue_map.get("NAACL 2016"))        # → "NAACL"
print(venue_map.get("BioNLP@ACL"))        # → "ACL"
print(venue_map.get("WASSA@EMNLP"))       # → "EMNLP"
```

---

## Statistics

### Full Text Coverage
- Papers with `abstract`: 21,743 (97.2%)
- Papers with non-empty `fulltext_title`: ~100%
- Papers with non-empty `fulltext_abstract`: ~98%
- Papers with non-empty `fulltext_body`: 21,662 (96.8%)

### Venue Type Distribution
- Main conference papers: 18,804 (84.1%)
- Workshop papers: 3,567 (15.9%)

### Author Affiliation Coverage
- Papers found in OpenAlex: 8,038 / 22,371 (35.9%)
- Authors with affiliations: 13,830
- Average affiliations per author: ~1.3

### Top 10 Venues (with workshops included)
1. NAACL (1,708 papers - 2 variants)
2. ACL (1,615 papers - 59 workshops + variants)
3. *SEMEVAL (1,082 papers)
4. EMNLP (752 papers - 28 workshops + variants)
5. CoNLL (659 papers - 2 variants)
6. EACL (658 papers)
7. TACL (286 papers)
8. Machine Learning (226 papers)
9. COLING (197 papers - 18 workshops + variants)
10. HLT (188 papers - 9 variants)

Note: Main conferences now include their associated workshops (e.g., ACL includes BioNLP@ACL, SemEval@ACL, etc.)

### Institution Types
From OpenAlex affiliation data:
- Education institutions (universities)
- Government research labs
- Nonprofit research organizations
- Commercial facilities

---

## Files

### Scripts
- `1_create_nlp_ml_ai_subset_fast.py` - Filter DBLP by field
- `2_create_oa_recent_subset.py` - Filter for open access + recent
- `3_fetch_fulltext_from_vespa.py` - Fetch full text from Vespa API
- `4_convert_to_jsonlines.py` - Convert to JSONLines format
- `5_build_author_affiliation_map.py` - Build author→affiliation map (OpenAlex)
- `6_build_venue_mapping.py` - Build venue normalization map
- `7_add_venue_type_tag.py` - Add main/workshop tags to papers

### Datasets
- `dblp-nlp-ml-ai-oa-recent-subset/` - HuggingFace dataset (31,442 papers)
- `dblp-nlp-ml-ai-oa-recent-with-fulltext/` - With full text (22,371 papers)
- `dblp-nlp-ml-ai-oa-recent-with-fulltext.jsonl` - JSONLines export (589 MB)
- `dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged.jsonl` - **Final dataset** with venue_type tags (589 MB)

### Mappings
- `author_affiliation_map.json` - S2 author ID → affiliations (13,830 authors)
- `venue_to_id.json` - Venue string → canonical venue (428 → 223 mappings)
- `venue_stats.json` - Canonical venue statistics with variants (223 venues)


---

## API Details

### Vespa API (Full Text)
- **Endpoint**: Internal Vespa endpoint
- **Query**: By Semantic Scholar corpus ID
- **Returns**: Snippets (title, abstract, body sections)
- **Rate limit**: 10 requests/sec (100ms delay)

### OpenAlex API (Affiliations)
- **Endpoint**: `https://api.openalex.org/works`
- **Query**: By DOI using batch filter `doi:DOI1|DOI2|...`
- **Batch size**: 50 papers per request
- **Rate limit**: 10 requests/sec (polite pool with email)
- **Returns**: Authorships with institution data (id, name, ROR, country, type)

---

## Notes

- **Full text quality**: Body text reconstructed from snippets may have ordering artifacts
- **Affiliation matching**: OpenAlex authors matched to S2 authors by position and last name
- **Venue normalization**:
  - Workshops automatically mapped to main conferences (e.g., BioNLP@ACL → ACL)
  - Year variants merged (e.g., NAACL 2016 → NAACL)
  - 177 workshops collapsed into main conferences
  - 47.9% reduction in unique venues (428 → 223)
- **Missing data**: ~29% of papers failed full text retrieval (likely not in Vespa)
- **OpenAlex coverage**: Only 35.9% of papers found (limited by DOI availability and OpenAlex coverage)
