# Synthetic Query Generation Strategies

This directory contains different approaches for generating synthetic query-document pairs from paper metadata and content.

## Strategies

### `title_as_query/`
Uses lowercased paper titles as queries. Simplest baseline that tests exact title matching.

### `metadata_as_query/`
Generates queries by randomly shuffling metadata fields: title, year, venue (canonical), author name, and author affiliation. Supports difficulty tuning via:
- **title_dropout** (td): Randomly drops words from title (0.0 = full title, 0.8 = 80% of words dropped)
- **metadata_dropout** (md): Randomly omits entire metadata fields like year/venue/author (0.0 = all fields included, 0.8 = 80% chance each field is dropped)

**Parameter sweep**: Generates 25 datasets with all combinations of td and md ∈ {0.0, 0.2, 0.4, 0.6, 0.8}

**Metadata fields used**:
- Title (always included, with word-level dropout)
- Year
- Venue (mapped to canonical names via `venue_to_id.json`: e.g., "NAACL 2016" → "NAACL", "BioNLP@ACL" → "ACL")
- First author name
- First author affiliation (from `author_affiliation_map.json`: e.g., "University of Virginia")

Simulates how users recall partial metadata when searching.

### `content_as_query/`
Extracts keyphrases and key passages from paper content (title, abstract, full text body) using LLMs. Uses GPT-4o-mini for extraction.

**Two extraction styles:**

#### Keywords Style
Extracts 3-5 keyphrases representing searchable concepts:
- Named entities (methods, datasets, algorithms, systems)
- Key concepts and technical terminology
- Research topics and domains

**Example query:**
```
adversarial attacks, data augmentation, textattack, nlp models
```

#### Key Passages Style
Extracts 1-2 distinctive sentences that capture unique contributions:
- Novel findings or results
- Specific methodological contributions
- Concrete examples and applications
- Distinctive features that distinguish the paper

**Example query:**
```
textattacks modular design enables researchers to easily construct attacks from combinations of novel and existing components facilitating the exploration of adversarial methods in nlp
```

**Output files:**
- `train_gpt_keywords.jsonl` - Short keyphrases (2-4 terms)
- `train_gpt_key_passages.jsonl` - Longer descriptive passages (1-2 sentences)
- `train_*_tokens.jsonl` - Token usage per query for cost analysis

Simulates users searching by research topics and distinctive findings rather than bibliographic metadata.

### `hybrid_as_query/`
Combines metadata and content queries by interleaving keywords from both sources. Targets 3-5 keywords per query with configurable metadata/content mixing ratio.

Simulates realistic queries that blend bibliographic and topical search terms.
