import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import clean_query, textual_overlap


def create_synthetic_query(paper, venue_mappings=None, author_affiliation_map=None, title_dropout=0.0, metadata_dropout=0.0):
    """
    Create a synthetic query by randomly shuffling paper metadata fields.

    Args:
        paper: Dictionary containing paper metadata
        venue_mappings: Optional dict mapping venue names to canonical names
        author_affiliation_map: Optional dict mapping author IDs to affiliation data
        title_dropout: Probability of dropping each word from the title (0.0-1.0)
                      Higher values = more words dropped = harder queries
        metadata_dropout: Probability of dropping entire metadata fields (0.0-1.0)
                         Simulates users forgetting year, venue, etc.
                         Does NOT apply to title.

    Returns:
        String with whitespace-delimited shuffled metadata
    """
    # Extract metadata fields
    fields = []

    # Title - with optional word dropout (always included, never fully dropped)
    if paper.get('title'):
        title = paper['title']
        if title_dropout > 0:
            # Drop words randomly from title
            title_words = title.split()
            kept_words = [w for w in title_words if random.random() > title_dropout]
            # Ensure at least one word remains
            if kept_words:
                title = ' '.join(kept_words)
        fields.append(title)

    # Year - with optional field dropout
    if paper.get('year') and random.random() > metadata_dropout:
        fields.append(str(paper['year']))

    # Venue - with optional field dropout and canonical name mapping
    if paper.get('venue') and random.random() > metadata_dropout:
        venue = paper['venue']
        if venue_mappings and venue in venue_mappings:
            # Map to canonical venue name (e.g., "NAACL 2016" → "NAACL", "BioNLP@ACL" → "ACL")
            venue = venue_mappings[venue]
        fields.append(venue)

    # First author name - with optional field dropout
    if paper.get('authors') and len(paper['authors']) > 0 and random.random() > metadata_dropout:
        fields.append(paper['authors'][0]['name'])

    # First author affiliation - with optional field dropout
    if author_affiliation_map and paper.get('authors') and len(paper['authors']) > 0 and random.random() > metadata_dropout:
        author_id = paper['authors'][0].get('authorId')
        if author_id and author_id in author_affiliation_map:
            affil_data = author_affiliation_map[author_id]
            if affil_data.get('affiliations') and len(affil_data['affiliations']) > 0:
                # Use first affiliation's display name
                affil_name = affil_data['affiliations'][0].get('display_name')
                if affil_name:
                    fields.append(affil_name)

    # Clean each field individually
    cleaned_fields = [clean_query(field) for field in fields]

    # Randomly shuffle the fields
    random.shuffle(cleaned_fields)

    # Join with commas (like content queries)
    return ', '.join(cleaned_fields)


def create_dataset(input_path, output_path, venues_path, author_affiliation_path, title_dropout=0.0, metadata_dropout=0.0, seed=42):
    """
    Create a synthetic query dataset with specified difficulty level.

    Args:
        input_path: Path to input papers file
        output_path: Path to output JSONL file
        venues_path: Path to venue mappings JSON
        author_affiliation_path: Path to author affiliation map JSON
        title_dropout: Probability of dropping title words (higher = harder)
        metadata_dropout: Probability of dropping metadata fields (year, venue, etc.)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    input_path = Path(input_path)
    output_path = Path(output_path)
    venues_path = Path(venues_path)
    author_affiliation_path = Path(author_affiliation_path)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Title dropout: {title_dropout:.2f}")
    print(f"Metadata dropout: {metadata_dropout:.2f}")

    # Load venue mappings
    venue_mappings = {}
    if venues_path.exists():
        with open(venues_path, 'r') as f:
            venue_mappings = json.load(f)
        print(f"Loaded {len(venue_mappings)} venue mappings")
    else:
        print("Warning: venue_to_id.json not found, using original venue names")

    # Load author affiliation map
    author_affiliation_map = {}
    if author_affiliation_path.exists():
        with open(author_affiliation_path, 'r') as f:
            author_affiliation_map = json.load(f)
        print(f"Loaded {len(author_affiliation_map)} author affiliations")
    else:
        print("Warning: author_affiliation_map.json not found, skipping affiliation data")

    # Process papers
    papers_processed = 0
    overlap_stats = []

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            paper = json.loads(line)

            # Create synthetic query (already cleaned within the function)
            query = create_synthetic_query(paper, venue_mappings, author_affiliation_map, title_dropout, metadata_dropout)

            # Calculate overlap with title for statistics
            if paper.get('title'):
                overlap_score = calculate_overlap_score(query, paper['title'])
                overlap_stats.append(overlap_score)

            # Create output record with required schema
            output_record = {
                'query': query,
                'corpus_id': paper['corpusid'],
                'relevance': 1
            }

            # Write to output file
            outfile.write(json.dumps(output_record) + '\n')

            papers_processed += 1

            # Print first few examples
            if papers_processed <= 3:
                print(f"\nPaper {papers_processed}:")
                print(f"  corpus_id: {paper['corpusid']}")
                print(f"  Title: {paper.get('title', 'N/A')[:60]}...")
                print(f"  Query: {query[:100]}...")
                if overlap_stats:
                    print(f"  Overlap with title: {overlap_stats[-1]:.2%}")

    # Print statistics
    print(f"\nProcessed {papers_processed} papers successfully!")
    print(f"Output saved to: {output_path}")
    if overlap_stats:
        avg_overlap = sum(overlap_stats) / len(overlap_stats)
        print(f"\nOverlap Statistics:")
        print(f"  Mean overlap: {avg_overlap:.2%}")
        print(f"  Min overlap: {min(overlap_stats):.2%}")
        print(f"  Max overlap: {max(overlap_stats):.2%}")


def calculate_overlap_score(query, title):
    """
    Calculate the actual overlap score between query and title.

    Returns:
        Float between 0.0 and 1.0 representing the proportion of query words in title
    """
    query_clean = clean_query(query)
    title_clean = clean_query(title)

    query_words = query_clean.split()
    title_words = set(title_clean.split())

    if not query_words:
        return 0.0

    matched = sum(1 for word in query_words if word in title_words)
    return matched / len(query_words)


def main():
    # Paths (relative to script location, not CWD)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # data/synth/metadata_as_query -> data/synth -> data -> project_root
    input_path = project_root / 'raw' / 'dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged-100.jsonl'
    output_dir = script_dir
    venues_path = project_root / 'raw' / 'venue_to_id.json'
    author_affiliation_path = project_root / 'raw' / 'author_affiliation_map.json'

    # Sweep over dropout values
    dropout_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    total_combinations = len(dropout_values) * len(dropout_values)
    current = 0

    for td in dropout_values:
        for md in dropout_values:
            current += 1
            print("=" * 70)
            print(f"CREATING DATASET {current}/{total_combinations} (td={td}, md={md})")
            print("=" * 70)
            create_dataset(
                input_path=input_path,
                output_path=output_dir / f'train_td{td}_md{md}.jsonl',
                venues_path=venues_path,
                author_affiliation_path=author_affiliation_path,
                title_dropout=td,
                metadata_dropout=md,
                seed=42
            )
            print("\n")


if __name__ == '__main__':
    main()
