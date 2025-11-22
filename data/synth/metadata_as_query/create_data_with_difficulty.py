import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import clean_query, textual_overlap


def create_synthetic_query(paper, venue_mappings=None, title_dropout=0.0, metadata_dropout=0.0):
    """
    Create a synthetic query by randomly shuffling paper metadata fields.

    Args:
        paper: Dictionary containing paper metadata
        venue_mappings: Optional dict mapping venue names to alternative names
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

    # Venue - with optional field dropout and alternative names
    if paper.get('venue') and random.random() > metadata_dropout:
        venue = paper['venue']
        if venue_mappings and venue in venue_mappings:
            # Only use the short alternatives, not the full original name
            venue = random.choice(venue_mappings[venue])
        fields.append(venue)

    # Field of study - with optional field dropout
    if paper.get('fieldsOfStudy') and len(paper['fieldsOfStudy']) > 0 and random.random() > metadata_dropout:
        fields.append(paper['fieldsOfStudy'][0])

    # First author name - with optional field dropout
    if paper.get('authors') and len(paper['authors']) > 0 and random.random() > metadata_dropout:
        fields.append(paper['authors'][0]['name'])

    # Randomly shuffle the fields
    random.shuffle(fields)

    # Join with whitespace
    return ' '.join(fields)


def create_dataset(input_path, output_path, venues_path, title_dropout=0.0, metadata_dropout=0.0, seed=42):
    """
    Create a synthetic query dataset with specified difficulty level.

    Args:
        input_path: Path to input papers file
        output_path: Path to output JSONL file
        venues_path: Path to venue mappings JSON
        title_dropout: Probability of dropping title words (higher = harder)
        metadata_dropout: Probability of dropping metadata fields (year, venue, etc.)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    input_path = Path(input_path)
    output_path = Path(output_path)
    venues_path = Path(venues_path)

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
        print("Warning: venues.json not found, using original venue names")

    # Process papers
    papers_processed = 0
    overlap_stats = []

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            paper = json.loads(line)

            # Create synthetic query
            query = create_synthetic_query(paper, venue_mappings, title_dropout, metadata_dropout)

            # Clean the query
            query = clean_query(query)

            # Calculate overlap with title for statistics
            if paper.get('title'):
                overlap_score = calculate_overlap_score(query, paper['title'])
                overlap_stats.append(overlap_score)

            # Create output record with required schema
            output_record = {
                'query': query,
                'paperId': paper['paperId'],
                'relevance': 1
            }

            # Write to output file
            outfile.write(json.dumps(output_record) + '\n')

            papers_processed += 1

            # Print first few examples
            if papers_processed <= 3:
                print(f"\nPaper {papers_processed}:")
                print(f"  paperId: {paper['paperId']}")
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
    # Paths
    input_path = Path(__file__).parent.parent.parent / 'raw' / 'papers_100.jsonl'
    output_dir = Path(__file__).parent
    venues_path = Path(__file__).parent.parent.parent / 'raw' / 'venues.json'

    # Create easy version (low dropout = high overlap with title, all metadata)
    title_dropout_easy = 0.0
    metadata_dropout_easy = 0.0
    print("=" * 70)
    print(f"CREATING DATASET (td={title_dropout_easy}, md={metadata_dropout_easy})")
    print("=" * 70)
    create_dataset(
        input_path=input_path,
        output_path=output_dir / f'train_td{title_dropout_easy}_md{metadata_dropout_easy}.jsonl',
        venues_path=venues_path,
        title_dropout=title_dropout_easy,
        metadata_dropout=metadata_dropout_easy,
        seed=42
    )

    print("\n\n")

    # Create hard version (high dropout = low overlap with title, missing metadata)
    title_dropout_hard = 0.7
    metadata_dropout_hard = 0.3
    print("=" * 70)
    print(f"CREATING DATASET (td={title_dropout_hard}, md={metadata_dropout_hard})")
    print("=" * 70)
    create_dataset(
        input_path=input_path,
        output_path=output_dir / f'train_td{title_dropout_hard}_md{metadata_dropout_hard}.jsonl',
        venues_path=venues_path,
        title_dropout=title_dropout_hard,
        metadata_dropout=metadata_dropout_hard,
        seed=42
    )

    print("\n\n")

    # Create medium version (moderate title dropout, moderate metadata dropout)
    title_dropout_medium = 0.4
    metadata_dropout_medium = 0.3
    print("=" * 70)
    print(f"CREATING DATASET (td={title_dropout_medium}, md={metadata_dropout_medium})")
    print("=" * 70)
    create_dataset(
        input_path=input_path,
        output_path=output_dir / f'train_td{title_dropout_medium}_md{metadata_dropout_medium}.jsonl',
        venues_path=venues_path,
        title_dropout=title_dropout_medium,
        metadata_dropout=metadata_dropout_medium,
        seed=42
    )

    print("\n\n")

    # Create hardest version (moderate title dropout, high metadata dropout)
    title_dropout_hardest = 0.4
    metadata_dropout_hardest = 0.7
    print("=" * 70)
    print(f"CREATING DATASET (td={title_dropout_hardest}, md={metadata_dropout_hardest})")
    print("=" * 70)
    create_dataset(
        input_path=input_path,
        output_path=output_dir / f'train_td{title_dropout_hardest}_md{metadata_dropout_hardest}.jsonl',
        venues_path=venues_path,
        title_dropout=title_dropout_hardest,
        metadata_dropout=metadata_dropout_hardest,
        seed=42
    )

    print("\n\n")

    # Create extreme version (high title dropout, high metadata dropout)
    title_dropout_extreme = 0.7
    metadata_dropout_extreme = 0.7
    print("=" * 70)
    print(f"CREATING DATASET (td={title_dropout_extreme}, md={metadata_dropout_extreme})")
    print("=" * 70)
    create_dataset(
        input_path=input_path,
        output_path=output_dir / f'train_td{title_dropout_extreme}_md{metadata_dropout_extreme}.jsonl',
        venues_path=venues_path,
        title_dropout=title_dropout_extreme,
        metadata_dropout=metadata_dropout_extreme,
        seed=42
    )


if __name__ == '__main__':
    main()
