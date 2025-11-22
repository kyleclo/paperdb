import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import clean_query


def create_synthetic_query(paper, venue_mappings=None):
    """
    Create a synthetic query by randomly shuffling paper metadata fields.

    Args:
        paper: Dictionary containing paper metadata
        venue_mappings: Optional dict mapping venue names to alternative names

    Returns:
        String with whitespace-delimited shuffled metadata
    """
    # Extract metadata fields
    fields = []

    # Title
    if paper.get('title'):
        fields.append(paper['title'])

    # Year
    if paper.get('year'):
        fields.append(str(paper['year']))

    # Venue - randomly choose a short alternative name if available
    if paper.get('venue'):
        venue = paper['venue']
        if venue_mappings and venue in venue_mappings:
            # Only use the short alternatives, not the full original name
            venue = random.choice(venue_mappings[venue])
        fields.append(venue)

    # Field of study (take the first one if multiple exist)
    if paper.get('fieldsOfStudy') and len(paper['fieldsOfStudy']) > 0:
        fields.append(paper['fieldsOfStudy'][0])

    # First author name
    if paper.get('authors') and len(paper['authors']) > 0:
        fields.append(paper['authors'][0]['name'])

    # Randomly shuffle the fields
    random.shuffle(fields)

    # Join with whitespace
    return ' '.join(fields)


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Input and output paths
    input_path = Path(__file__).parent.parent.parent / 'raw' / 'papers_100.jsonl'
    output_path = Path(__file__).parent / 'train.jsonl'
    venues_path = Path(__file__).parent.parent.parent / 'raw' / 'venues.json'

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

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

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            paper = json.loads(line)

            # Create synthetic query
            query = create_synthetic_query(paper, venue_mappings)

            # Clean the query
            query = clean_query(query)

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
            if papers_processed <= 5:
                print(f"\nPaper {papers_processed}:")
                print(f"  paperId: {paper['paperId']}")
                print(f"  Query: {query[:100]}...")

    print(f"\nProcessed {papers_processed} papers successfully!")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
