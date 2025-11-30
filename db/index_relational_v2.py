import json
import argparse
import psycopg2
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def load_papers(paper_file: str) -> List[Dict[str, Any]]:
    """Load papers from JSONL file."""
    papers = []
    print(f"Loading papers from: {paper_file}")
    with open(paper_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f"Loaded {len(papers):,} papers")
    return papers


def load_mappings(author_map_file: str, venue_map_file: str):
    """Load author affiliation and venue normalization mappings."""
    print("\nLoading mappings...")

    # Load author affiliations
    author_map = {}
    if Path(author_map_file).exists():
        with open(author_map_file, 'r') as f:
            author_map = json.load(f)
        print(f"  Loaded {len(author_map):,} author affiliations")

    # Load venue normalization
    venue_map = {}
    if Path(venue_map_file).exists():
        with open(venue_map_file, 'r') as f:
            venue_map = json.load(f)
        print(f"  Loaded {len(venue_map):,} venue mappings")

    return author_map, venue_map


def create_schema(conn):
    """Create enhanced database schema with full text and affiliations."""
    cursor = conn.cursor()

    # Drop existing tables if they exist (in reverse dependency order)
    print("Dropping existing tables if they exist...")
    cursor.execute("""
        DROP TABLE IF EXISTS AuthorInstitutions CASCADE;
        DROP TABLE IF EXISTS Institutions CASCADE;
        DROP TABLE IF EXISTS PaperAuthors CASCADE;
        DROP TABLE IF EXISTS Authors CASCADE;
        DROP TABLE IF EXISTS Papers CASCADE;
    """)

    # Create Papers table (metadata only, no full text)
    print("Creating Papers table...")
    cursor.execute("""
        CREATE TABLE Papers (
            corpus_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,

            -- Venue information
            venue VARCHAR(500),
            venue_type VARCHAR(20),  -- 'main' or 'workshop'
            canonical_venue VARCHAR(500),  -- Normalized venue

            -- Temporal
            year INTEGER,
            month INTEGER,  -- 1-12

            -- Citation metrics
            citation_count INTEGER DEFAULT 0
        );
    """)

    # Create Authors table
    print("Creating Authors table...")
    cursor.execute("""
        CREATE TABLE Authors (
            author_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(500) NOT NULL
        );
    """)

    # Create PaperAuthors junction table
    print("Creating PaperAuthors table...")
    cursor.execute("""
        CREATE TABLE PaperAuthors (
            corpus_id INTEGER REFERENCES Papers(corpus_id) ON DELETE CASCADE,
            author_id VARCHAR(255) REFERENCES Authors(author_id) ON DELETE CASCADE,
            author_position INTEGER NOT NULL,
            PRIMARY KEY (corpus_id, author_id)
        );
    """)

    # Create Institutions table
    print("Creating Institutions table...")
    cursor.execute("""
        CREATE TABLE Institutions (
            institution_id VARCHAR(500) PRIMARY KEY,
            display_name TEXT NOT NULL,
            ror VARCHAR(255),
            country_code VARCHAR(10),
            institution_type VARCHAR(50)
        );
    """)

    # Create AuthorInstitutions junction table
    print("Creating AuthorInstitutions table...")
    cursor.execute("""
        CREATE TABLE AuthorInstitutions (
            author_id VARCHAR(255) REFERENCES Authors(author_id) ON DELETE CASCADE,
            institution_id VARCHAR(500) REFERENCES Institutions(institution_id) ON DELETE CASCADE,
            PRIMARY KEY (author_id, institution_id)
        );
    """)

    # Create indexes for better query performance (metadata only)
    print("Creating indexes...")
    cursor.execute("""
        -- Papers indexes
        CREATE INDEX idx_papers_year ON Papers(year);
        CREATE INDEX idx_papers_month ON Papers(month);
        CREATE INDEX idx_papers_canonical_venue ON Papers(canonical_venue);
        CREATE INDEX idx_papers_venue_type ON Papers(venue_type);
        CREATE INDEX idx_papers_citation_count ON Papers(citation_count);

        -- Authors indexes
        CREATE INDEX idx_authors_name ON Authors(name);

        -- Junction table indexes
        CREATE INDEX idx_paper_authors_corpus ON PaperAuthors(corpus_id);
        CREATE INDEX idx_paper_authors_author ON PaperAuthors(author_id);
        CREATE INDEX idx_author_institutions_author ON AuthorInstitutions(author_id);
        CREATE INDEX idx_author_institutions_institution ON AuthorInstitutions(institution_id);

        -- Institutions indexes
        CREATE INDEX idx_institutions_country ON Institutions(country_code);
        CREATE INDEX idx_institutions_type ON Institutions(institution_type);
    """)

    conn.commit()
    print("Schema created successfully!")


def insert_papers(conn, papers: List[Dict[str, Any]], author_map: Dict, venue_map: Dict):
    """Insert papers and related data into database."""
    cursor = conn.cursor()

    # Track inserted records to avoid duplicates
    authors_inserted = set()
    institutions_inserted = set()

    print("\nInserting papers and metadata...")
    for paper in tqdm(papers, desc="Processing papers"):
        corpus_id = paper.get('corpusid')
        if not corpus_id:
            continue

        # Get normalized venue
        venue = paper.get('venue', '')
        canonical_venue = venue_map.get(venue, venue) if venue else None

        # Parse month from publication_date
        month = None
        pubdate = paper.get('publicationdate')
        if pubdate and isinstance(pubdate, str):
            try:
                # Parse YYYY-MM-DD format
                parts = pubdate.split('T')[0].split('-')
                if len(parts) >= 2:
                    month = int(parts[1])
            except (ValueError, IndexError):
                pass

        # Insert paper (metadata only)
        cursor.execute("""
            INSERT INTO Papers (
                corpus_id, title, abstract,
                venue, venue_type, canonical_venue,
                year, month, citation_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (corpus_id) DO NOTHING;
        """, (
            corpus_id,
            paper.get('title'),
            paper.get('abstract'),
            venue,
            paper.get('venue_type'),
            canonical_venue,
            paper.get('year'),
            month,
            paper.get('citationcount', 0)
        ))

        # Insert authors and affiliations
        authors = paper.get('authors', [])
        for position, author in enumerate(authors):
            author_id = str(author.get('authorId', ''))
            author_name = author.get('name')

            if not author_id or not author_name:
                continue

            # Insert author if not already inserted
            if author_id not in authors_inserted:
                cursor.execute("""
                    INSERT INTO Authors (author_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (author_id) DO NOTHING;
                """, (author_id, author_name))
                authors_inserted.add(author_id)

                # Insert affiliations if available
                if author_id in author_map:
                    affil_data = author_map[author_id]
                    affiliations = affil_data.get('affiliations', [])

                    for affil in affiliations:
                        inst_id = affil.get('id')
                        if not inst_id:
                            continue

                        # Insert institution if not already inserted
                        if inst_id not in institutions_inserted:
                            cursor.execute("""
                                INSERT INTO Institutions (
                                    institution_id, display_name, ror,
                                    country_code, institution_type
                                ) VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (institution_id) DO NOTHING;
                            """, (
                                inst_id,
                                affil.get('display_name'),
                                affil.get('ror'),
                                affil.get('country_code'),
                                affil.get('type')
                            ))
                            institutions_inserted.add(inst_id)

                        # Insert author-institution relationship
                        cursor.execute("""
                            INSERT INTO AuthorInstitutions (author_id, institution_id)
                            VALUES (%s, %s)
                            ON CONFLICT (author_id, institution_id) DO NOTHING;
                        """, (author_id, inst_id))

            # Insert paper-author relationship
            cursor.execute("""
                INSERT INTO PaperAuthors (corpus_id, author_id, author_position)
                VALUES (%s, %s, %s)
                ON CONFLICT (corpus_id, author_id) DO NOTHING;
            """, (corpus_id, author_id, position))

    conn.commit()
    print("Data inserted successfully!")


def print_statistics(conn):
    """Print database statistics."""
    cursor = conn.cursor()

    print("\n" + "="*70)
    print("Database Statistics")
    print("="*70)

    # Count papers
    cursor.execute("SELECT COUNT(*) FROM Papers;")
    print(f"Total papers: {cursor.fetchone()[0]:,}")

    # Count by venue type
    cursor.execute("""
        SELECT venue_type, COUNT(*) as count
        FROM Papers
        WHERE venue_type IS NOT NULL
        GROUP BY venue_type
        ORDER BY count DESC;
    """)
    print("\nVenue type distribution:")
    for vtype, count in cursor.fetchall():
        print(f"  {vtype}: {count:,} papers")

    # Count papers with abstract
    cursor.execute("SELECT COUNT(*) FROM Papers WHERE abstract IS NOT NULL AND abstract != '';")
    print(f"\nPapers with abstract: {cursor.fetchone()[0]:,}")

    # Count authors
    cursor.execute("SELECT COUNT(*) FROM Authors;")
    print(f"\nTotal authors: {cursor.fetchone()[0]:,}")

    # Count institutions
    cursor.execute("SELECT COUNT(*) FROM Institutions;")
    inst_count = cursor.fetchone()[0]
    print(f"Total institutions: {inst_count:,}")

    # Count authors with affiliations
    cursor.execute("SELECT COUNT(DISTINCT author_id) FROM AuthorInstitutions;")
    authors_with_affil = cursor.fetchone()[0]
    print(f"Authors with affiliations: {authors_with_affil:,}")

    # Top institutions by author count
    if inst_count > 0:
        cursor.execute("""
            SELECT i.display_name, i.country_code, COUNT(*) as author_count
            FROM Institutions i
            JOIN AuthorInstitutions ai ON i.institution_id = ai.institution_id
            GROUP BY i.institution_id, i.display_name, i.country_code
            ORDER BY author_count DESC
            LIMIT 5;
        """)
        print("\nTop institutions by author count:")
        for inst_name, country, count in cursor.fetchall():
            print(f"  {inst_name} ({country}): {count} authors")

    # Top canonical venues
    cursor.execute("""
        SELECT canonical_venue, COUNT(*) as count
        FROM Papers
        WHERE canonical_venue IS NOT NULL AND canonical_venue != ''
        GROUP BY canonical_venue
        ORDER BY count DESC
        LIMIT 10;
    """)
    print("\nTop canonical venues:")
    for venue, count in cursor.fetchall():
        venue_short = venue[:60] + '...' if len(venue) > 60 else venue
        print(f"  {venue_short}: {count:,} papers")

    # Year distribution
    cursor.execute("""
        SELECT year, COUNT(*) as count
        FROM Papers
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year DESC
        LIMIT 10;
    """)
    print("\nYear distribution (top 10):")
    for year, count in cursor.fetchall():
        print(f"  {year}: {count:,} papers")

    # Most prolific authors
    cursor.execute("""
        SELECT a.name, a.author_id, COUNT(*) as paper_count
        FROM Authors a
        JOIN PaperAuthors pa ON a.author_id = pa.author_id
        GROUP BY a.author_id, a.name
        ORDER BY paper_count DESC
        LIMIT 5;
    """)
    print("\nMost prolific authors:")
    for author, author_id, count in cursor.fetchall():
        print(f"  {author} ({author_id}): {count:,} papers")

    print("="*70)


def build_index(paper_file: str, author_map_file: str, venue_map_file: str,
                db_name: str, db_user: str, db_password: str,
                db_host: str = 'localhost', db_port: int = 5432):
    """Build enhanced relational database index."""

    # Load papers and mappings
    papers = load_papers(paper_file)
    author_map, venue_map = load_mappings(author_map_file, venue_map_file)

    if not papers:
        print("No papers found!")
        return

    # Connect to PostgreSQL
    print(f"\nConnecting to PostgreSQL database: {db_name}")
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        print("Connected successfully!")
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print(f"You can create it with: createdb -U {db_user} {db_name}")
        return

    try:
        # Create schema
        create_schema(conn)

        # Insert data
        insert_papers(conn, papers, author_map, venue_map)

        # Print statistics
        print_statistics(conn)

        print("\n✓ Enhanced relational index built successfully!")

    except Exception as e:
        print(f"\nError building index: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build enhanced relational database index with full text and affiliations"
    )
    parser.add_argument('--paper_file', type=str, required=True,
                       help='Path to JSONL file containing papers')
    parser.add_argument('--author_map', type=str,
                       default='raw/author_affiliation_map.json',
                       help='Path to author affiliation mapping JSON')
    parser.add_argument('--venue_map', type=str,
                       default='raw/venue_to_id.json',
                       help='Path to venue normalization mapping JSON')
    parser.add_argument('--db_name', type=str, required=True,
                       help='PostgreSQL database name')
    parser.add_argument('--db_user', type=str, required=True,
                       help='PostgreSQL username')
    parser.add_argument('--db_password', type=str, required=True,
                       help='PostgreSQL password')
    parser.add_argument('--db_host', type=str, default='localhost',
                       help='PostgreSQL host (default: localhost)')
    parser.add_argument('--db_port', type=int, default=5432,
                       help='PostgreSQL port (default: 5432)')

    args = parser.parse_args()

    build_index(
        args.paper_file,
        args.author_map,
        args.venue_map,
        args.db_name,
        args.db_user,
        args.db_password,
        args.db_host,
        args.db_port
    )


if __name__ == '__main__':
    main()
