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
    print(f"Loaded {len(papers)} papers")
    return papers


def create_schema(conn):
    """Create database schema with proper normalization."""
    cursor = conn.cursor()
    
    # Drop existing tables if they exist (in reverse dependency order)
    print("Dropping existing tables if they exist...")
    cursor.execute("""
        DROP TABLE IF EXISTS PaperAuthors CASCADE;
        DROP TABLE IF EXISTS Authors CASCADE;
        DROP TABLE IF EXISTS Papers CASCADE;
    """)
    
    # Create Papers table
    print("Creating Papers table...")
    cursor.execute("""
        CREATE TABLE Papers (
            paper_id VARCHAR(255) PRIMARY KEY,
            corpus_id VARCHAR(255),
            title TEXT NOT NULL,
            abstract TEXT,
            venue VARCHAR(500),
            year INTEGER,
            publication_date VARCHAR(50),
            citation_count INTEGER DEFAULT 0,
            open_access_url TEXT,
            open_access_status VARCHAR(50),
            open_access_license VARCHAR(100)
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
    
    # Create PaperAuthors junction table (many-to-many)
    print("Creating PaperAuthors table...")
    cursor.execute("""
        CREATE TABLE PaperAuthors (
            paper_id VARCHAR(255) REFERENCES Papers(paper_id) ON DELETE CASCADE,
            author_id VARCHAR(255) REFERENCES Authors(author_id) ON DELETE CASCADE,
            author_position INTEGER NOT NULL,
            PRIMARY KEY (paper_id, author_id)
        );
    """)
    
    # Create indexes for better query performance
    print("Creating indexes...")
    cursor.execute("""
        CREATE INDEX idx_papers_year ON Papers(year);
        CREATE INDEX idx_papers_venue ON Papers(venue);
        CREATE INDEX idx_papers_citation_count ON Papers(citation_count);
        CREATE INDEX idx_authors_name ON Authors(name);
        CREATE INDEX idx_paper_authors_paper ON PaperAuthors(paper_id);
        CREATE INDEX idx_paper_authors_author ON PaperAuthors(author_id);
    """)
    
    conn.commit()
    print("Schema created successfully!")


def insert_papers(conn, papers: List[Dict[str, Any]]):
    """Insert papers and related data into database."""
    cursor = conn.cursor()
    
    # Track inserted authors to avoid duplicates
    authors_inserted = set()
    
    print("\nInserting papers and metadata...")
    for paper in tqdm(papers, desc="Processing papers"):
        paper_id = paper.get('paperId', paper.get('corpusId', ''))
        if not paper_id:
            continue
        
        # Extract open access info
        open_access = paper.get('openAccessPdf', {})
        if open_access is None:
            open_access = {}
        
        # Insert paper
        cursor.execute("""
            INSERT INTO Papers (
                paper_id, corpus_id, title, abstract, venue, year,
                publication_date, citation_count, open_access_url,
                open_access_status, open_access_license
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_id) DO NOTHING;
        """, (
            paper_id,
            paper.get('corpusId'),
            paper.get('title'),
            paper.get('abstract'),
            paper.get('venue'),
            paper.get('year'),
            paper.get('publicationDate'),
            paper.get('citationCount', 0),
            open_access.get('url'),
            open_access.get('status'),
            open_access.get('license')
        ))
        
        # Insert authors and paper-author relationships
        authors = paper.get('authors', [])
        for position, author in enumerate(authors):
            author_id = author.get('authorId')
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
            
            # Insert paper-author relationship
            cursor.execute("""
                INSERT INTO PaperAuthors (paper_id, author_id, author_position)
                VALUES (%s, %s, %s)
                ON CONFLICT (paper_id, author_id) DO NOTHING;
            """, (paper_id, author_id, position))
    
    conn.commit()
    print("Data inserted successfully!")


def print_statistics(conn):
    """Print database statistics."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("Database Statistics")
    print("="*60)
    
    # Count papers
    cursor.execute("SELECT COUNT(*) FROM Papers;")
    print(f"Total papers: {cursor.fetchone()[0]}")
    
    # Count authors
    cursor.execute("SELECT COUNT(*) FROM Authors;")
    print(f"Total authors: {cursor.fetchone()[0]}")
    
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
        print(f"  {year}: {count} papers")
    
    # Top venues
    cursor.execute("""
        SELECT venue, COUNT(*) as count
        FROM Papers
        WHERE venue IS NOT NULL AND venue != ''
        GROUP BY venue
        ORDER BY count DESC
        LIMIT 5;
    """)
    print("\nTop venues:")
    for venue, count in cursor.fetchall():
        print(f"  {venue}: {count} papers")
    
    # Most prolific authors
    cursor.execute("""
        SELECT a.name, COUNT(*) as paper_count
        FROM Authors a
        JOIN PaperAuthors pa ON a.author_id = pa.author_id
        GROUP BY a.name
        ORDER BY paper_count DESC
        LIMIT 5;
    """)
    print("\nMost prolific authors:")
    for author, count in cursor.fetchall():
        print(f"  {author}: {count} papers")
    
    print("="*60)


def build_index(paper_file: str, db_name: str, db_user: str, 
                db_password: str, db_host: str = 'localhost', 
                db_port: int = 5432):
    """Build relational database index."""
    
    # Load papers
    papers = load_papers(paper_file)
    
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
        insert_papers(conn, papers)
        
        # Print statistics
        print_statistics(conn)
        
        print("\n✓ Relational index built successfully!")
        
    except Exception as e:
        print(f"\nError building index: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build relational database index for papers"
    )
    parser.add_argument('--paper_file', type=str, required=True,
                       help='Path to JSONL file containing papers')
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
        args.db_name,
        args.db_user,
        args.db_password,
        args.db_host,
        args.db_port
    )


if __name__ == '__main__':
    main()

