import json
import argparse
import psycopg2
from pathlib import Path
from typing import List, Dict, Any
import openai
import os
import re
from async_completion import batch_chat_complete


def connect_db(db_name: str, db_user: str, db_password: str, 
               db_host: str = 'localhost', db_port: int = 5432):
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        return None


def get_database_schema() -> str:
    """Get database schema information."""
    return """
DATABASE SCHEMA:

Table: Papers
- paper_id (VARCHAR, PRIMARY KEY): Unique paper identifier
- corpus_id (VARCHAR): Corpus identifier
- title (TEXT, NOT NULL): Paper title
- abstract (TEXT): Paper abstract
- venue (VARCHAR): Publication venue (conference/journal name)
- year (INTEGER): Publication year
- publication_date (VARCHAR): Full publication date
- citation_count (INTEGER): Number of citations
- open_access_url (TEXT): URL to open access PDF
- open_access_status (VARCHAR): Open access status
- open_access_license (VARCHAR): License type

Table: Authors
- author_id (VARCHAR, PRIMARY KEY): Unique author identifier
- name (VARCHAR, NOT NULL): Author name

Table: PaperAuthors (Junction table for many-to-many relationship)
- paper_id (VARCHAR, FOREIGN KEY → Papers.paper_id): Reference to paper
- author_id (VARCHAR, FOREIGN KEY → Authors.author_id): Reference to author
- author_position (INTEGER): Author position in paper (0 = first author)
- PRIMARY KEY: (paper_id, author_id)

RELATIONSHIPS:
- Papers ←→ Authors (many-to-many through PaperAuthors)
- To get papers with their authors: JOIN Papers with PaperAuthors with Authors
- To filter by author: JOIN through PaperAuthors and filter on Authors.name
- author_position indicates author order (0 is first author, 1 is second, etc.)

INDEXES:
- idx_papers_year on Papers(year)
- idx_papers_venue on Papers(venue)
- idx_papers_citation_count on Papers(citation_count)
- idx_authors_name on Authors(name)
"""


def prepare_messages(user_query: str, schema: str, prompt_type: str = 'detailed') -> List[Dict[str, str]]:
    """Prepare messages for OpenAI API."""
    
    if prompt_type == 'detailed':
        system_prompt = """You are a SQL expert helping researchers find academic papers in a database.

Given a database schema and a natural language query (which might be conversational or informal), generate a PostgreSQL SQL query that retrieves the relevant papers.

Key requirements:
- Return ONLY the SQL query, no explanations or markdown formatting
- Always select at least the paper_id column
- Use ILIKE for case-insensitive partial text matching
- Use DISTINCT when joining with PaperAuthors to avoid duplicate papers
- Order by relevance (typically citation_count DESC, year DESC)
- Limit to 100 results maximum
- Do not end with semicolon

The database has Papers, Authors, and PaperAuthors tables. Use JOINs appropriately."""
    elif prompt_type == 'minimal':
        system_prompt = "Generate a PostgreSQL SQL query based on the database schema and user query."
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'minimal' or 'detailed'")

    user_prompt = f"""{schema}

Researcher's query: {user_query}

Generate a PostgreSQL SQL query to find the relevant papers."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def process_api_response(response, log_first: bool = False) -> Dict[str, Any]:
    """Process OpenAI API response and extract SQL."""
    try:
        # Extract and clean SQL query
        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r'^```(sql)?\s*', '', sql_query)
        sql_query = re.sub(r'\s*```$', '', sql_query).rstrip(';').strip()
        
        # Build result
        result = {
            'sql': sql_query,
            'input_tokens': response.usage.prompt_tokens if response.usage else 0,
            'output_tokens': response.usage.completion_tokens if response.usage else 0
        }
        
        # Add reasoning if available (for o1/o3 models)
        if hasattr(response.choices[0].message, 'reasoning_content'):
            result['reasoning'] = response.choices[0].message.reasoning_content
        
        # Log first response for debugging
        if log_first:
            result['raw_response'] = {
                'model': response.model,
                'id': response.id,
                'finish_reason': response.choices[0].finish_reason
            }
            print(f"\n{'='*60}")
            print(f"DEBUG: First API Response")
            print(f"{'='*60}")
            print(f"Model: {response.model}")
            print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
            print(f"SQL: {sql_query}")
            if result.get('reasoning'):
                print(f"Reasoning: {result['reasoning'][:200]}...")
            print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        return {'sql': None, 'input_tokens': 0, 'output_tokens': 0, 'error': str(e)}


def execute_sql_query(conn, sql_query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results."""
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch results
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        results = []
        for row in rows:
            result = {}
            for i, col_name in enumerate(column_names):
                result[col_name] = row[i]
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None


def execute_query_with_sql(conn, user_query: str, llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SQL query and return results."""
    # Base result with LLM metadata
    result = {
        'query': user_query,
        'sql': llm_result.get('sql'),
        **{k: v for k, v in llm_result.items() if k not in ['sql']},
        'paper_ids': [],
        'count': 0
    }
    
    # Early return if SQL generation failed
    if not result['sql']:
        result.setdefault('error', 'Failed to generate SQL query')
        return result
    
    # Execute SQL query
    query_results = execute_sql_query(conn, result['sql'])
    
    if query_results is None:
        result['error'] = 'Failed to execute SQL query'
        return result
    
    # Extract paper IDs
    result['paper_ids'] = [r['paper_id'] for r in query_results if 'paper_id' in r]
    result['count'] = len(result['paper_ids'])
    
    return result


def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """Load queries from JSONL file."""
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve papers using text-to-SQL with LLM"
    )
    
    # Database parameters
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
    
    # Query parameters
    parser.add_argument('--query_file', type=str, required=True,
                       help='Path to query file (JSONL format with "query" and "paperId" fields)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output file (JSONL format)')
    
    # OpenAI parameters
    parser.add_argument('--api_key', type=str,
                       help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--model', type=str, default='gpt-5.1',
                       help='OpenAI model to use (default: gpt-5.1)')
    parser.add_argument('--system_prompt', type=str, choices=['minimal', 'detailed'], default='detailed',
                       help='System prompt type: "minimal" or "detailed" (full instructions) (default: detailed)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided. Set --api_key or OPENAI_API_KEY environment variable.")
        return
    
    # Connect to database
    print(f"Connecting to database: {args.db_name}")
    conn = connect_db(args.db_name, args.db_user, args.db_password,
                     args.db_host, args.db_port)
    if not conn:
        return
    
    try:
        # Load queries
        print(f"\nLoading queries from: {args.query_file}")
        queries = load_queries(args.query_file)
        print(f"Loaded {len(queries)} queries")
        
        # Get database schema
        schema = get_database_schema()
        
        # Prepare all messages for batch processing
        print(f"Preparing messages for batch API calls (system prompt: {args.system_prompt})...")
        messages_list = [prepare_messages(q.get('query', ''), schema, args.system_prompt) for q in queries]
        
        # Batch call OpenAI API
        print(f"Calling OpenAI API in batches (model: {args.model})...")
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare completion parameters
        completion_kwargs = {"model": args.model, "temperature": 0.0}
        if args.model.startswith(('gpt-5', 'o1', 'o3')):
            completion_kwargs["max_completion_tokens"] = 500
        else:
            completion_kwargs["max_tokens"] = 500
        
        # Batch API calls
        api_responses = batch_chat_complete(client, messages_list, batch_size=50, **completion_kwargs)
        
        # Process results
        print("\nProcessing results and executing SQL queries...")
        all_results = []
        
        for i, (query_data, api_response) in enumerate(zip(queries, api_responses)):
            # Handle API errors
            if isinstance(api_response, Exception):
                result = {
                    'query': query_data.get('query', ''),
                    'sql': None,
                    'error': str(api_response),
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'paper_ids': [],
                    'count': 0,
                    'expected': query_data.get('paperId', ''),
                    'retrieved': []
                }
                all_results.append(result)
                continue
            
            # Process API response
            llm_result = process_api_response(api_response, log_first=(i == 0))
            
            # Execute SQL query
            result = execute_query_with_sql(conn, query_data.get('query', ''), llm_result)
            
            # Add evaluation fields
            result['expected'] = query_data.get('paperId', '')
            result['retrieved'] = result['paper_ids']
            
            all_results.append(result)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
        
        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                # Remove internal fields not needed in output
                output = {k: v for k, v in result.items() if k != 'paper_ids'}
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        # Print summary
        total_tokens = sum(r.get('input_tokens', 0) + r.get('output_tokens', 0) for r in all_results)
        avg_papers = sum(r['count'] for r in all_results) / len(all_results)
        
        print(f"\n{'='*60}")
        print(f"✓ Processed {len(all_results)} queries")
        print(f"  Output: {output_path}")
        print(f"  Avg papers retrieved: {avg_papers:.1f}")
        print(f"  Total tokens: {total_tokens:,} (avg: {total_tokens/len(all_results):.1f}/query)")
        print(f"{'='*60}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
