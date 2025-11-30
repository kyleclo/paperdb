import json
import argparse
import psycopg2
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
import os
from retrieval.dense import DenseRetriever


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
    """Get database schema information for SQL generation."""
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


def execute_sql_query(conn, sql_query: str) -> Optional[List[str]]:
    """Execute SQL query and return list of paper_ids."""
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Try to find paper_id column
        column_names = [desc[0] for desc in cursor.description]
        if 'paper_id' not in column_names:
            print(f"Warning: SQL query did not return paper_id column")
            return []
        
        paper_id_idx = column_names.index('paper_id')
        paper_ids = [row[paper_id_idx] for row in rows if row[paper_id_idx]]
        
        return paper_ids
        
    except Exception as e:
        conn.rollback()
        print(f"Error executing SQL query: {e}\nSQL: {sql_query}")
        return []
    finally:
        cursor.close()


def get_paper_metadata(conn, paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Get paper metadata (title, authors, abstract) from database."""
    if not paper_ids:
        return []
    
    cursor = conn.cursor()
    
    try:
        # Build query to get papers and their authors
        placeholders = ','.join(['%s'] * len(paper_ids))
        query = f"""
        SELECT 
            p.paper_id,
            p.title,
            p.abstract,
            p.year,
            p.venue,
            p.citation_count,
            COALESCE(
                json_agg(
                    json_build_object('name', a.name, 'position', pa.author_position)
                    ORDER BY pa.author_position
                ) FILTER (WHERE a.author_id IS NOT NULL),
                '[]'
            ) as authors
        FROM Papers p
        LEFT JOIN PaperAuthors pa ON p.paper_id = pa.paper_id
        LEFT JOIN Authors a ON pa.author_id = a.author_id
        WHERE p.paper_id IN ({placeholders})
        GROUP BY p.paper_id, p.title, p.abstract, p.year, p.venue, p.citation_count
        """
        
        cursor.execute(query, paper_ids)
        rows = cursor.fetchall()
        
        metadata = []
        for row in rows:
            paper_id, title, abstract, year, venue, citation_count, authors_json = row
            authors = json.loads(authors_json) if isinstance(authors_json, str) else authors_json
            author_names = [a['name'] for a in sorted(authors, key=lambda x: x['position'])]
            
            metadata.append({
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract or '',
                'year': year,
                'venue': venue or '',
                'citation_count': citation_count or 0,
                'authors': author_names
            })
        
        # Maintain order of paper_ids
        metadata_dict = {m['paper_id']: m for m in metadata}
        ordered_metadata = [metadata_dict[pid] for pid in paper_ids if pid in metadata_dict]
        
        return ordered_metadata
        
    except Exception as e:
        print(f"Error getting paper metadata: {e}")
        return []
    finally:
        cursor.close()


def format_paper_metadata_for_prompt(papers: List[Dict[str, Any]]) -> str:
    """Format paper metadata as a readable list for the agent."""
    if not papers:
        return "No papers retrieved yet."
    
    lines = []
    for i, paper in enumerate(papers, 1):
        authors_str = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += f' et al. ({len(paper["authors"])} authors)'
        
        lines.append(f"{i}. [{paper['paper_id']}]")
        lines.append(f"   Title: {paper['title']}")
        lines.append(f"   Authors: {authors_str}")
        lines.append(f"   Year: {paper.get('year', 'N/A')}, Venue: {paper.get('venue', 'N/A')}, Citations: {paper.get('citation_count', 0)}")
        lines.append(f"   Abstract: {paper['abstract'][:200]}..." if len(paper['abstract']) > 200 else f"   Abstract: {paper['abstract']}")
        lines.append("")
    
    return '\n'.join(lines)


def get_agent_system_prompt(schema: str) -> str:
    """Get the detailed system prompt for the agent."""
    return f"""You are an expert research assistant helping users find academic papers. You have access to two retrieval systems:

1. **Relational Database (SQL)**: A PostgreSQL database with structured metadata about papers, authors, venues, years, and citation counts. Use this for queries about specific attributes (e.g., "papers by author X", "papers from venue Y", "papers published in year Z").

2. **Vector Database (Semantic Search)**: A FAISS index with semantic embeddings of paper content. Use this for queries about topics, concepts, or research questions (e.g., "papers about neural networks", "research on climate change").

{schema}

**Your Task:**
Given a user's search query, you should strategically use these tools to find the most relevant papers. You can:
- Query the relational database with SQL
- Query the vector database with a text query
- Call both tools multiple times to refine your search
- Combine results from both tools

**Important Guidelines:**
1. You can make multiple tool calls to explore different angles or refine your search
2. After each tool call, you'll see the metadata (title, authors, abstract) of retrieved papers
3. Use this information to decide if you need to search more or if you have enough results
4. When you're confident you have found relevant papers, return your final list of paper IDs
5. Return at most 20 paper IDs in your final answer, ordered by relevance
6. Avoid returning duplicate papers - track what you've already seen

**Strategy Tips:**
- For queries with specific attributes (author, venue, year, citation count), start with SQL
- For conceptual/topical queries, start with vector search
- If initial results are not satisfactory, try alternative queries or the other tool
- Combine results from multiple searches for comprehensive coverage
"""


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Define the three function tools available to the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_relational_db",
                "description": "Query the relational database using SQL to find papers based on structured attributes like authors, venues, years, citation counts. Returns a list of matching paper IDs with their metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "A PostgreSQL SQL query to retrieve papers. Must include paper_id in SELECT. Use ILIKE for case-insensitive text matching. Example: SELECT DISTINCT p.paper_id FROM Papers p JOIN PaperAuthors pa ON p.paper_id = pa.paper_id JOIN Authors a ON pa.author_id = a.author_id WHERE a.name ILIKE '%Smith%' ORDER BY p.citation_count DESC LIMIT 50"
                        }
                    },
                    "required": ["sql_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_vector_db",
                "description": "Query the vector database using semantic search to find papers based on content, topics, or research questions. Returns a list of semantically similar papers with their metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_query": {
                            "type": "string",
                            "description": "A natural language query describing the topic, concept, or research question you're looking for. Examples: 'neural networks for image classification', 'climate change adaptation strategies', 'quantum computing algorithms'"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to retrieve (default: 50, max: 100)",
                            "default": 50
                        }
                    },
                    "required": ["text_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "return_papers",
                "description": "Return your final list of paper IDs as the answer to the user's query. Call this when you're confident you have found the most relevant papers. You should return at most 20 paper IDs ordered by relevance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of paper IDs to return, ordered by relevance (most relevant first). Maximum 20 papers.",
                            "maxItems": 20
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why you selected these papers"
                        }
                    },
                    "required": ["paper_ids"]
                }
            }
        }
    ]


class AgentRetriever:
    """Multi-turn agent-based retrieval system."""
    
    def __init__(self, conn, dense_retriever: DenseRetriever, client: openai.OpenAI, 
                 model: str, max_turns: int = 5):
        self.conn = conn
        self.dense_retriever = dense_retriever
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.schema = get_database_schema()
        self.system_prompt = get_agent_system_prompt(self.schema)
        self.tools = get_tool_definitions()
        
    def retrieve(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute multi-turn retrieval for a single query.
        
        Returns:
            Dict with keys:
                - 'query': original query
                - 'retrieved': final list of paper IDs
                - 'turns': list of turns with tool calls and results
                - 'total_input_tokens': total input tokens used
                - 'total_output_tokens': total output tokens used
        """
        # Initialize conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Find relevant papers for this query: {user_query}"}
        ]
        
        # Track state
        seen_papers = set()  # Track all papers seen across turns
        all_paper_metadata = []  # Track all paper metadata
        turns = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_paper_ids = []
        
        for turn_num in range(1, self.max_turns + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Turn {turn_num}/{self.max_turns}")
                print(f"{'='*60}")
            
            # Force termination at max_turns
            force_return = (turn_num == self.max_turns)
            if force_return:
                force_msg = "\n\n**IMPORTANT: This is your final turn. You MUST call return_papers now with your best selection of up to 20 papers.**"
                messages.append({
                    "role": "user",
                    "content": force_msg
                })
                if verbose:
                    print(f"[Forcing return at max turns]")
            
            # Call LLM with tools
            try:
                completion_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools,
                    "tool_choice": "required"  # Force tool use
                }
                
                # Add max_tokens or max_completion_tokens based on model
                if self.model.startswith(('gpt-5', 'o1', 'o3')):
                    completion_kwargs["max_completion_tokens"] = 2000
                else:
                    completion_kwargs["max_tokens"] = 2000
                
                response = self.client.chat.completions.create(**completion_kwargs)
                
                # Track tokens
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens
                
                assistant_message = response.choices[0].message
                messages.append(assistant_message)
                
            except Exception as e:
                error_msg = f"Error calling LLM: {e}"
                if verbose:
                    print(f"[ERROR] {error_msg}")
                turns.append({
                    "turn": turn_num,
                    "error": error_msg
                })
                break
            
            # Process tool calls
            if not assistant_message.tool_calls:
                if verbose:
                    print("[No tool calls made]")
                break
            
            turn_results = {
                "turn": turn_num,
                "tool_calls": []
            }
            
            tool_messages = []
            turn_new_papers = []
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"\n[Tool Call: {tool_name}]")
                    print(f"Arguments: {json.dumps(tool_args, indent=2)}")
                
                # Execute tool
                if tool_name == "query_relational_db":
                    sql_query = tool_args.get("sql_query", "")
                    paper_ids = execute_sql_query(self.conn, sql_query)
                    
                    # Get metadata for new papers only
                    new_paper_ids = [pid for pid in paper_ids if pid not in seen_papers]
                    new_metadata = get_paper_metadata(self.conn, new_paper_ids)
                    
                    # Update tracking
                    seen_papers.update(new_paper_ids)
                    all_paper_metadata.extend(new_metadata)
                    turn_new_papers.extend(new_metadata)
                    
                    tool_result = {
                        "status": "success",
                        "total_results": len(paper_ids),
                        "new_results": len(new_paper_ids),
                        "message": f"Found {len(paper_ids)} papers ({len(new_paper_ids)} new)"
                    }
                    
                    if verbose:
                        print(f"Results: {len(paper_ids)} papers ({len(new_paper_ids)} new)")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "sql": sql_query,
                        "result": tool_result
                    })
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                    
                elif tool_name == "query_vector_db":
                    text_query = tool_args.get("text_query", "")
                    k = tool_args.get("k", 50)
                    k = min(k, 100)  # Cap at 100
                    
                    retrieved_data = self.dense_retriever.retrieve(text_query, k=k)
                    paper_ids = retrieved_data["paper_ids"]
                    
                    # Get metadata for new papers only
                    new_paper_ids = [pid for pid in paper_ids if pid not in seen_papers]
                    new_metadata = get_paper_metadata(self.conn, new_paper_ids)
                    
                    # Update tracking
                    seen_papers.update(new_paper_ids)
                    all_paper_metadata.extend(new_metadata)
                    turn_new_papers.extend(new_metadata)
                    
                    tool_result = {
                        "status": "success",
                        "total_results": len(paper_ids),
                        "new_results": len(new_paper_ids),
                        "message": f"Found {len(paper_ids)} papers ({len(new_paper_ids)} new)"
                    }
                    
                    if verbose:
                        print(f"Results: {len(paper_ids)} papers ({len(new_paper_ids)} new)")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "text_query": text_query,
                        "k": k,
                        "result": tool_result
                    })
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                    
                elif tool_name == "return_papers":
                    paper_ids = tool_args.get("paper_ids", [])
                    reasoning = tool_args.get("reasoning", "")
                    
                    # Validate and limit to 20 papers
                    final_paper_ids = paper_ids[:20]
                    
                    if verbose:
                        print(f"Returning {len(final_paper_ids)} papers")
                        if reasoning:
                            print(f"Reasoning: {reasoning}")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "paper_ids": final_paper_ids,
                        "reasoning": reasoning
                    })
                    
                    # Don't add tool message for return_papers - we're done
                    
            turns.append(turn_results)
            
            # If return_papers was called, we're done
            if final_paper_ids:
                break
            
            # Add tool results and paper metadata to conversation
            messages.extend(tool_messages)
            
            # Add paper metadata for next turn
            if turn_new_papers:
                metadata_text = format_paper_metadata_for_prompt(turn_new_papers)
                messages.append({
                    "role": "user",
                    "content": f"Here are the {len(turn_new_papers)} new papers retrieved:\n\n{metadata_text}\n\nYou can continue searching or return your final selection."
                })
        
        return {
            "query": user_query,
            "retrieved": final_paper_ids,
            "turns": turns,
            "total_papers_seen": len(seen_papers),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }


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
        description="Retrieve papers using multi-turn agent with SQL and vector search"
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
    
    # Dense index parameters
    parser.add_argument('--index_path', type=str, required=True,
                       help='Path to the FAISS index directory')
    
    # Query parameters
    parser.add_argument('--query_file', type=str, required=True,
                       help='Path to query file (JSONL format with "query" and "paperId" fields)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output file (JSONL format)')
    
    # Agent parameters
    parser.add_argument('--api_key', type=str,
                       help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--max_turns', type=int, default=5,
                       help='Maximum number of agent turns (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
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
        # Load dense retriever
        print(f"Loading dense retriever from: {args.index_path}")
        dense_retriever = DenseRetriever(index_dir=args.index_path)
        dense_retriever.load()
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Load queries
        print(f"\nLoading queries from: {args.query_file}")
        queries = load_queries(args.query_file)
        print(f"Loaded {len(queries)} queries")
        
        # Initialize agent retriever
        agent = AgentRetriever(
            conn=conn,
            dense_retriever=dense_retriever,
            client=client,
            model=args.model,
            max_turns=args.max_turns
        )
        
        # Process queries
        print(f"\nProcessing queries with {args.model} (max {args.max_turns} turns per query)...")
        print("="*60)
        
        all_results = []
        for i, query_data in enumerate(queries, 1):
            user_query = query_data.get('query', '')
            expected_paper_id = query_data.get('paperId', '')
            
            if args.verbose:
                print(f"\n\n{'#'*60}")
                print(f"Query {i}/{len(queries)}: {user_query}")
                print(f"Expected: {expected_paper_id}")
                print(f"{'#'*60}")
            else:
                print(f"Processing query {i}/{len(queries)}...", end='\r')
            
            # Run agent retrieval
            result = agent.retrieve(user_query, verbose=args.verbose)
            
            # Add evaluation fields
            result['expected'] = expected_paper_id
            
            # Remove internal fields for output
            output_result = {
                'query': result['query'],
                'expected': result['expected'],
                'retrieved': result['retrieved'],
                'turns': result['turns'],
                'total_papers_seen': result['total_papers_seen'],
                'total_input_tokens': result['total_input_tokens'],
                'total_output_tokens': result['total_output_tokens']
            }
            
            all_results.append(output_result)
            
            if not args.verbose:
                # Show quick summary
                num_turns = len(result['turns'])
                num_retrieved = len(result['retrieved'])
                print(f"Query {i}/{len(queries)}: {num_turns} turns, {num_retrieved} papers returned" + " "*20)
        
        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Print summary
        total_tokens = sum(r['total_input_tokens'] + r['total_output_tokens'] for r in all_results)
        avg_turns = sum(len(r['turns']) for r in all_results) / len(all_results)
        avg_papers = sum(len(r['retrieved']) for r in all_results) / len(all_results)
        
        print(f"\n{'='*60}")
        print(f"✓ Processed {len(all_results)} queries")
        print(f"  Model: {args.model}")
        print(f"  Max turns: {args.max_turns}")
        print(f"  Avg turns used: {avg_turns:.1f}")
        print(f"  Avg papers returned: {avg_papers:.1f}")
        print(f"  Total tokens: {total_tokens:,} (avg: {total_tokens/len(all_results):.1f}/query)")
        print(f"  Output: {output_path}")
        print(f"{'='*60}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

