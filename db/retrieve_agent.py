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
    """Get database schema information for SQL generation (v2 schema)."""
    return """
DATABASE SCHEMA (v2):

Table: Papers (metadata only, no full text stored)
- corpus_id (INTEGER, PRIMARY KEY): Semantic Scholar corpus ID
- title (TEXT, NOT NULL): Paper title
- abstract (TEXT): Paper abstract (97.2% coverage)
- venue (VARCHAR): Original venue string
- venue_type (VARCHAR): 'main' or 'workshop'
- canonical_venue (VARCHAR): Normalized venue (workshops → main conference)
- year (INTEGER): Publication year (100% coverage)
- month (INTEGER): Publication month 1-12 (95.7% coverage)
- citation_count (INTEGER): Number of citations

Table: Authors
- author_id (VARCHAR, PRIMARY KEY): Semantic Scholar author ID
- name (VARCHAR, NOT NULL): Author name

Table: PaperAuthors (Junction table for many-to-many relationship)
- corpus_id (INTEGER, FOREIGN KEY → Papers.corpus_id): Reference to paper
- author_id (VARCHAR, FOREIGN KEY → Authors.author_id): Reference to author
- author_position (INTEGER): Author position (0 = first author)
- PRIMARY KEY: (corpus_id, author_id)

Table: Institutions
- institution_id (VARCHAR, PRIMARY KEY): OpenAlex institution ID
- display_name (TEXT): Institution name
- ror (VARCHAR): Research Organization Registry ID
- country_code (VARCHAR): ISO country code (e.g., 'US', 'CN', 'GB')
- institution_type (VARCHAR): Type (e.g., 'education', 'government', 'nonprofit')

Table: AuthorInstitutions (Junction table for many-to-many relationship)
- author_id (VARCHAR, FOREIGN KEY → Authors.author_id): Reference to author
- institution_id (VARCHAR, FOREIGN KEY → Institutions.institution_id): Reference to institution
- PRIMARY KEY: (author_id, institution_id)

RELATIONSHIPS:
- Papers ←→ Authors (many-to-many through PaperAuthors)
- Authors ←→ Institutions (many-to-many through AuthorInstitutions)
- To get papers with authors: JOIN Papers with PaperAuthors with Authors
- To get author affiliations: JOIN Authors with AuthorInstitutions with Institutions
- To filter by institution: JOIN through PaperAuthors, AuthorInstitutions, and Institutions

IMPORTANT NOTES:
- Use canonical_venue to query across workshops (e.g., canonical_venue='ACL' includes BioNLP@ACL)
- Use venue_type to filter main vs workshop papers
- Only 35.9% of papers have affiliation data (coverage limitation)
- Full text NOT in database - query source JSONL by corpus_id if needed

INDEXES:
- Papers: year, month, canonical_venue, venue_type, citation_count
- Authors: name
- Institutions: country_code, institution_type
"""


def execute_sql_query(conn, sql_query: str) -> Optional[List[str]]:
    """Execute SQL query and return list of corpus_ids (as strings for compatibility).
    
    Limits results to at most 10 papers.
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Try to find corpus_id column (v2 schema uses corpus_id as INTEGER primary key)
        column_names = [desc[0] for desc in cursor.description]
        if 'corpus_id' not in column_names:
            print(f"Warning: SQL query did not return corpus_id column")
            return []
        
        corpus_id_idx = column_names.index('corpus_id')
        # Convert to strings for compatibility with dense retriever output
        corpus_ids = [str(row[corpus_id_idx]) for row in rows if row[corpus_id_idx] is not None]
        
        # Limit to 10 results
        return corpus_ids[:10]
        
    except Exception as e:
        conn.rollback()
        print(f"Error executing SQL query: {e}\nSQL: {sql_query}")
        return []
    finally:
        cursor.close()


def get_paper_metadata_from_index(dense_retriever: DenseRetriever, corpus_ids: List[str]) -> List[Dict[str, Any]]:
    """Get paper metadata from dense retrieval index.
    
    Args:
        dense_retriever: DenseRetriever instance with loaded index
        corpus_ids: List of corpus IDs (as strings)
    
    Returns:
        List of paper metadata dictionaries with 'corpus_id' key (as string for compatibility)
    """
    if not corpus_ids:
        return []
    
    metadata = []
    # Build a lookup dict from paper_objs
    paper_lookup = {}
    for paper_obj in dense_retriever.paper_objs:
        corpus_id = str(paper_obj.get("corpus_id", paper_obj.get("paper_id", "")))
        paper_lookup[corpus_id] = paper_obj
    
    # Get metadata for requested corpus_ids in order
    for cid in corpus_ids:
        cid_str = str(cid)
        if cid_str in paper_lookup:
            paper_obj = paper_lookup[cid_str]
            authors = paper_obj.get('authors', [])
            # Extract author names
            author_names = []
            if authors:
                for author in authors:
                    if isinstance(author, dict):
                        author_names.append(author.get('name', ''))
                    elif isinstance(author, str):
                        author_names.append(author)
            
            metadata.append({
                'corpus_id': cid_str,
                'title': paper_obj.get('title', ''),
                'abstract': paper_obj.get('abstract', ''),
                'year': paper_obj.get('year'),
                'venue': paper_obj.get('venue', ''),
                'citation_count': paper_obj.get('citation_count', 0),
                'authors': author_names
            })
    
    return metadata


def format_paper_metadata_for_prompt(papers: List[Dict[str, Any]], max_papers: int = 10) -> str:
    """Format paper metadata as a readable list for the agent.
    
    Shows top max_papers with title and first 500 chars of abstract.
    """
    if not papers:
        return "No papers retrieved yet."
    
    # Limit to top max_papers
    papers_to_show = papers[:max_papers]
    
    lines = []
    for i, paper in enumerate(papers_to_show, 1):
        authors_str = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += f' et al. ({len(paper["authors"])} authors)'
        
        abstract = paper.get('abstract', '')
        abstract_preview = abstract[:500] + "..." if len(abstract) > 500 else abstract
        
        lines.append(f"{i}. [{paper['corpus_id']}]")
        lines.append(f"   Title: {paper['title']}")
        lines.append(f"   Authors: {authors_str}")
        lines.append(f"   Year: {paper.get('year', 'N/A')}, Venue: {paper.get('venue', 'N/A')}, Citations: {paper.get('citation_count', 0)}")
        lines.append(f"   Abstract: {abstract_preview}")
        lines.append("")
    
    if len(papers) > max_papers:
        lines.append(f"... and {len(papers) - max_papers} more papers")
    
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
5. Return at most 10 paper IDs in your final answer, ordered by relevance
6. Avoid returning duplicate papers - track what you've already seen
7. Each retrieval method (SQL or vector search) should return at most 10 papers

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
                            "description": "A PostgreSQL SQL query to retrieve papers. Must include corpus_id in SELECT (v2 schema uses corpus_id as INTEGER primary key). Use ILIKE for case-insensitive text matching. IMPORTANT: Always include LIMIT 10 in your query to return at most 10 papers. Example: SELECT DISTINCT p.corpus_id FROM Papers p JOIN PaperAuthors pa ON p.corpus_id = pa.corpus_id JOIN Authors a ON pa.author_id = a.author_id WHERE a.name ILIKE '%Smith%' ORDER BY p.citation_count DESC LIMIT 10"
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
                            "description": "Number of results to retrieve (default: 10, max: 10)",
                            "default": 10
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
                "description": "Return your final list of corpus IDs as the answer to the user's query. Call this when you're confident you have found the most relevant papers. You should return at most 10 corpus IDs ordered by relevance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "corpus_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of corpus IDs to return, ordered by relevance (most relevant first). Maximum 10 papers.",
                            "maxItems": 10
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why you selected these papers"
                        }
                    },
                    "required": ["corpus_ids"]
                }
            }
        }
    ]


def message_to_dict(msg) -> Dict[str, Any]:
    """Convert a message (dict or ChatCompletionMessage) to a dictionary."""
    if isinstance(msg, dict):
        return msg
    else:
        # It's a ChatCompletionMessage object
        result = {
            "role": msg.role,
            "content": msg.content
        }
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls
            ]
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        return result


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
        final_corpus_ids = []
        
        for turn_num in range(1, self.max_turns + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Turn {turn_num}/{self.max_turns}")
                print(f"{'='*60}")
            
            # Force termination at max_turns
            force_return = (turn_num == self.max_turns)
            if force_return:
                force_msg = "\n\n**IMPORTANT: This is your final turn. You MUST call return_papers now with your best selection of up to 10 papers.**"
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
                
                # Convert messages to JSON-serializable format for logging
                messages_for_logging = []
                for msg in messages:
                    msg_dict = message_to_dict(msg)
                    # Remove tool_calls from messages for cleaner logging (they're in assistant_message)
                    if msg_dict.get("role") == "assistant" and "tool_calls" in msg_dict:
                        msg_dict_log = {k: v for k, v in msg_dict.items() if k != "tool_calls"}
                        messages_for_logging.append(msg_dict_log)
                    else:
                        messages_for_logging.append(msg_dict)
                
                # Create JSON payload for logging
                json_payload = {
                    "model": completion_kwargs["model"],
                    "messages": messages_for_logging,
                    "tools": completion_kwargs["tools"],
                    "tool_choice": completion_kwargs["tool_choice"]
                }
                if "max_completion_tokens" in completion_kwargs:
                    json_payload["max_completion_tokens"] = completion_kwargs["max_completion_tokens"]
                if "max_tokens" in completion_kwargs:
                    json_payload["max_tokens"] = completion_kwargs["max_tokens"]
                
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
                # Try to create json_payload if it wasn't created yet
                try:
                    json_payload_for_error = {
                        "model": self.model,
                        "messages": [message_to_dict(msg) for msg in messages],
                        "tools": self.tools,
                        "tool_choice": "required"
                    }
                except:
                    json_payload_for_error = None
                turns.append({
                    "turn": turn_num,
                    "llm_request_json": json_payload_for_error,
                    "error": error_msg
                })
                break
            
            # Process tool calls
            if not assistant_message.tool_calls:
                if verbose:
                    print("[No tool calls made]")
                # Still log the turn even if no tool calls
                # json_payload already created above
                msg_dicts = [message_to_dict(msg) for msg in messages]
                turns.append({
                    "turn": turn_num,
                    "input_messages": [msg for msg in msg_dicts if msg.get("role") != "tool"],
                    "llm_request_json": json_payload,
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "assistant_message": {
                        "content": assistant_message.content,
                        "tool_calls": []
                    },
                    "error": "No tool calls made"
                })
                break
            
            # Convert messages to dicts for logging
            msg_dicts = [message_to_dict(msg) for msg in messages]
            turn_results = {
                "turn": turn_num,
                "input_messages": [msg for msg in msg_dicts if msg.get("role") != "tool"],  # Log prompts (exclude tool messages for brevity)
                "llm_request_json": json_payload,  # Exact JSON sent to LLM
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "assistant_message": {
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in (assistant_message.tool_calls or [])
                    ]
                },
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
                    corpus_ids = execute_sql_query(self.conn, sql_query)
                    
                    # Get metadata for new papers only
                    new_corpus_ids = [cid for cid in corpus_ids if cid not in seen_papers]
                    new_metadata = get_paper_metadata_from_index(self.dense_retriever, new_corpus_ids)
                    
                    # Update tracking
                    seen_papers.update(new_corpus_ids)
                    all_paper_metadata.extend(new_metadata)
                    turn_new_papers.extend(new_metadata)
                    
                    tool_result = {
                        "status": "success",
                        "total_results": len(corpus_ids),
                        "new_results": len(new_corpus_ids),
                        "message": f"Found {len(corpus_ids)} papers ({len(new_corpus_ids)} new)",
                        "corpus_ids": corpus_ids,  # Include all corpus IDs found
                        "new_corpus_ids": new_corpus_ids,  # Include new corpus IDs
                        "paper_metadata": new_metadata  # Include full metadata for new papers
                    }
                    
                    if verbose:
                        print(f"Results: {len(corpus_ids)} papers ({len(new_corpus_ids)} new)")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "sql": sql_query,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                    
                elif tool_name == "query_vector_db":
                    text_query = tool_args.get("text_query", "")
                    k = tool_args.get("k", 10)
                    k = min(k, 10)  # Cap at 10
                    
                    retrieved_data = self.dense_retriever.retrieve(text_query, k=k)
                    corpus_ids = retrieved_data["corpus_ids"]
                    # Limit to 10 results
                    corpus_ids = corpus_ids[:10]
                    
                    # Get metadata for new papers only
                    new_corpus_ids = [cid for cid in corpus_ids if cid not in seen_papers]
                    new_metadata = get_paper_metadata_from_index(self.dense_retriever, new_corpus_ids)
                    
                    # Update tracking
                    seen_papers.update(new_corpus_ids)
                    all_paper_metadata.extend(new_metadata)
                    turn_new_papers.extend(new_metadata)
                    
                    tool_result = {
                        "status": "success",
                        "total_results": len(corpus_ids),
                        "new_results": len(new_corpus_ids),
                        "message": f"Found {len(corpus_ids)} papers ({len(new_corpus_ids)} new)",
                        "corpus_ids": corpus_ids,  # Include all corpus IDs found
                        "new_corpus_ids": new_corpus_ids,  # Include new corpus IDs
                        "paper_metadata": new_metadata  # Include full metadata for new papers
                    }
                    
                    if verbose:
                        print(f"Results: {len(corpus_ids)} papers ({len(new_corpus_ids)} new)")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "text_query": text_query,
                        "k": k,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                    
                elif tool_name == "return_papers":
                    corpus_ids = tool_args.get("corpus_ids", tool_args.get("paper_ids", []))  # Support both for backward compatibility
                    reasoning = tool_args.get("reasoning", "")
                    
                    # Validate and limit to 10 papers
                    final_corpus_ids = corpus_ids[:10]
                    
                    if verbose:
                        print(f"Returning {len(final_corpus_ids)} papers")
                        if reasoning:
                            print(f"Reasoning: {reasoning}")
                    
                    turn_results["tool_calls"].append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "corpus_ids": final_corpus_ids,
                        "reasoning": reasoning
                    })
                    
                    # Don't add tool message for return_papers - we're done
                    
            turns.append(turn_results)
            
            # If return_papers was called, we're done
            if final_corpus_ids:
                break
            
            # Add tool results and paper metadata to conversation
            messages.extend(tool_messages)
            
            # Add paper metadata for next turn (top 10 papers with title and first 500 chars of abstract)
            if turn_new_papers:
                metadata_text = format_paper_metadata_for_prompt(turn_new_papers, max_papers=10)
                user_message = {
                    "role": "user",
                    "content": f"Here are the top {min(len(turn_new_papers), 10)} papers retrieved (showing {len(turn_new_papers)} total):\n\n{metadata_text}\n\nYou can continue searching or return your final selection."
                }
                messages.append(user_message)
                # Store the user message in turn results for logging
                turn_results["user_feedback"] = user_message["content"]
            
            # Store tool messages in turn results
            turn_results["tool_messages"] = [
                {
                    "role": tm["role"],
                    "tool_call_id": tm.get("tool_call_id"),
                    "content": tm["content"]
                } for tm in tool_messages
            ]
        
        return {
            "query": user_query,
            "retrieved": final_corpus_ids,
            "turns": turns,
            "total_papers_seen": len(seen_papers),
            "all_papers_seen": list(seen_papers),  # Include all corpus IDs seen across all turns
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "model": self.model,
            "max_turns": self.max_turns,
            "turns_used": len(turns),
            "system_prompt": self.system_prompt  # Include system prompt for reference
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
                       help='Path to query file (JSONL format with "query" and "paperId"/"corpusId" fields)')
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
            # Support corpus_id (with underscore) as primary, fallback to other formats for compatibility
            expected_corpus_id = query_data.get('corpus_id', query_data.get('corpusId', query_data.get('corpusid', query_data.get('paperId', ''))))
            
            if args.verbose:
                print(f"\n\n{'#'*60}")
                print(f"Query {i}/{len(queries)}: {user_query}")
                print(f"Expected: {expected_corpus_id}")
                print(f"{'#'*60}")
            else:
                print(f"Processing query {i}/{len(queries)}...", end='\r')
            
            # Run agent retrieval
            result = agent.retrieve(user_query, verbose=args.verbose)
            
            # Add evaluation fields
            result['expected'] = expected_corpus_id
            
            # Include all detailed information in output
            output_result = {
                'query': result['query'],
                'expected': result['expected'],
                'retrieved': result['retrieved'],
                'turns': result['turns'],
                'total_papers_seen': result['total_papers_seen'],
                'all_papers_seen': result.get('all_papers_seen', []),
                'total_input_tokens': result['total_input_tokens'],
                'total_output_tokens': result['total_output_tokens'],
                'total_tokens': result.get('total_tokens', result['total_input_tokens'] + result['total_output_tokens']),
                'model': result.get('model', args.model),
                'max_turns': result.get('max_turns', args.max_turns),
                'turns_used': result.get('turns_used', len(result['turns'])),
                'system_prompt': result.get('system_prompt', '')  # Include system prompt
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

