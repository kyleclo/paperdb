import json
import argparse
import pickle
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import faiss
from vllm import LLM


def load_papers(paper_file):
    """Load papers from JSONL file."""
    papers = []
    print(f"Loading papers from: {paper_file}")
    with open(paper_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f"Loaded {len(papers)} papers")
    return papers


def chunk_text_by_tokens(text, tokenizer, max_tokens=512):
    """Split text into chunks of at most max_tokens tokens."""
    if not text:
        return []
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    # Split into chunks of max_tokens
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks


def extract_retrieval_units(papers, retrieval_units, tokenizer=None):
    """Extract retrieval units from papers."""
    units = []
    
    for paper in tqdm(papers, desc="Extracting retrieval units"):
        # Support both v1 (camelCase) and v2 (lowercase) formats
        corpus_id = paper.get('paperId', paper.get('corpusId', paper.get('corpusid', '')))
        
        # Fulltext body chunks (replaces paragraphs)
        if 'fulltext' in retrieval_units or 'paragraphs' in retrieval_units:
            fulltext_body = paper.get('fulltext_body')
            if fulltext_body and tokenizer is not None:
                # Chunk the fulltext_body into 512-token chunks
                chunks = chunk_text_by_tokens(fulltext_body, tokenizer, max_tokens=512)
                for chunk_idx, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        unit_id = f"{corpus_id}_fulltext_{chunk_idx}"
                        metadata = {
                            'unit_type': 'fulltext',
                            'chunk_index': chunk_idx
                        }
                        units.append((unit_id, corpus_id, chunk_text, metadata))
        
        # Abstract
        if 'abstracts' in retrieval_units and 'abstract' in paper:
            abstract = paper.get('abstract')
            if abstract:  # Handle None and empty string
                unit_id = f"{corpus_id}_abstract"
                metadata = {'unit_type': 'abstract'}
                units.append((unit_id, corpus_id, abstract, metadata))
        
        # Title
        if 'title' in retrieval_units and 'title' in paper:
            title = paper.get('title', '')
            if title:
                unit_id = f"{corpus_id}_title"
                metadata = {'unit_type': 'title'}
                units.append((unit_id, corpus_id, title, metadata))
        
        # Metadata
        if 'metadata' in retrieval_units:
            parts = []
            if 'authors' in paper and paper['authors']:
                names = [a.get('name', '') for a in paper['authors']]
                parts.append(f"Authors: {', '.join(names)}")
            if 'venue' in paper and paper['venue']:
                parts.append(f"Venue: {paper['venue']}")
            if 'year' in paper and paper['year']:
                parts.append(f"Year: {paper['year']}")
            
            # Handle s2fieldsofstudy (list of dicts with 'category' key)
            s2fields = paper.get('s2fieldsofstudy', [])
            if s2fields:
                # Extract categories from dicts
                categories = [f.get('category', '') for f in s2fields if isinstance(f, dict) and f.get('category')]
                if categories:
                    parts.append(f"Fields: {', '.join(categories)}")
            
            # Fallback to old format if s2fieldsofstudy not available
            if not s2fields:
                fields_of_study = paper.get('fieldsOfStudy', paper.get('fieldsofstudy', []))
                if fields_of_study:
                    # Handle both list of strings and list of dicts
                    if fields_of_study and isinstance(fields_of_study[0], str):
                        parts.append(f"Fields: {', '.join(fields_of_study)}")
                    elif fields_of_study:
                        categories = [f.get('category', '') for f in fields_of_study if isinstance(f, dict) and f.get('category')]
                        if categories:
                            parts.append(f"Fields: {', '.join(categories)}")
            
            publication_types = paper.get('publicationTypes', paper.get('publicationtypes', []))
            if publication_types:
                parts.append(f"Publication Types: {', '.join(publication_types)}")
            
            metadata_text = ' | '.join(parts)
            if metadata_text:
                unit_id = f"{corpus_id}_metadata"
                # Support both citationCount (v1) and citationcount (v2)
                citation_count = paper.get('citationCount', paper.get('citationcount', 0))
                metadata = {
                    'unit_type': 'metadata',
                    'year': paper.get('year'),
                    'venue': paper.get('venue'),
                    'citation_count': citation_count
                }
                units.append((unit_id, corpus_id, metadata_text, metadata))
    
    print(f"Extracted {len(units)} retrieval units")
    return units


def compute_embeddings(texts, model, batch_size=32):
    """Compute embeddings for texts using vllm."""
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # vllm handles embedding computation internally
        outputs = model.embed(batch_texts)
        
        # Extract embeddings from outputs
        batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        # vllm embeddings are already normalized, but ensure numpy format
        all_embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def build_index(paper_file, retrieval_units, output_dir, model_name='Qwen/Qwen3-Embedding-0.6B', batch_size=32, test_mode=False):
    """Build retrieval index."""
    # Load tokenizer for text chunking (still needed for fulltext chunking)
    print(f"Loading tokenizer for text chunking: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load vllm model for embeddings
    print(f"Loading vllm model: {model_name}")
    model = LLM(model=model_name, task="embed")
    print(f"Model loaded")
    
    # Load papers
    papers = load_papers(paper_file)
    
    # Limit to first 100 papers in test mode
    if test_mode:
        original_count = len(papers)
        papers = papers[:100]
        print(f"TEST MODE: Limited to first 100 papers (from {original_count} total)")
    
    # Extract units (pass tokenizer for fulltext chunking)
    units = extract_retrieval_units(papers, retrieval_units, tokenizer=tokenizer)
    if not units:
        print("Warning: No retrieval units extracted!")
        return
    
    unit_ids = [u[0] for u in units]
    corpus_ids = [u[1] for u in units]
    texts = [u[2] for u in units]
    metadatas = [u[3] for u in units]
    
    # Compute embeddings
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = compute_embeddings(texts, model, batch_size)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("Building FAISS index...")
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (embeddings already normalized)
    faiss_index.add(embeddings)
    print(f"FAISS index built with {faiss_index.ntotal} vectors")
    
    # Build paper objects
    print("Building paper objects...")
    paper_objs = {}
    for paper in papers:
        # Support both v1 (camelCase) and v2 (lowercase) formats
        corpus_id = paper.get('paperId', paper.get('corpusId', paper.get('corpusid', '')))
        # Support both citationCount (v1) and citationcount (v2)
        citation_count = paper.get('citationCount', paper.get('citationcount', 0))
        # Handle s2fieldsofstudy (list of dicts with 'category' key)
        s2fields = paper.get('s2fieldsofstudy', [])
        if s2fields:
            # Extract categories from dicts
            fields_of_study = [f.get('category', '') for f in s2fields if isinstance(f, dict) and f.get('category')]
        else:
            # Fallback to old format
            fields_of_study = paper.get('fieldsOfStudy', paper.get('fieldsofstudy', []))
            # If it's a list of dicts, extract categories
            if fields_of_study and isinstance(fields_of_study[0], dict):
                fields_of_study = [f.get('category', '') for f in fields_of_study if isinstance(f, dict) and f.get('category')]
        
        paper_objs[corpus_id] = {
            'corpus_id': corpus_id,
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'authors': paper.get('authors', []),
            'year': paper.get('year'),
            'venue': paper.get('venue', ''),
            'citation_count': citation_count,
            'fields_of_study': fields_of_study,
            'unit_ids_to_retrieval_units': {}
        }
    
    # Map unit_ids to retrieval units
    for unit_id, corpus_id, text, metadata in units:
        if corpus_id in paper_objs:
            paper_objs[corpus_id]['unit_ids_to_retrieval_units'][unit_id] = {
                'text': text,
                'metadata': metadata
            }
    
    paper_objs_list = list(paper_objs.values())
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss_file = output_path / 'faiss_index.faiss'
    print(f"Saving FAISS index to: {faiss_file}")
    faiss.write_index(faiss_index, str(faiss_file))
    
    # Save unit_ids
    unit_ids_file = output_path / 'unit_ids.pkl'
    print(f"Saving unit IDs to: {unit_ids_file}")
    with open(unit_ids_file, 'wb') as f:
        pickle.dump(unit_ids, f)
    
    # Save paper objects
    paper_objs_file = output_path / 'paper_objs.pkl'
    print(f"Saving paper objects to: {paper_objs_file}")
    with open(paper_objs_file, 'wb') as f:
        pickle.dump(paper_objs_list, f)
    
    # Save metadata
    unit_type_counts = {}
    for m in metadatas:
        unit_type = m['unit_type']
        unit_type_counts[unit_type] = unit_type_counts.get(unit_type, 0) + 1
    
    metadata_summary = {
        'model_name': model_name,
        'retrieval_units': retrieval_units,
        'n_papers': len(papers),
        'n_units': len(units),
        'embedding_dim': int(embedding_dim),
        'unit_type_counts': unit_type_counts
    }
    
    metadata_file = output_path / 'index_metadata.json'
    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Index building completed!")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total papers: {len(papers)}")
    print(f"Total retrieval units: {len(units)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Unit type distribution:")
    for unit_type, count in unit_type_counts.items():
        print(f"  - {unit_type}: {count}")
    print(f"\nOutput files:")
    print(f"  - FAISS Index: {faiss_file}")
    print(f"  - Unit IDs: {unit_ids_file}")
    print(f"  - Paper Objects: {paper_objs_file}")
    print(f"  - Metadata: {metadata_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval index for papers")
    parser.add_argument('--paper_file', type=str, required=True, help='Path to JSONL file')
    parser.add_argument('--retrieval_units', type=str, nargs='+', required=True,
                       choices=['fulltext', 'paragraphs', 'abstracts', 'title', 'metadata'],
                       help='Unit types to index (fulltext replaces paragraphs for new format)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                       help='Embedding model name')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: only use the first 100 papers')
    
    args = parser.parse_args()
    
    build_index(args.paper_file, args.retrieval_units, args.output_dir, 
                args.model_name, args.batch_size, test_mode=args.test)


if __name__ == '__main__':
    main()

