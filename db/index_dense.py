import json
import argparse
import pickle
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss


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


def extract_retrieval_units(papers, retrieval_units):
    """Extract retrieval units from papers."""
    units = []
    
    for paper in tqdm(papers, desc="Extracting retrieval units"):
        paper_id = paper.get('paperId', paper.get('corpusId', ''))
        
        # Paragraphs
        if 'paragraphs' in retrieval_units and 'paragraphs' in paper:
            for para_idx, paragraph in enumerate(paper['paragraphs']):
                text = paragraph.get('text', '')
                if text:
                    unit_id = f"{paper_id}_para_{para_idx}"
                    metadata = {
                        'unit_type': 'paragraph',
                        'section_title': paragraph.get('sectionTitle', ''),
                        'paragraph_title': paragraph.get('title', ''),
                        'paragraph_id': paragraph.get('paragraphId', '')
                    }
                    units.append((unit_id, paper_id, text, metadata))
        
        # Abstract
        if 'abstracts' in retrieval_units and 'abstract' in paper:
            abstract = paper.get('abstract', '')
            if abstract:
                unit_id = f"{paper_id}_abstract"
                metadata = {'unit_type': 'abstract'}
                units.append((unit_id, paper_id, abstract, metadata))
        
        # Title
        if 'title' in retrieval_units and 'title' in paper:
            title = paper.get('title', '')
            if title:
                unit_id = f"{paper_id}_title"
                metadata = {'unit_type': 'title'}
                units.append((unit_id, paper_id, title, metadata))
        
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
            if 'fieldsOfStudy' in paper and paper['fieldsOfStudy']:
                parts.append(f"Fields: {', '.join(paper['fieldsOfStudy'])}")
            if 'publicationTypes' in paper and paper['publicationTypes']:
                parts.append(f"Publication Types: {', '.join(paper['publicationTypes'])}")
            
            metadata_text = ' | '.join(parts)
            if metadata_text:
                unit_id = f"{paper_id}_metadata"
                metadata = {
                    'unit_type': 'metadata',
                    'year': paper.get('year'),
                    'venue': paper.get('venue'),
                    'citation_count': paper.get('citationCount')
                }
                units.append((unit_id, paper_id, metadata_text, metadata))
    
    print(f"Extracted {len(units)} retrieval units")
    return units


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings using last token (following official Qwen3-Embedding example)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def compute_embeddings(texts, tokenizer, model, device, batch_size=32):
    """Compute embeddings for texts using last token pooling."""
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                             max_length=8192, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # Last token pooling (official method)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def build_index(paper_file, retrieval_units, output_dir, model_name='Qwen/Qwen3-Embedding-0.6B', batch_size=32):
    """Build retrieval index."""
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    
    # Load papers
    papers = load_papers(paper_file)
    
    # Extract units
    units = extract_retrieval_units(papers, retrieval_units)
    if not units:
        print("Warning: No retrieval units extracted!")
        return
    
    unit_ids = [u[0] for u in units]
    paper_ids = [u[1] for u in units]
    texts = [u[2] for u in units]
    metadatas = [u[3] for u in units]
    
    # Compute embeddings
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = compute_embeddings(texts, tokenizer, model, device, batch_size)
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
        paper_id = paper.get('paperId', paper.get('corpusId', ''))
        paper_objs[paper_id] = {
            'paper_id': paper_id,
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'authors': paper.get('authors', []),
            'year': paper.get('year'),
            'venue': paper.get('venue', ''),
            'citation_count': paper.get('citationCount', 0),
            'fields_of_study': paper.get('fieldsOfStudy', []),
            'unit_ids_to_retrieval_units': {}
        }
    
    # Map unit_ids to retrieval units
    for unit_id, paper_id, text, metadata in units:
        if paper_id in paper_objs:
            paper_objs[paper_id]['unit_ids_to_retrieval_units'][unit_id] = {
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
                       choices=['paragraphs', 'abstracts', 'title', 'metadata'],
                       help='Unit types to index')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                       help='Embedding model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    build_index(args.paper_file, args.retrieval_units, args.output_dir, 
                args.model_name, args.batch_size)


if __name__ == '__main__':
    main()

