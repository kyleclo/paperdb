import json
import pickle
from pathlib import Path
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings using last token (following official Qwen3-Embedding example)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class DenseRetriever:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", index_dir: str = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.faiss_index = None
        self.unit_ids = None
        self.paper_objs = None
        self.unit_id_to_paper_id = {}
        
        if index_dir is None:
            # Default to the index-all-units directory
            self.index_dir = Path(__file__).parent.parent.parent / "data" / "index-all-units"
        else:
            self.index_dir = Path(index_dir)

    def load(self):
        """Load FAISS index and associated data."""
        print(f"Loading FAISS index from: {self.index_dir}")
        
        # Load metadata and get model_name if available
        metadata_file = self.index_dir / "index_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if "model_name" in metadata:
                    self.model_name = metadata["model_name"]
                    print(f"Using model from metadata: {self.model_name}")
        
        # Load FAISS index
        faiss_file = self.index_dir / "faiss_index.faiss"
        self.faiss_index = faiss.read_index(str(faiss_file))
        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Load unit IDs
        unit_ids_file = self.index_dir / "unit_ids.pkl"
        with open(unit_ids_file, "rb") as f:
            self.unit_ids = pickle.load(f)
        print(f"Loaded {len(self.unit_ids)} unit IDs")
        
        # Load paper objects
        paper_objs_file = self.index_dir / "paper_objs.pkl"
        with open(paper_objs_file, "rb") as f:
            self.paper_objs = pickle.load(f)
        print(f"Loaded {len(self.paper_objs)} paper objects")
        
        # Build unit_id to paper_id mapping
        for paper_obj in self.paper_objs:
            paper_id = paper_obj["paper_id"]
            for unit_id in paper_obj["unit_ids_to_retrieval_units"].keys():
                self.unit_id_to_paper_id[unit_id] = paper_id
        
        # Load embedding model
        print(f"Loading embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into an embedding vector."""
        with torch.no_grad():
            inputs = self.tokenizer([query], padding=True, truncation=True, 
                                   max_length=8192, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Last token pooling
            embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()[0]

    def get_unit_text(self, unit_id: str) -> str:
        """Get the text content for a given unit_id."""
        paper_id = self.unit_id_to_paper_id.get(unit_id)
        if not paper_id:
            return ""
        
        # Find the paper object
        for paper_obj in self.paper_objs:
            if paper_obj["paper_id"] == paper_id:
                retrieval_unit = paper_obj["unit_ids_to_retrieval_units"].get(unit_id)
                if retrieval_unit:
                    return retrieval_unit.get("text", "")
        return ""

    def retrieve(self, query: str, k: int = 5) -> dict:
        """
        Retrieve top-k units for a query.
        
        Returns:
            dict with keys:
                - 'unit_ids': list of unit IDs
                - 'paper_ids': list of deduplicated paper IDs (in order of first occurrence)
                - 'unit_texts': list of text content for each unit
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        indices = indices[0]  # Get first row
        
        # Get unit IDs
        retrieved_unit_ids = [self.unit_ids[i] for i in indices]
        
        # Get unit texts
        retrieved_unit_texts = [self.get_unit_text(unit_id) for unit_id in retrieved_unit_ids]
        
        # Get paper IDs and deduplicate while preserving order
        retrieved_paper_ids = []
        seen_paper_ids = set()
        for unit_id in retrieved_unit_ids:
            paper_id = self.unit_id_to_paper_id.get(unit_id)
            if paper_id and paper_id not in seen_paper_ids:
                retrieved_paper_ids.append(paper_id)
                seen_paper_ids.add(paper_id)
        
        return {
            'unit_ids': retrieved_unit_ids,
            'paper_ids': retrieved_paper_ids,
            'unit_texts': retrieved_unit_texts
        }
