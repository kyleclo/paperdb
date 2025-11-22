import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.papers = None
        self.index_path = Path(__file__).parent / "index.pkl"

    def index(self, papers: list[dict]):
        texts = [f"{p['title']}. {p['abstract']}" for p in papers]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.papers = papers

        with open(self.index_path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "papers": self.papers}, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.papers = data["papers"]

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_k = np.argsort(scores)[::-1][:k]
        return [self.papers[i]["id"] for i in top_k]
