from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    """Get sentence embedding"""
    return model.encode([text])[0]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query: str, corpus: List[str], top_k=5) -> List[Dict]:
    """Semantic search implementation"""
    query_embed = model.encode([query])
    corpus_embeds = model.encode(corpus)
    
    similarities = np.dot(query_embed, corpus_embeds.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [{
        "text": corpus[i],
        "score": float(similarities[i])
    } for i in top_indices]