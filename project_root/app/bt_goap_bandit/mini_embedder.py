# ==========================================================
# mini_embedder.py — Local MiniLM Embeddings (Ultra-Fast)
# ==========================================================
# Provides:
#   • embed_text(text) → 384-dim vector
#   • embed_batch(list[str]) → list[vectors]
#
# Latency: ~2–5 ms per call (CPU) for single sentence
# Model: all-MiniLM-L6-v2 (from sentence-transformers)
# ==========================================================

import numpy as np
from sentence_transformers import SentenceTransformer

class MiniLMEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str):
        if not text:
            return np.zeros(self.dim)
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, sentences):
        if not sentences:
            return []
        return self.model.encode(sentences, normalize_embeddings=True).tolist()


# Create singleton (import-safe)
minilm = MiniLMEmbedder()
