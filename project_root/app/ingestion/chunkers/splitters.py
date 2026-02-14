# project_root/app/ingestion/chunkers/splitters.py


from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitters(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
'''

# project_root/app/ingestion/chunkers/splitters.py

import requests
import numpy as np
import re
from typing import List
from langchain_core.documents import Document


class OllamaSemanticChunker:
    """
    Semantic chunker that uses Ollama embeddings and a single percentile-based split threshold.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        percentile: int = 85,
        max_chunk_chars: int = 1200,
        ollama_url: str = "http://localhost:11434/api/embeddings",
        request_timeout: int = 30,
        chunk_overlap: int = 100,  # accepted for compatibility but ignored in this implementation
    ):
        self.model_name = model_name
        self.percentile = int(percentile)
        self.max_chunk_chars = int(max_chunk_chars)
        self.chunk_overlap = int(chunk_overlap)  
        self.ollama_url = ollama_url
        self.request_timeout = request_timeout
        self._embed_cache = {}

    # ---------- Sentence Split ----------
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ---------- Embedding ----------
    def _embed_once(self, text: str):
        key = text.strip()
        if not key:
            return []
        if key in self._embed_cache:
            return self._embed_cache[key]

        resp = requests.post(
            self.ollama_url,
            json={"model": self.model_name, "prompt": key},
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        embedding = None
        if isinstance(data, dict) and "embedding" in data:
            embedding = data["embedding"]
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            embedding = data["data"][0].get("embedding")
        elif isinstance(data, list) and len(data) and isinstance(data[0], dict) and "embedding" in data[0]:
            embedding = data[0]["embedding"]

        if embedding is None:
            raise RuntimeError(f"Unexpected embedding response shape: {data}")

        self._embed_cache[key] = embedding
        return embedding

    def embed_sentences(self, sentences: List[str]):
        return [self._embed_once(s) for s in sentences]

    # ---------- Cosine Similarity ----------
    @staticmethod
    def cosine_sim(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---------- Core Chunk Logic (percentile-only) ----------
    def _chunk_text(self, text: str) -> List[str]:
        sentences = self.split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self.embed_sentences(sentences)

        distances = []
        for i in range(1, len(embeddings)):
            sim = self.cosine_sim(embeddings[i - 1], embeddings[i])
            distances.append(1 - sim)

        # Percentile-based threshold only
        split_threshold = float(np.percentile(distances, self.percentile))

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            distance = distances[i - 1]

            # split when semantic distance is above the percentile threshold OR chunk length exceeds max chars
            if distance > split_threshold or len(" ".join(current_chunk)) > self.max_chunk_chars:
                # finalize previous chunk
                prev_chunk = current_chunk
                chunks.append(" ".join(prev_chunk))

                # start next chunk with `chunk_overlap` sentences from the tail of the previous chunk
                if self.chunk_overlap > 0:
                    overlap_count = min(self.chunk_overlap, len(prev_chunk))
                    if overlap_count > 0:
                        current_chunk = prev_chunk[-overlap_count:].copy()
                    else:
                        current_chunk = []
                else:
                    current_chunk = []

            # always append the current sentence to the active chunk
            current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ---------- LangChain Compatible ----------
    def create_documents(self, texts: List[str]) -> List[Document]:
        documents = []
        for text in texts:
            chunks = self._chunk_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk))
        return documents


# -------- Factory (compatible signature) --------
def get_splitters(chunk_size: int = 1200, chunk_overlap: int = 0, percentile: int = 90) -> OllamaSemanticChunker:
    """
    Returns an OllamaSemanticChunker that uses percentile-based splitting.
    chunk_size -> max_chunk_chars. chunk_overlap is accepted for compatibility but ignored.
    """
    return OllamaSemanticChunker(
        percentile=percentile,
        max_chunk_chars=chunk_size,
        chunk_overlap = chunk_overlap
        )
'''