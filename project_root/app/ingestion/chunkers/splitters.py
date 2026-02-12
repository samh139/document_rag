# project_root/app/ingestion/chunkers/splitters.py

'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitters(chunk_size: int = 1500, chunk_overlap: int = 150):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
'''

# project_root/app/ingestion/chunkers/splitters.py

import requests
import numpy as np
import re
from typing import List
from langchain_core.documents import Document


class OllamaSemanticChunker:
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        method: str = "percentile",
        percentile: int = 90,
        threshold: float = 0.8,
        std_multiplier: float = 1.0,
        max_chunk_chars: int = 1200,
        ollama_url: str = "http://localhost:11434/api/embeddings"
    ):
        self.model_name = model_name
        self.method = method
        self.percentile = percentile
        self.threshold = threshold
        self.std_multiplier = std_multiplier
        self.max_chunk_chars = max_chunk_chars
        self.ollama_url = ollama_url

    # ---------- Sentence Split ----------
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ---------- Embedding ----------
    def embed(self, text: str):
        response = requests.post(
            self.ollama_url,
            json={"model": self.model_name, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_sentences(self, sentences: List[str]):
        return [self.embed(s) for s in sentences]

    # ---------- Cosine Similarity ----------
    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ---------- Core Chunk Logic ----------
    def _chunk_text(self, text: str) -> List[str]:
        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            return sentences

        embeddings = self.embed_sentences(sentences)

        distances = []
        for i in range(1, len(embeddings)):
            sim = self.cosine_sim(embeddings[i - 1], embeddings[i])
            distances.append(1 - sim)

        # determine threshold
        if self.method == "percentile":
            split_threshold = np.percentile(distances, self.percentile)

        elif self.method == "standard_deviation":
            mean = np.mean(distances)
            std = np.std(distances)
            split_threshold = mean + self.std_multiplier * std

        elif self.method == "fixed":
            split_threshold = 1 - self.threshold

        else:
            raise ValueError("Invalid method")

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            distance = distances[i - 1]

            if distance > split_threshold or \
               len(" ".join(current_chunk)) > self.max_chunk_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
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


# -------- Factory --------
def get_splitters(chunk_size=1200, overlap=0):
    return OllamaSemanticChunker(
        method="percentile",
        percentile=90,
        max_chunk_chars=chunk_size
    )

