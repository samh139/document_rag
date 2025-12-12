# agents/rag/embedding_client.py
import os, requests
OLLAMA = os.getenv("OLLAMA_URL","http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL","nomic-embed-text")

def embed_text(text: str):
    body = {"model": EMBED_MODEL, "input": [text]}
    resp = requests.post(f"{OLLAMA}/api/embed", json=body, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    # expects {"model":..., "embeddings":[[...]...]}
    if isinstance(out, dict) and "embeddings" in out and len(out["embeddings"])>0:
        return out["embeddings"][0]
    # fallback: try other keys
    return out
