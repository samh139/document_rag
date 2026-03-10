# agents/rag/synthesizer.py
import os, requests
import json

OLLAMA = os.getenv("OLLAMA_URL","http://localhost:11434")
MODEL = os.getenv("SYNTH_MODEL","llama2-mini")

PROMPT_TEMPLATE = """
You are an assistant that answers user questions using the provided context chunks.
Context:
{context}

User question: {question}

Provide a concise answer (2-6 sentences). Also include a "sources" list with chunk ids used.
Return only JSON: {{ "answer":"...", "sources": ["chunk_id1","chunk_id2"], "confidence": 0.9 }}
"""

def synthesize_answer(chunks, question):
    context = "\n\n".join([f"[{c['chunk_id']}] {c['source']['content']}" for c in chunks[:6]])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    body = {"model": MODEL, "input": prompt}
    resp = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    text = out.get("text") or ""
    # extract JSON object from response
    start = text.find("{"); end = text.rfind("}")
    if start!=-1 and end!=-1 and end>start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    # fallback minimal
    return {"answer": text.strip(), "sources": [c["chunk_id"] for c in chunks[:3]], "confidence": 0.5}
