# retrieval/chunk_retriever.py
# ---------------------------------------------------------
# Hybrid retriever with CE + ChunkRanker + LLM fallback gatekeeper
# ---------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from agent_system.agentic.utils.retrieval.text_bm25 import BM25Retriever
from agent_system.agentic.utils.es.es_utils import get_es_connection
from agent_system.agentic.model_classes.chunk_data import ChunkData
import re
import os
import math
import logging

from sentence_transformers import CrossEncoder
import warnings

from transformers import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder =  CrossEncoder(ce_model_name,device="mps")

logger = logging.get_logger("ChunkRetriever")

TAG_WEIGHT = 0.30
RRF_K = 30
CE_CANDIDATES_INPUT = 50
BM_CANDIDATES=100
KNN_CANDIDATES=100

# ---------------------------------------------------------
def normalize(vals):
    if not vals:
        return vals
    arr = np.array(vals).reshape(-1, 1)
    scaled = MinMaxScaler().fit_transform(arr)
    return [float(v[0]) for v in scaled]

def ce_raw_to_01(ce_raw: float) -> float:
    return 1.0 / (1.0 + math.exp(-ce_raw))

def ce_raw_to_01_for_list(list:List[float]):
    result=[ ce_raw_to_01(item) for item in list]
    return result

def rrf_merge(*lists):
    fused = {}
    for lst in lists:
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            fused.setdefault(cid, item.copy())
            fused[cid]["rrf"] = fused[cid].get("rrf", 0) + 1 / (RRF_K + rank)

    norms = normalize([v["rrf"] for v in fused.values()])
    for (cid, v), nr in zip(fused.items(), norms):
        v["rrf"] = nr

    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


# ---------------------------------------------------------
class ChunkRetriever:
    def __init__(self, top_k=20, bm25=None):
        self.top_k = top_k
        self.index = "dsprawl_documents"

        # Tests patch these
        self.es = get_es_connection()
        self.bm25 = bm25 or BM25Retriever()

    # -----------------------------------------------------
    def _ann(self, field, qvec, restrict_ids=None):

        body = {
            "size": KNN_CANDIDATES,
            "query": {
                "knn": {
                    "field": field,
                    "query_vector": list(qvec),
                    "k": KNN_CANDIDATES,
                    "num_candidates": KNN_CANDIDATES,
                }
            }
        }

        if restrict_ids:
            body["query"] = {
                "bool": {
                    "must": [{"knn": body["query"]["knn"]}],
                    "filter": [{"terms": {"chunk_id.keyword": restrict_ids}}],
                }
            }

        try:
            print(f"[ChunkRetriever._ann] field={field} restrict_ids count={len(restrict_ids) if restrict_ids else 0}")
            res = self.es.search(index=self.index, body=body)
        except Exception:
            return []

        hits = res.get("hits", {}).get("hits", [])
        results = []
        for h in hits:
            src = h.get("_source", {})
            results.append({
                "chunk_id": src.get("chunk_id"),
                "content": src.get("content"),
                "chunk_metadata": src.get("chunk_metadata"),
                "score": float(h.get("_score", 0)),
            })
        return results

    # -----------------------------------------------------
    def retrieve(self, query: str, qvec, restrict_ids=None) -> List[ChunkData]:
        # ANN
        ann_text = self._ann("vector", qvec, restrict_ids)
        ann_tags = self._ann("tags_vector", qvec, restrict_ids)

        for a in ann_text:
            a["score"] *= (1 - TAG_WEIGHT)
        for a in ann_tags:
            a["score"] *= TAG_WEIGHT

        all_ann = ann_text + ann_tags
        norm = normalize([a["score"] for a in all_ann])
        for a, n in zip(all_ann, norm):
            a["ann"] = n

        # BM25
        bm25_hits = self.bm25.search(query, BM_CANDIDATES, restrict_ids)
        bm_norm = normalize([b.get("score", 0) for b in bm25_hits])
        for b, n in zip(bm25_hits, bm_norm):
            b["bm25"] = n

        fused = rrf_merge(ann_text, ann_tags, bm25_hits)

        # CE
        ce = cross_encoder
        candidates = fused[:CE_CANDIDATES_INPUT]

        pairs = [(query, c.get("content", "") or "") for c in candidates]
        ce_scores = ce_raw_to_01_for_list(list(ce.predict(pairs)))
        
        for c, s in zip(candidates, ce_scores):
            c["ce_score"] = s
        candidates=sorted(candidates, key=lambda x: x["ce_score"], reverse=True)

        results=[]    
        top_k_candidates=candidates[:self.top_k]
        logger.debug(f"top k candidates count ={len(top_k_candidates)}")
        for rank, item in enumerate(top_k_candidates, 1):
            chunk_metadata=item.get("chunk_metadata")
            file_metadata = item.get("file_metadata") or {}
            title = file_metadata.get("title", "")
            content = item.get("content", "")
            #title=item.get("file_metadata",{}).get("title","")
            chunk_tags=[]
            if chunk_metadata:
                chunk_tags=chunk_metadata.get("tags",[])
            
            '''
            content = prepend_file_name_to_images(
                item["content"],
                item["file_name"]
            )

            parent_file = item.get("parent_file", None)
            '''
            ce_score = item["ce_score"]
            results.append(ChunkData(
                chunk_id=item["chunk_id"],
                file=item.get("file_name") or "",
                content=content,
                file_title=title,
                searched_for=query,
                rrf=item.get("rrf", 0),
                bm_score=item.get("bm25", 0),
                relevant_score=ce_score,
                final_score=ce_score,
                final_rank=rank,
                chunk_tags=chunk_tags,
            ))  
        return results



def prepend_file_name_to_images(content: str, file_name: str) -> str:
    # Remove the file extension from file_name if present
    file_base = os.path.splitext(file_name)[0]
    pattern = r'\b([\w-]+\.(?:png|jpg|jpeg))\b'
    def replacer(match):
        filename = match.group(1)
        if filename.startswith(f"{file_base}/"):
            return filename
        return f"{file_base}/{filename}"
    return re.sub(pattern, replacer, content, flags=re.IGNORECASE)
