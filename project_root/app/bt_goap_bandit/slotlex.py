# ==========================================================
# slotlex.py — MiniLM + ES Retrieval, Gemini Flash ONLY for Semantic Tags
# ==========================================================

import logging
import json
import numpy as np
from typing import List, Dict, Any, Union

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import requests
import re
from service.retrieval.full_chunk_retriever import fetch_relavant_result_by_query
from app_configs.llm_config import fire_fast_modal_request_chat_for_force_json,fire_fast_modal_request_chat_get_dict
from app_logger import get_app_logger
from app_configs.app_env import app_env
from model_classes.intent_agent_response import Enriched, Summary

logger = get_app_logger(__name__)

# -------------------------------------------------------------------
# Embedding Model (MiniLM)
# -------------------------------------------------------------------
logger.info("[SlotLex] 🔄 Loading MiniLM semantic embedder...")

# -------------------------------------------------------------------
# Gemini Flash (Semantic Tags)
# -------------------------------------------------------------------

def _gemini_tags(context: str) -> List[str]:
    """Call Gemini Flash and robustly extract semantic tags."""
    if not context.strip():
        logger.warning("[SlotLex] ⚠ Empty context sent to Gemini → no tags")
        return []
    
    system_prompt="""
        - Extract 3–5 short semantic tags using the provided Content.
        - 1–2 words each.
        - STRICT JSON ONLY. Respond in exactly this format:
            {"tags": ["tag1", "tag2", "tag3"]}
        - Do not output anything else: no prose, no explanations.
        - DO NOT return a bare list.

        """
    user_prompt= f"Content:\n\n {context}"
    llm_result=fire_fast_modal_request_chat_get_dict(system_prompt=system_prompt,user_prompt=user_prompt)
    tags=[]
    if isinstance(llm_result, dict):
        tags=llm_result.get("tags")
    else:
        tags=llm_result
    logger.info(f"semantic tags fetched {tags}")
    return tags

   
# -------------------------------------------------------------------
# ES Connect
# -------------------------------------------------------------------
def _connect_es(host):
    try:
        es = Elasticsearch(hosts=[host])
        es.ping()
        logger.info(f"[SlotLex] Connected → {host}")
        return es
    except Exception as e:
        logger.error(f"[SlotLex] Failed ES connect: {e}")
        return None



# -------------------------------------------------------------------
# SlotLex — Final Class
# -------------------------------------------------------------------
class SlotLex:

    def __init__(self, es_host=app_env.get_es_host(), index=app_env.get_es_chunks_index()):
        self.es = _connect_es(es_host)
        self.index = index

    def enrich(self, text, intent=None, entities=None)->Enriched:

        # ES Retrieval + Ranking
        chunks=fetch_relavant_result_by_query(query=text,top_k=4)
        # Safely extract top contexts
        top_ids = []
        top_texts = []
        scores:List[float]=[]
        chunk_tags=[]
        for chunk in chunks:
            cid = chunk.chunk_id
            ctext = chunk.content
            scores.append(chunk.final_score)
            chunk_tags.extend(chunk.chunk_tags)

            if cid:
                top_ids.append(cid)
            if ctext.strip():
                top_texts.append(ctext)

        # Gemini Flash Semantic Tags
        context = "\n\n".join(top_texts)
        #semantic_tags = _gemini_tags(context)
        

        return Enriched(
            text=text,
            intent=intent,
            entities= entities ,
            semantic_tags= chunk_tags,
            retrieved_contexts= top_texts,
            summary= Summary(
                num_chunks= len(chunks),
                top_chunks= top_ids,
                tags= chunk_tags,
                scores=scores,
                context_chars= len(context)
            )
        )

# Global singleton — fully compatible
_slotlex = SlotLex()

def enrich_slots(text, intent=None, entities=None):
    return _slotlex.enrich(text, intent, entities)
