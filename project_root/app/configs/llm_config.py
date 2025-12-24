# app/configs/llm_config.py

import os

# Ollama base
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Models
ENGAGEMENT_MODEL = os.getenv("ENGAGEMENT_MODEL", "gemma3:4b")
QUERY_REFINER_MODEL = os.getenv("QUERY_REFINER_MODEL", "gemma3:4b")
RAG_SYNTHESIS_MODEL = os.getenv("RAG_SYNTHESIS_MODEL", "gemma3:8b")
STM_SUMMARY_MODEL = os.getenv("STM_SUMMARY_MODEL", "gemma3:4b")

# Defaults
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
