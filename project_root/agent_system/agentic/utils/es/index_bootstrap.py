# app/memory_team/ltm/index_bootstrap.py
from agent_system.agentic.utils.es.create_es_index import  ltm_index_name, ltm_mapping
from agent_system.agentic.utils.es.create_es_index import agent_responses_index_name, agent_responses_mapping
from agent_system.agentic.utils.es.es_utils import get_es_connection

es = get_es_connection()

def ensure_ltm_index():
    if not es.indices.exists(index=ltm_index_name):
        es.indices.create(index=ltm_index_name, body=ltm_mapping)
        print(f"Created LTM index: {ltm_index_name}")
    else:
        print(f"Index '{ltm_index_name}' already exists")

def ensure_agent_responses_index():
    if not es.indices.exists(index=agent_responses_index_name):
        es.indices.create(index=agent_responses_index_name, body=agent_responses_mapping)
        print(f"Created agent responses index: {agent_responses_index_name}")
    else:
        print(f"Index '{agent_responses_index_name}' already exists")
