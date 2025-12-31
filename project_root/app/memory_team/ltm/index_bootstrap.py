# app/memory_team/ltm/index_bootstrap.py
from app.memory_team.ltm.create_ltm_index import es, index_name, mapping

def ensure_ltm_index():
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Created LTM index: {index_name}")