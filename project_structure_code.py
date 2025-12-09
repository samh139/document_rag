import os

PROJECT_ROOT = "project_root"

# 💠 Folder + file structure (exactly as designed)
structure = {
    "app": {
        "__init__.py": "",
        "config": {
            "__init__.py": "",
            "base_config.yaml": "",
            "ingestion.yaml": "",
            "retrieval.yaml": "",
            "logging.yaml": "",
            "secrets.env": "",
        },
        "core": {
            "es": {
                "connection.py": "",
                "indices.py": "",
                "queries.py": "",
            },
            "minio": {
                "minio_client.py": "",
            },
            "kafka": {
                "producer.py": "",
            },
            "utils": {
                "file_utils.py": "",
                "text_cleaner.py": "",
                "logger.py": "",
                "id_generator.py": "",
            },
        },
        "ingestion": {
            "__init__.py": "",
            "chunkers": {
                "pdf_chunker.py": "",
                "docx_chunker.py": "",
                "md_chunker.py": "",
                "splitter.py": "",
            },
            "services": {
                "chunks_store_service.py": "",
                "file_ingestion_service.py": "",
            },
            "pipelines": {
                "ingestion_pipeline.py": "",
            },
        },
        "enrichment": {
            "ai_tags.py": "",
            "ai_summary.py": "",
            "ai_embeddings.py": "",
            "postprocess.py": "",
        },
        "retrieval": {
            "hybrid_retriever.py": "",
            "query_classifier.py": "",
            "rrf_fusion.py": "",
            "reranker.py": "",
        },
        "api": {
            "main.py": "",
            "routers": {
                "ingestion_router.py": "",
                "search_router.py": "",
                "health_router.py": "",
            },
            "models": {
                "search_request.py": "",
                "ingestion_request.py": "",
            },
        },
        "workers": {
            "kafka_chunk_worker.py": "",
            "summary_worker.py": "",
        },
    },
    "tests": {
        "ingestion": {},
        "retrieval": {},
        "unit": {},
    },
    "scripts": {
        "create_indices.py": "",
        "sample_ingest.py": "",
        "sample_query.py": "",
    },
    "requirements.txt": "",
    "README.md": "",
    "pyproject.toml": "",
}

# ---------------------------------------------------------
# Recursive function to create folders & files
# ---------------------------------------------------------
def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)

# ---------------------------------------------------------
# Run generator
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 Creating project structure inside: {PROJECT_ROOT}")
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    create_structure(PROJECT_ROOT, structure)
    print("✅ Project structure created successfully!")
