# clustering/es_writer.py
# ---------------------------------------------------------
# ES Writer for cluster documents
# ---------------------------------------------------------

from elasticsearch import Elasticsearch
from app.clustering.config import ES_HOST, EMBEDDING_DIM
from app.app_logger import LoggerFactory
logger = LoggerFactory.get_logger("es_writer")


class ESWriter:
    """
    Handles all Elasticsearch writes for cluster documents.
    """

    def __init__(self, index_name: str):
        """
        index_name = the target index (temp index)
        """
        self.index_name = index_name

        # Provide compatibility field expected by cluster_builder
        self.active_index = index_name

        # Real ES client
        self.es = Elasticsearch([ES_HOST])

    # -----------------------------------------------------
    # Create index
    # -----------------------------------------------------

    def create_index(self, index_name: str):
        logger.info(f"[ES] Creating index: {index_name} (dims={EMBEDDING_DIM})")

        body = {
            "mappings": {
                "properties": {
                    "cluster_id": {"type": "keyword"},
                    "level": {"type": "integer"},
                    "parent_cluster": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "chunk_ids": {"type": "keyword"},
                    "subclusters": {"type": "keyword"},
                    "meta_stats": {"type": "object"},
                    "last_updated": {"type": "date"}
                }
            }
        }

        self.es.indices.create(index=index_name, body=body, ignore=400)
    
    # -----------------------------------------------------
    # Write a cluster doc
    # -----------------------------------------------------
    def write_cluster_doc(self, index: str, doc_id: str, body: dict):
        self.es.index(index=index, id=doc_id, document=body)

    # -----------------------------------------------------
    # Update field
    # -----------------------------------------------------
    def update_cluster_field(self, index: str, cluster_id: str, field_name: str, field_value):
        self.es.update(
            index=index,
            id=cluster_id,
            body={"doc": {field_name: field_value}},
            retry_on_conflict=5
        )

    # -----------------------------------------------------
    # Delete index
    # -----------------------------------------------------
    def safe_delete_index(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
