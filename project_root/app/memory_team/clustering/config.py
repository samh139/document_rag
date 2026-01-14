"""
Configuration for D-Sprawl Clustering Pipeline (Pointer Model)
Author: Yogesh
Version: 2.0
Updated: Dec 2025 (Stabilized Clustering Parameters)

This module defines the static configuration for the nightly clustering
pipeline using the pointer-model embeddings. The parameters below are tuned
to prevent cluster collapse, ensure semantic separation, and produce stable,
hierarchical clusters suited for mixed-domain enterprise documents.
"""
#from configs.app_env import app_env
import os

# ----------------------------------------------------------------------
# Elasticsearch Host (HARD-CODED as requested)
# ----------------------------------------------------------------------
#ES_HOST = app_env.get_es_host()
ES_HOST = os.getenv("ES_HOST","http://localhost:9200")

# ----------------------------------------------------------------------
# Index Names
# ----------------------------------------------------------------------
# Source index for chunked embeddings
CHUNKS_INDEX = "dsprawl_documents"
EMBEDDING_DIM = 768   # <-- set this to the ACTUAL dim of nomic-embed-text


# Alias pointing to the active cluster index
CLUSTERS_ALIAS = "clusters_v2"

# Temporary index used during nightly rebuild
TEMP_INDEX_PREFIX = "clusters_v2_temp_"

# ----------------------------------------------------------------------
# LEVEL-1 CLUSTERING — AGGLOMERATIVE
# ----------------------------------------------------------------------
"""
Pointer-model embeddings (MiniLM-family, 384-d) show optimal semantic
separation at cosine distance ~0.55–0.70. A strict threshold is required
to prevent unrelated documents from collapsing into a giant cluster.
"""

# Cosine distance threshold (lower = stricter separation)
AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.65

# Small clusters typically represent niche semantic concepts;
# they should not be force-merged early.
AGGLOMERATIVE_MIN_CLUSTER_SIZE = 5


# ----------------------------------------------------------------------
# LEVEL-2 CLUSTERING — ADAPTIVE K-MEANS
# ----------------------------------------------------------------------
"""
Subclustering is used to break large Level-1 clusters into more
coherent subtopics. The following parameters control whether a cluster
should be subdivided and the range of K values for adaptive K-means.
"""

# Minimum size for a Level-1 cluster to be subdivided
SUBCLUSTER_MIN_SIZE = 150

# Clusters larger than this threshold must be subdivided.
# This prevents runaway clusters (e.g., > 9000 chunks).
SUBCLUSTER_MAX_SIZE = 800

# Range of K for adaptive subclustering
SUBCLUSTER_MIN_K = 3
SUBCLUSTER_MAX_K = 8


# ----------------------------------------------------------------------
# Batch Job Settings
# ----------------------------------------------------------------------
"""
Vector loading is memory- and IO-heavy. Conservative batching prevents
pressure on the ES cluster and ensures smooth nightly execution.
"""

BATCH_SIZE = 3000
MAX_CHUNKS = 2_000_000     # scalability guard for large tenants
JOB_START_TIME = "02:00"   # informational only


# ----------------------------------------------------------------------
# Logging / Debug Options
# ----------------------------------------------------------------------
VERBOSE = True

# When True: simulate job without writing clusters to Elasticsearch.
DRY_RUN = False
 