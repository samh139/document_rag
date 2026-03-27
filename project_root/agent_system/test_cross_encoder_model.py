import warnings
from transformers import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is the capital of France?"
docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is in Spain."
]

pairs = [(query, doc) for doc in docs]
scores = model.predict(pairs)

for doc, score in zip(docs, scores):
    print(f"Score: {score:.4f} | Doc: {doc}")