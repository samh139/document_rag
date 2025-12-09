from elasticsearch import Elasticsearch
from config import ES_HOST

class ESRetriever:
    def __init__(self, index="faq_index"):
        self.es = Elasticsearch(ES_HOST)
        self.index = index

    def search(self, query, k=3):
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title"]
                }
            },
            "size": k
        }

        res = self.es.search(index=self.index, body=body)
        hits = [h["_source"]["content"] for h in res["hits"]["hits"]]
        return hits
