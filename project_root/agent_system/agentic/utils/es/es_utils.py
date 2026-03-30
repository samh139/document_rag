from elasticsearch import Elasticsearch

#using elastic search server 
ES_HOST =  "http://localhost:9200"


def es_connect(es_host:str):
    return Elasticsearch(es_host)

es_connection=es_connect(es_host=ES_HOST)

def get_es_connection()->Elasticsearch:
    global es_connection
    if  es_connection:
        return es_connection
    
    es_connection=es_connect(es_host=ES_HOST)
    return es_connection