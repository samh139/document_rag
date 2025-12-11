# project_root/app/ingestion/chunkers/splitters.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitters(chunk_size: int = 1500, chunk_overlap: int = 150):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
