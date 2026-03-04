# project_root/app/ingestion/chunkers/splitters.py


from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitters(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
