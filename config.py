import os
from dotenv import load_dotenv

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
LOCAL_MODEL = os.getenv("GENERAL_PURPOSE_MODEL_BASE_URL", "models/llama3.gguf")