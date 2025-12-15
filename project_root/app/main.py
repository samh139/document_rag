# project_root/app/main.py

from fastapi import FastAPI
from app.api.fastapi_route import router as chat_router

app = FastAPI(title="Banking RAG Assistant")

app.include_router(chat_router, prefix="/api")
