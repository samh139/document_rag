# app/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from request_response_router import generate_response, UserMessage

app = FastAPI()

class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    roles: list = []

@app.post("/v1/query")
async def query(req: QueryRequest):
    msg = UserMessage(user_id=req.user_id, session_id=req.session_id, current_query=req.query, roles=req.roles)
    res = await generate_response(msg)
    return res
