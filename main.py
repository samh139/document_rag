from fastapi import FastAPI
from pydantic import BaseModel
from agents.react.agent import ReActAgent

app = FastAPI()

# initialize the agent
agent = ReActAgent()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_agent(query: QueryRequest):
    response = agent.run(query.query)
    return {"response": response}
