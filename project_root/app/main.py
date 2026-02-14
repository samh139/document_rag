from fastapi import FastAPI

from app.websockets.router import router as ws_router
from app.memory_team.ltm.index_bootstrap import ensure_ltm_index

app = FastAPI(title="Banking RAG Assistant")


@app.on_event("startup")
async def startup_event():
    # Only global infra things here
    ensure_ltm_index()


app.include_router(ws_router)
