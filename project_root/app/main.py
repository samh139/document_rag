from fastapi import FastAPI

from app.api.fastapi_route import router as chat_router
from app.agentic.runtime_instance import runtime
from app.agentic.state import response_queue

from app.agentic.collectors.final_answer_collector import FinalAnswerCollector
from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from app.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent
from app.agentic.rag.query_refiner_agent import QueryRefinerAgent


app = FastAPI(title="Banking RAG Assistant")


@app.on_event("startup")
async def startup_event():
    # Register agents ONCE
    await BankEngagementAgent.register(
        runtime, "bank_engagement", BankEngagementAgent
    )

    await QueryRefinerAgent.register(
    runtime, "query_refiner", QueryRefinerAgent,
    )

    await BankRAGRetrievalAgent.register(
        runtime, "bank_rag_retrieval", BankRAGRetrievalAgent
    )

    await BankRAGSynthesisAgent.register(
        runtime, "bank_rag_synthesis", BankRAGSynthesisAgent
    )

    await FinalAnswerCollector.register(
        runtime,
        "final_answer_collector",
        lambda: FinalAnswerCollector(response_queue),
    )

    # 🚀 Start runtime once
    runtime.start()


@app.on_event("shutdown")
async def shutdown_event():
    await runtime.stop_when_idle()


app.include_router(chat_router, prefix="/api")
