# tests/helpers/runtime_bootstrap.py

from app.agentic.runtime_instance import runtime
from app.agentic.collectors.final_answer_collector import FinalAnswerCollector
from app.agentic.state import response_queue

from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from app.agentic.cluster_agent.cluster_agent import ClusterRouterAgent
from app.agentic.clarification.clarification_agent import ClarificationAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from app.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent

async def start_test_runtime():
    await BankEngagementAgent.register(runtime, "bank_engagement", BankEngagementAgent)
    await QueryRefinerAgent.register(runtime, "query_refiner", QueryRefinerAgent)
    await ClusterRouterAgent.register(runtime, "cluster_router", ClusterRouterAgent)
    await ClarificationAgent.register(runtime, "clarification", ClarificationAgent)
    await BankRAGRetrievalAgent.register(runtime, "rag_retrieval", BankRAGRetrievalAgent)
    await BankRAGSynthesisAgent.register(runtime, "rag_synthesis", BankRAGSynthesisAgent)

    await FinalAnswerCollector.register(
        runtime,
        "final_answer_collector",
        lambda: FinalAnswerCollector(response_queue),
    )

    runtime.start()
