from autogen_core import SingleThreadedAgentRuntime

from agent_system.agentic.classifier.classifier_agent import ClassifierAgent
from agent_system.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from agent_system.agentic.knowledge.knowledge_agent import KnowledgeAgent
from agent_system.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from agent_system.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent
from agent_system.agentic.collectors.clarification_collector import ClarificationCollector
from agentic.collectors.final_answer_collector import FinalAnswerCollector

from .state import pending_requests

runtime = SingleThreadedAgentRuntime()

async def initialize_runtime():

    await ClassifierAgent.register(
        runtime, "classification", ClassifierAgent
    )

    await QueryRefinerAgent.register(
        runtime, "query_refiner", QueryRefinerAgent
    )

    await KnowledgeAgent.register(
        runtime, "knowledge_agent", KnowledgeAgent
    )

    await BankRAGRetrievalAgent.register(
        runtime, "rag_retrieval", BankRAGRetrievalAgent
    )

    await BankRAGSynthesisAgent.register(
        runtime, "rag_synthesis", BankRAGSynthesisAgent
    )

    await ClarificationCollector.register(
        runtime, "clarification_collector", ClarificationCollector
    )

    await FinalAnswerCollector.register(
        runtime,
        "final_answer_collector",
        lambda: FinalAnswerCollector(pending_requests),
    )

    runtime.start()