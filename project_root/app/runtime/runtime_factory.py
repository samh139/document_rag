# app/runtime/runtime_factory.py

from autogen_core import SingleThreadedAgentRuntime, AgentId

from app.runtime.event_sink import RuntimeEventSink

# Agents
from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from app.agentic.cluster_agent.cluster_agent import ClusterRouterAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from app.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent
from app.agentic.clarification.clarification_agent import ClarificationAgent

# Collectors
from app.agentic.collectors.final_answer_collector import FinalAnswerCollector
from app.agentic.collectors.clarification_collector import ClarificationCollector


async def create_runtime(sink: RuntimeEventSink) -> SingleThreadedAgentRuntime:
    runtime = SingleThreadedAgentRuntime()

    # -------------------------
    # Core Processing Agents
    # -------------------------

    await runtime.register_agent_instance(
        BankEngagementAgent(),
        agent_id=AgentId("BankEngagementAgent", "core"),
    )

    await runtime.register_agent_instance(
        QueryRefinerAgent(),
        agent_id=AgentId("QueryRefinerAgent", "core"),
    )

    await runtime.register_agent_instance(
        ClusterRouterAgent(),
        agent_id=AgentId("ClusterRouterAgent", "core"),
    )

    await runtime.register_agent_instance(
        BankRAGRetrievalAgent(),
        agent_id=AgentId("RAGRetrievalAgent", "core"),
    )

    await runtime.register_agent_instance(
        BankRAGSynthesisAgent(),
        agent_id=AgentId("RAGSynthesisAgent", "core"),
    )

    await runtime.register_agent_instance(
        ClarificationAgent(),
        agent_id=AgentId("ClarificationAgent", "core"),
    )

    # -------------------------
    # Output Collectors
    # -------------------------

    await runtime.register_agent_instance(
        FinalAnswerCollector(sink),
        agent_id=AgentId("FinalAnswerCollector", "collector"),
    )

    await runtime.register_agent_instance(
        ClarificationCollector(sink),
        agent_id=AgentId("ClarificationCollector", "collector"),
    )

    return runtime
