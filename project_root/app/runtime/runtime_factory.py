# app/runtime/runtime_factory.py

from autogen_core import SingleThreadedAgentRuntime, AgentId

from app.runtime.event_sink import RuntimeEventSink
from app.agentic.collectors.final_answer_collector import FinalAnswerCollector
from app.agentic.collectors.clarification_collector import ClarificationCollector


async def create_runtime(sink: RuntimeEventSink) -> SingleThreadedAgentRuntime:
    runtime = SingleThreadedAgentRuntime()

    await runtime.register_agent_instance(
        FinalAnswerCollector(sink),
        agent_id=AgentId("FinalAnswerCollector", "collector"),
    )

    await runtime.register_agent_instance(
        ClarificationCollector(sink),
        agent_id=AgentId("ClarificationCollector", "collector"),
    )

    return runtime
