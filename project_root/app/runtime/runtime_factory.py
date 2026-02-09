#!project_root/app/runtime/runtime_factory.py
from autogen_core import SingleThreadedAgentRuntime

from app.runtime.event_sink import RuntimeEventSink
from app.agentic.collectors.final_answer_collector import FinalAnswerCollector
from app.agentic.collectors.clarification_collector import ClarificationCollector


def create_runtime(
    sink: RuntimeEventSink
) -> SingleThreadedAgentRuntime:
    runtime = SingleThreadedAgentRuntime()

    runtime.register_agent_instance(FinalAnswerCollector(sink))
    runtime.register_agent_instance(ClarificationCollector(sink))

    return runtime
