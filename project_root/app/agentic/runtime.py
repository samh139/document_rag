#app/agentic/runtime.py

from autogen_core import SingleThreadedAgentRuntime

def create_runtime() -> SingleThreadedAgentRuntime:
    runtime = SingleThreadedAgentRuntime()
    return runtime