from tests.helpers.test_event_sink import TestEventSink
from app.runtime.runtime_factory import create_runtime

async def start_test_runtime():
    sink = TestEventSink()
    runtime = await create_runtime(sink)
    return runtime, sink
