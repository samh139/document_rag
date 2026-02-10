import asyncio
from app.runtime.event_sink import RuntimeEventSink
from app.runtime.runtime_result import RuntimeResult

class TestEventSink(RuntimeEventSink):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def handle_runtime_result(self, result: RuntimeResult):
        await self.queue.put(result)
