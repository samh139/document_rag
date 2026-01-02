#app/agentic/state.py

import asyncio

# Global response queue for final answers
response_queue: asyncio.Queue = asyncio.Queue()
