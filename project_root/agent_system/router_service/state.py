import asyncio

# session_id -> Future
pending_requests: dict[str, asyncio.Future] = {}