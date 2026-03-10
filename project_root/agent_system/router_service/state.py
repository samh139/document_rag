import asyncio

pending_requests: dict[str, asyncio.Future] = {}