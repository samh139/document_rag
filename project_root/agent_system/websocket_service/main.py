from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio

from kafka_producer import send_message
from kafka_consumer import consume_loop

app = FastAPI()


# 🔹 Connection Manager
class ConnectionManager:
    def __init__(self):
        self.connections = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.connections.pop(session_id, None)

    async def send(self, session_id: str, message: dict):
        ws = self.connections.get(session_id)
        if ws:
            await ws.send_json(message)


manager = ConnectionManager()


# 🔹 Start Kafka consumer in background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_loop(manager))


# 🔹 WebSocket Endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            print("Data Received :", data)

            send_message(
                topic="chat-requests",
                key=session_id,
                value={
                    "session_id": session_id,
                    "message": data["message"]
                }
            )

    except WebSocketDisconnect:
        manager.disconnect(session_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)