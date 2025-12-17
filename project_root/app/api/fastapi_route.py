from fastapi import APIRouter
from pydantic import BaseModel

from autogen_core import TopicId

from app.agentic.runtime_instance import runtime
from app.agentic.state import response_queue
from app.agentic.messages import BankUserMessage
from app.agentic.topics import AgenticTopic

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "user-123"
    session_id: str = "sess-1"


@router.post("/chat")
async def chat(req: ChatRequest):
    await runtime.publish_message(
        BankUserMessage(
            content=req.message,
            session_id=req.session_id,
            user_id=req.user_id,
        ),
        topic_id=TopicId(
            AgenticTopic.USER_INPUT.value,
            source="api",
        ),
    )

    result = await response_queue.get()
    return result
