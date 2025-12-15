# project_root/app/api/fastapi_route.py

from fastapi import APIRouter
from pydantic import BaseModel
from app.request_response_router import RequestResponseRouter

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_acl: list[str] | None = None

@router.post("/chat")
def chat(req: ChatRequest):
    return RequestResponseRouter.handle(
        user_input=req.message,
        user_acl=req.user_acl
    )
