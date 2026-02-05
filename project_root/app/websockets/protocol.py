#!project_root/app/websockets/protocol.py

from pydantic import BaseModel
from typing import Literal, Any


# Every WS message uses this envelope
class WSMessage(BaseModel):
    type: Literal[
        # inbound
        "BANK_USER_MESSAGE",
        "CLARIFICATION_REPLY",

        # outbound
        "CLARIFICATION_QUESTION",
        "FINAL_ANSWER",

        # optional / infra
        "ERROR"
    ]
    payload: Any
