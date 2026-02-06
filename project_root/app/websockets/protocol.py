from pydantic import BaseModel
from typing import Literal, Any


class WSMessage(BaseModel):
    type: Literal[
        # inbound
        "BANK_USER_MESSAGE",
        "CLARIFICATION_REPLY",

        # outbound
        "CLARIFICATION_QUESTION",
        "FINAL_ANSWER",

        # infra
        "ERROR",
    ]
    payload: Any
