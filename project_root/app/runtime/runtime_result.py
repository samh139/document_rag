#!project_root/app/runtime/runtime_result.py
from typing import Literal, Optional, Any

RuntimeStatus = Literal[
    "CONTINUE",
    "WAIT",
    "COMPLETE",
    "ERROR"
]

class RuntimeResult:
    def __init__(
        self,
        status: RuntimeStatus,
        payload: Optional[Any] = None,
        reason: Optional[str] = None
    ):
        self.status = status
        self.payload = payload
        self.reason = reason
