from abc import ABC, abstractmethod
from app.runtime.runtime_result import RuntimeResult


class RuntimeEventSink(ABC):
    """
    Runtime pushes results here.
    Session implements this.
    """

    @abstractmethod
    async def handle_runtime_result(
        self,
        result: RuntimeResult
    ):
        ...
