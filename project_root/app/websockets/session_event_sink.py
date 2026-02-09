# project_root/app/websockets/session_event_sink.py

from app.runtime.event_sink import RuntimeEventSink
from app.runtime.runtime_result import RuntimeResult
from app.websockets.session import ConversationSession
from app.websockets.protocol import WSMessage

from app.agentic.messages import (
    ClarificationQuestionMessage,
    FinalAnswerMessage,
)


class SessionEventSink(RuntimeEventSink):
    """
    Concrete sink that:
    - sends WS messages
    - updates ConversationSession state
    """

    def __init__(self, session: ConversationSession):
        self.session = session

    async def handle_runtime_result(self, result: RuntimeResult):
        if result.status == "WAIT":
            await self._handle_wait(result)

        elif result.status == "COMPLETE":
            await self._handle_complete(result)

        elif result.status == "ERROR":
            await self._handle_error(result)

        # CONTINUE → ignore at WS layer
        # runtime keeps running

    # ---------- handlers ----------

    async def _handle_wait(self, result: RuntimeResult):
        """
        Runtime requests clarification.
        """
        question = ClarificationQuestionMessage(**result.payload)

        self.session.mark_waiting(payload=result.payload)

        await self.session.websocket.send_json(
            WSMessage(
                type="CLARIFICATION_QUESTION",
                payload=question.dict(),
            ).dict()
        )

    async def _handle_complete(self, result: RuntimeResult):
        """
        Runtime finished with final answer.
        """
        answer = FinalAnswerMessage(**result.payload)

        self.session.mark_completed()

        await self.session.websocket.send_json(
            WSMessage(
                type="FINAL_ANSWER",
                payload=answer.dict(),
            ).dict()
        )

    async def _handle_error(self, result: RuntimeResult):
        self.session.status = "ERROR"

        await self.session.websocket.send_json(
            WSMessage(
                type="ERROR",
                payload={
                    "reason": result.reason or "Unknown runtime error"
                },
            ).dict()
        )
