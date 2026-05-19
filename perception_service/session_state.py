from __future__ import annotations

from datetime import datetime, timezone

from .models import SessionStateRecord, SessionStateResponse, WorkingMemoryState


class SessionStateStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionStateRecord] = {}

    def get(self, session_id: str) -> SessionStateResponse:
        record = self._sessions.get(session_id)
        return SessionStateResponse(session=record, found=record is not None)

    def set(self, session_id: str, working_memory_state: WorkingMemoryState) -> SessionStateRecord:
        record = SessionStateRecord(
            session_id=session_id,
            working_memory_state=working_memory_state,
            updated_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = record
        return record

    def clear(self, session_id: str) -> SessionStateResponse:
        record = self._sessions.pop(session_id, None)
        return SessionStateResponse(session=record, found=record is not None)
