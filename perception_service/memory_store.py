from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from uuid import uuid4

from .models import (
    DecayResult,
    MemoryArchiveRequest,
    MemoryArchiveResponse,
    MemoryDecayEvaluation,
    MemoryDecayRequest,
    MemoryDecayResponse,
    MemoryMutationResponse,
    MemoryQueryEvaluation,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryRecord,
    MemoryUpsertRequest,
)

logger = logging.getLogger("memory_store_service")


class MemoryRepository(ABC):
    @abstractmethod
    def upsert(self, request: MemoryUpsertRequest) -> MemoryMutationResponse:
        raise NotImplementedError

    @abstractmethod
    def query(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        raise NotImplementedError

    @abstractmethod
    def archive(self, request: MemoryArchiveRequest) -> MemoryArchiveResponse:
        raise NotImplementedError

    @abstractmethod
    def decay(self, request: MemoryDecayRequest) -> MemoryDecayResponse:
        raise NotImplementedError


class InMemoryMemoryRepository(MemoryRepository):
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def upsert(self, request: MemoryUpsertRequest) -> MemoryMutationResponse:
        now = datetime.now(timezone.utc)
        existing = self._find_existing(request)
        if existing:
            updated = existing.model_copy(
                update={
                    "value": request.value,
                    "confidence": request.confidence,
                    "source_turn_id": request.source_turn_id or existing.source_turn_id,
                    "tags": self._merge_lists(existing.tags, request.tags),
                    "evidence": self._merge_lists(existing.evidence, request.evidence),
                    "metadata": existing.metadata | request.metadata,
                    "updated_at": now,
                    "reinforcement_count": existing.reinforcement_count + 1,
                    "archived": False,
                }
            )
            self._records[updated.memory_id] = updated
            logger.info("memory_upsert_updated", extra={"memory_id": updated.memory_id})
            return MemoryMutationResponse(memory=updated, created=False)

        memory = MemoryRecord(
            memory_id=request.memory_id or f"mem_{uuid4().hex[:12]}",
            memory_type=request.memory_type,
            key=request.key,
            value=request.value,
            confidence=request.confidence,
            source_turn_id=request.source_turn_id,
            tags=request.tags,
            evidence=request.evidence,
            created_at=now,
            updated_at=now,
            last_accessed_at=None,
            reinforcement_count=1,
            contradiction_count=0,
            archived=False,
            metadata=request.metadata,
        )
        self._records[memory.memory_id] = memory
        logger.info("memory_upsert_created", extra={"memory_id": memory.memory_id})
        return MemoryMutationResponse(memory=memory, created=True)

    def query(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        matches: list[MemoryRecord] = []
        query_tokens = self._tokenize(request.query_text or "")
        tag_set = {tag.lower() for tag in request.tags}

        for memory in self._records.values():
            if memory.archived and not request.include_archived:
                continue
            if request.memory_type and memory.memory_type != request.memory_type:
                continue
            if tag_set and not tag_set.issubset({tag.lower() for tag in memory.tags}):
                continue
            if query_tokens and not self._matches_query(memory, query_tokens):
                continue
            accessed = memory.model_copy(update={"last_accessed_at": datetime.now(timezone.utc)})
            self._records[memory.memory_id] = accessed
            matches.append(accessed)

        ranked = sorted(
            matches,
            key=lambda record: (record.archived, -record.confidence, -record.reinforcement_count, record.updated_at.timestamp()),
            reverse=False,
        )[: request.limit]
        return MemoryQueryResponse(
            memories=ranked,
            evaluation=MemoryQueryEvaluation(
                total_matches=len(matches),
                used_type_filter=bool(request.memory_type),
                used_text_filter=bool(query_tokens),
                used_tag_filter=bool(tag_set),
            ),
        )

    def archive(self, request: MemoryArchiveRequest) -> MemoryArchiveResponse:
        memory = self._records[request.memory_id]
        updated = memory.model_copy(
            update={
                "archived": True,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._records[updated.memory_id] = updated
        logger.info("memory_archived", extra={"memory_id": updated.memory_id})
        return MemoryArchiveResponse(memory=updated, archived=True)

    def decay(self, request: MemoryDecayRequest) -> MemoryDecayResponse:
        now = datetime.now(timezone.utc)
        results: list[DecayResult] = []
        archived_count = 0

        for memory_id, memory in list(self._records.items()):
            if memory.archived and not request.include_archived:
                continue
            strength, reason = self._compute_strength(memory, now, request.max_idle_days)
            should_archive = strength < request.threshold
            updated = memory
            if should_archive and not memory.archived:
                updated = memory.model_copy(
                    update={
                        "archived": True,
                        "updated_at": now,
                    }
                )
                self._records[memory_id] = updated
                archived_count += 1
            results.append(
                DecayResult(
                    memory_id=memory_id,
                    strength=round(strength, 2),
                    archived=updated.archived,
                    reason=reason,
                )
            )

        return MemoryDecayResponse(
            results=results,
            evaluation=MemoryDecayEvaluation(
                processed_count=len(results),
                archived_count=archived_count,
                threshold=request.threshold,
            ),
        )

    def _find_existing(self, request: MemoryUpsertRequest) -> MemoryRecord | None:
        if request.memory_id and request.memory_id in self._records:
            return self._records[request.memory_id]
        for memory in self._records.values():
            if memory.memory_type == request.memory_type and memory.key == request.key and not memory.archived:
                return memory
        return None

    def _matches_query(self, memory: MemoryRecord, query_tokens: set[str]) -> bool:
        searchable = " ".join(
            [
                memory.key,
                memory.value,
                " ".join(memory.tags),
                " ".join(memory.evidence),
            ]
        ).lower()
        return query_tokens.issubset(self._tokenize(searchable))

    def _tokenize(self, text: str) -> set[str]:
        return {token.strip().lower() for token in text.split() if token.strip()}

    def _merge_lists(self, existing: list[str], new_values: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in existing + new_values:
            cleaned = value.strip()
            key = cleaned.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
        return merged

    def _compute_strength(
        self,
        memory: MemoryRecord,
        now: datetime,
        max_idle_days: int,
    ) -> tuple[float, str]:
        age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400.0)
        last_seen = memory.last_accessed_at or memory.updated_at
        idle_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
        age_penalty = min(0.35, age_days / 365.0)
        idle_penalty = min(0.3, idle_days / max(1, max_idle_days) * 0.3)
        reinforcement_bonus = min(0.25, 0.05 * memory.reinforcement_count)
        contradiction_penalty = min(0.35, 0.12 * memory.contradiction_count)
        strength = memory.confidence + reinforcement_bonus - contradiction_penalty - age_penalty - idle_penalty
        bounded = max(0.0, min(1.0, strength))
        reason = (
            f"confidence={memory.confidence:.2f}, reinforcement_bonus={reinforcement_bonus:.2f}, "
            f"contradiction_penalty={contradiction_penalty:.2f}, age_penalty={age_penalty:.2f}, "
            f"idle_penalty={idle_penalty:.2f}"
        )
        return bounded, reason


class MemoryStore:
    def __init__(self, repository: MemoryRepository | None = None) -> None:
        self.repository = repository or InMemoryMemoryRepository()

    def upsert(self, request: MemoryUpsertRequest) -> MemoryMutationResponse:
        return self.repository.upsert(request)

    def query(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        return self.repository.query(request)

    def archive(self, request: MemoryArchiveRequest) -> MemoryArchiveResponse:
        return self.repository.archive(request)

    def decay(self, request: MemoryDecayRequest) -> MemoryDecayResponse:
        return self.repository.decay(request)
