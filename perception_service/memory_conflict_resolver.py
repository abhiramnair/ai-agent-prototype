from __future__ import annotations

from datetime import datetime, timezone
import logging

from .memory_store import MemoryStore
from .models import (
    MemoryConflictCandidate,
    MemoryConflictResolutionEvaluation,
    MemoryConflictResolutionRequest,
    MemoryConflictResolutionResponse,
    MemoryConflictValue,
    MemoryRecord,
)

logger = logging.getLogger("memory_conflict_resolver_service")


class MemoryConflictResolver:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def resolve(self, request: MemoryConflictResolutionRequest) -> MemoryConflictResolutionResponse:
        source_memories = self._filter_source_memories(request)
        candidates = self._build_candidates(source_memories)[: request.limit]
        resolved_memories: list[MemoryRecord] = []
        switched_value_count = 0

        if request.persist_resolution:
            for candidate in candidates:
                original = next(memory for memory in source_memories if memory.memory_id == candidate.memory_id)
                resolved = self._resolve_memory(original, request.resolution_strategy)
                self.store.save(resolved)
                resolved_memories.append(resolved)
                if resolved.value != original.value:
                    switched_value_count += 1

        response = MemoryConflictResolutionResponse(
            candidates=candidates,
            resolved_memories=resolved_memories,
            evaluation=MemoryConflictResolutionEvaluation(
                source_count=len(source_memories),
                candidate_count=len(candidates),
                resolved_count=len(resolved_memories),
                switched_value_count=switched_value_count,
            ),
        )
        logger.info(
            "memory_conflict_resolution_completed",
            extra={
                "source_count": response.evaluation.source_count,
                "candidate_count": response.evaluation.candidate_count,
                "resolved_count": response.evaluation.resolved_count,
                "switched_value_count": response.evaluation.switched_value_count,
            },
        )
        return response

    def _filter_source_memories(self, request: MemoryConflictResolutionRequest) -> list[MemoryRecord]:
        tag_set = {tag.lower() for tag in request.tags}
        memories = self.store.list_records(include_archived=request.include_archived)
        filtered: list[MemoryRecord] = []
        for memory in memories:
            if request.memory_type and memory.memory_type != request.memory_type:
                continue
            if tag_set and not tag_set.issubset({tag.lower() for tag in memory.tags}):
                continue
            if not memory.metadata.get("conflict_history"):
                continue
            filtered.append(memory)
        filtered.sort(key=lambda item: (-item.contradiction_count, -item.updated_at.timestamp()))
        return filtered

    def _build_candidates(self, memories: list[MemoryRecord]) -> list[MemoryConflictCandidate]:
        candidates: list[MemoryConflictCandidate] = []
        for memory in memories:
            conflicting_values = [
                MemoryConflictValue(
                    value=str(item.get("value", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    source_turn_id=item.get("source_turn_id"),
                    observed_at=self._parse_datetime(item.get("observed_at")),
                )
                for item in memory.metadata.get("conflict_history", [])
                if item.get("value")
            ]
            if not conflicting_values:
                continue
            candidates.append(
                MemoryConflictCandidate(
                    memory_id=memory.memory_id,
                    memory_type=memory.memory_type,
                    key=memory.key,
                    current_value=memory.value,
                    current_confidence=memory.confidence,
                    conflicting_values=conflicting_values,
                    contradiction_count=memory.contradiction_count,
                    severity=self._severity(memory, conflicting_values),
                    suggested_resolution=self._suggested_resolution(memory, conflicting_values),
                )
            )
        return candidates

    def _resolve_memory(self, memory: MemoryRecord, strategy: str) -> MemoryRecord:
        options = [
            {
                "value": memory.value,
                "confidence": memory.confidence,
                "source_turn_id": memory.source_turn_id,
                "observed_at": memory.updated_at.isoformat(),
                "is_current": True,
            }
        ]
        options.extend(memory.metadata.get("conflict_history", []))
        chosen = self._choose_option(options, strategy)
        remaining = [option for option in options if not self._same_option(option, chosen)]
        now = datetime.now(timezone.utc)
        resolved_conflicts = list(memory.metadata.get("resolved_conflicts", []))
        resolved_conflicts.append(
            {
                "resolved_at": now.isoformat(),
                "strategy": strategy,
                "chosen_value": chosen.get("value", ""),
                "discarded_values": [option.get("value", "") for option in remaining if option.get("value")],
            }
        )
        metadata = memory.metadata | {
            "conflict_history": remaining,
            "resolution_status": "resolved",
            "resolved_conflicts": resolved_conflicts,
            "last_resolved_at": now.isoformat(),
        }
        return memory.model_copy(
            update={
                "value": str(chosen.get("value", memory.value)),
                "confidence": float(chosen.get("confidence", memory.confidence)),
                "source_turn_id": chosen.get("source_turn_id") or memory.source_turn_id,
                "metadata": metadata,
                "updated_at": now,
            }
        )

    def _choose_option(self, options: list[dict], strategy: str) -> dict:
        if strategy == "prefer_latest":
            return max(options, key=lambda item: self._parse_datetime(item.get("observed_at")) or datetime.min.replace(tzinfo=timezone.utc))
        if strategy == "prefer_current":
            current = next((item for item in options if item.get("is_current")), None)
            if current is not None:
                return current
        return max(
            options,
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                self._parse_datetime(item.get("observed_at")) or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )

    def _same_option(self, left: dict, right: dict) -> bool:
        return (
            str(left.get("value", "")) == str(right.get("value", ""))
            and float(left.get("confidence", 0.0)) == float(right.get("confidence", 0.0))
            and (left.get("source_turn_id") or "") == (right.get("source_turn_id") or "")
        )

    def _severity(self, memory: MemoryRecord, conflicts: list[MemoryConflictValue]) -> str:
        if memory.contradiction_count >= 3 or len(conflicts) >= 3:
            return "high"
        if memory.contradiction_count >= 2 or len(conflicts) >= 2:
            return "medium"
        return "low"

    def _suggested_resolution(self, memory: MemoryRecord, conflicts: list[MemoryConflictValue]) -> str:
        best_conflict = max((item.confidence for item in conflicts), default=0.0)
        if best_conflict > memory.confidence:
            return "prefer_high_confidence"
        return "prefer_latest"

    def _parse_datetime(self, raw_value: str | datetime | None) -> datetime | None:
        if isinstance(raw_value, datetime):
            return raw_value
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(str(raw_value))
        except ValueError:
            return None
