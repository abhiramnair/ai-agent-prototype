from __future__ import annotations

from collections import defaultdict
import json
import logging

from .memory_store import MemoryStore
from .models import (
    ConsolidationCandidate,
    MemoryArchiveRequest,
    MemoryConsolidationEvaluation,
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    MemoryQueryRequest,
    MemoryRecord,
    MemoryUpsertRequest,
)

logger = logging.getLogger("memory_consolidator_service")


class MemoryConsolidator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def consolidate(self, request: MemoryConsolidationRequest) -> MemoryConsolidationResponse:
        source_memories = self.store.query(
            MemoryQueryRequest(
                memory_type=request.memory_type,
                tags=request.tags,
                include_archived=request.include_archived,
                limit=request.limit,
            )
        ).memories

        candidates = self._build_candidates(source_memories, request.min_group_size)
        consolidated_memories: list[MemoryRecord] = []
        archived_source_ids: list[str] = []

        if request.persist_consolidated:
            for candidate in candidates:
                if candidate.commit_decision != "commit":
                    continue
                result = self.store.upsert(
                    MemoryUpsertRequest(
                        memory_type=candidate.target_memory_type,
                        key=candidate.target_key,
                        value=candidate.target_value,
                        confidence=candidate.confidence,
                        tags=["consolidated", candidate.group_key],
                        evidence=[candidate.rationale],
                        metadata={
                            "source_memory_ids": candidate.source_memory_ids,
                            "group_key": candidate.group_key,
                        },
                    )
                )
                consolidated_memories.append(result.memory)
                if request.archive_sources:
                    for memory_id in candidate.source_memory_ids:
                        self.store.archive(MemoryArchiveRequest(memory_id=memory_id))
                        archived_source_ids.append(memory_id)

        response = MemoryConsolidationResponse(
            candidates=candidates,
            consolidated_memories=consolidated_memories,
            archived_source_ids=archived_source_ids,
            evaluation=MemoryConsolidationEvaluation(
                source_count=len(source_memories),
                candidate_count=len(candidates),
                consolidated_count=len(consolidated_memories),
                archived_source_count=len(archived_source_ids),
            ),
        )
        logger.info(
            "memory_consolidation_completed",
            extra={
                "source_count": response.evaluation.source_count,
                "candidate_count": response.evaluation.candidate_count,
                "consolidated_count": response.evaluation.consolidated_count,
            },
        )
        return response

    def _build_candidates(
        self,
        memories: list[MemoryRecord],
        min_group_size: int,
    ) -> list[ConsolidationCandidate]:
        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for memory in memories:
            grouped[self._group_key(memory)].append(memory)

        candidates: list[ConsolidationCandidate] = []
        for group_key, group_memories in grouped.items():
            reinforcement_total = sum(memory.reinforcement_count for memory in group_memories)
            if len(group_memories) < min_group_size and reinforcement_total < min_group_size:
                continue
            target_memory_type, target_key, target_value = self._build_consolidated_payload(group_key, group_memories)
            average_confidence = sum(memory.confidence for memory in group_memories) / max(1, len(group_memories))
            confidence = min(1.0, round(average_confidence + min(0.15, 0.02 * reinforcement_total), 2))
            candidates.append(
                ConsolidationCandidate(
                    group_key=group_key,
                    target_memory_type=target_memory_type,
                    target_key=target_key,
                    target_value=target_value,
                    confidence=confidence,
                    source_memory_ids=[memory.memory_id for memory in group_memories],
                    rationale=(
                        f"Consolidated {len(group_memories)} memories with reinforcement_total={reinforcement_total} "
                        f"into a reusable {target_memory_type} memory."
                    ),
                    commit_decision="commit",
                )
            )

        candidates.sort(key=lambda item: (-len(item.source_memory_ids), -item.confidence, item.group_key))
        return candidates

    def _group_key(self, memory: MemoryRecord) -> str:
        key_parts = [part for part in memory.key.split(".") if part]
        if memory.memory_type == "episodic" and len(key_parts) > 2:
            key_parts = key_parts[:-1]
        if key_parts and key_parts[0] == "response_policy" and len(key_parts) > 2:
            key_parts = key_parts[:2]
        return ".".join(key_parts[:3]) or memory.key

    def _build_consolidated_payload(self, group_key: str, memories: list[MemoryRecord]) -> tuple[str, str, str]:
        if group_key.startswith("response_policy"):
            return self._build_policy_summary(group_key, memories)
        return self._build_semantic_summary(group_key, memories)

    def _build_policy_summary(self, group_key: str, memories: list[MemoryRecord]) -> tuple[str, str, str]:
        merged: dict[str, str] = {}
        for memory in memories:
            try:
                payload = json.loads(memory.value)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if key not in merged and isinstance(value, str):
                    merged[key] = value
        merged["consolidated_from_count"] = str(len(memories))
        return "semantic", f"consolidated.{group_key}", json.dumps(merged)

    def _build_semantic_summary(self, group_key: str, memories: list[MemoryRecord]) -> tuple[str, str, str]:
        values = self._dedupe([memory.value for memory in memories])
        summary = f"Consolidated memory for {group_key}: " + " | ".join(values[:3])
        return "semantic", f"consolidated.{group_key}", summary

    def _dedupe(self, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output
