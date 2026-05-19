from __future__ import annotations

import logging

from .memory_store import MemoryStore
from .models import (
    MemoryQueryRequest,
    MemoryRecord,
    MemoryRetrievalEvaluation,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    RetrievedMemory,
)

logger = logging.getLogger("memory_retriever_service")


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(self, request: MemoryRetrievalRequest) -> MemoryRetrievalResponse:
        candidate_records: list[MemoryRecord] = []
        memory_types = request.memory_types or [None]

        for memory_type in memory_types:
            query = MemoryQueryRequest(
                memory_type=memory_type,
                query_text=request.query_text,
                tags=request.tags,
                include_archived=request.include_archived,
                limit=max(request.limit * 3, 10),
            )
            result = self.store.query(query)
            candidate_records.extend(result.memories)

        ranked = self._rank_records(candidate_records, request.query_text, request.tags)
        memories = ranked[: request.limit]
        top_score = memories[0].retrieval_score if memories else 0.0
        response = MemoryRetrievalResponse(
            memories=memories,
            evaluation=MemoryRetrievalEvaluation(
                retrieved_count=len(memories),
                used_type_filters=bool(request.memory_types),
                used_tag_filters=bool(request.tags),
                top_score=round(top_score, 2),
            ),
        )
        logger.info(
            "memory_retrieval_completed",
            extra={
                "query_text": request.query_text,
                "retrieved_count": len(memories),
                "top_score": response.evaluation.top_score,
            },
        )
        return response

    def _rank_records(
        self,
        records: list[MemoryRecord],
        query_text: str,
        tags: list[str],
    ) -> list[RetrievedMemory]:
        deduped: dict[str, MemoryRecord] = {}
        for record in records:
            deduped[record.memory_id] = record

        query_tokens = self._tokenize(query_text)
        tag_tokens = {tag.lower() for tag in tags}
        ranked: list[RetrievedMemory] = []
        for record in deduped.values():
            searchable_tokens = self._tokenize(" ".join([record.key, record.value, " ".join(record.tags), " ".join(record.evidence)]))
            overlap = len(query_tokens.intersection(searchable_tokens))
            tag_overlap = len(tag_tokens.intersection({tag.lower() for tag in record.tags}))
            lexical_score = min(1.0, overlap / max(1, len(query_tokens))) if query_tokens else 0.5
            tag_score = min(1.0, tag_overlap / max(1, len(tag_tokens))) if tag_tokens else 0.0
            confidence_score = record.confidence * 0.35
            reinforcement_score = min(0.2, 0.05 * record.reinforcement_count)
            retrieval_score = min(1.0, 0.35 * lexical_score + 0.1 * tag_score + confidence_score + reinforcement_score)
            ranked.append(
                RetrievedMemory(
                    memory_id=record.memory_id,
                    memory_type=record.memory_type,
                    key=record.key,
                    value=record.value,
                    confidence=record.confidence,
                    retrieval_score=round(retrieval_score, 2),
                    tags=record.tags,
                    source_turn_id=record.source_turn_id,
                    retrieval_reason=self._build_reason(record, overlap, tag_overlap),
                )
            )
        ranked.sort(key=lambda item: (-item.retrieval_score, -item.confidence, item.memory_id))
        return ranked

    def _build_reason(self, record: MemoryRecord, overlap: int, tag_overlap: int) -> str:
        reasons: list[str] = []
        if overlap:
            reasons.append("matched query terms")
        if tag_overlap:
            reasons.append("matched requested tags")
        if record.reinforcement_count > 1:
            reasons.append("reinforced by repeated evidence")
        if not reasons:
            reasons.append("ranked by confidence")
        return ", ".join(reasons)

    def _tokenize(self, text: str) -> set[str]:
        return {token.strip().lower() for token in text.split() if token.strip()}
