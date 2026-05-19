from __future__ import annotations

from datetime import datetime
import logging
from uuid import uuid4

from .memory_store import MemoryStore
from .models import (
    MemoryCandidate,
    MemoryCommitEvaluation,
    MemoryCommitRequest,
    MemoryCommitResponse,
    MemoryRecord,
    MemoryUpsertRequest,
)

logger = logging.getLogger("memory_committer_service")


class MemoryCommitter:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def commit(self, request: MemoryCommitRequest) -> MemoryCommitResponse:
        candidates = self._build_candidates(request)
        committed_memories: list[MemoryRecord] = []

        if request.persist_committed:
            for candidate in candidates:
                if candidate.commit_decision != "commit":
                    continue
                upsert = self.store.upsert(
                    MemoryUpsertRequest(
                        memory_type=candidate.candidate_type,
                        key=candidate.key,
                        value=candidate.content,
                        confidence=candidate.trust_score,
                        source_turn_id=candidate.source_turn_id,
                        tags=candidate.tags,
                        evidence=[candidate.rationale],
                        metadata={
                            "novelty_score": candidate.novelty_score,
                            "importance_score": candidate.importance_score,
                            "repeat_score": candidate.repeat_score,
                            "future_utility_score": candidate.future_utility_score,
                            "explicit_user_signal": candidate.explicit_user_signal,
                        },
                    )
                )
                committed_memories.append(upsert.memory)

        committed_count = sum(1 for candidate in candidates if candidate.commit_decision == "commit")
        response = MemoryCommitResponse(
            candidates=candidates,
            committed_memories=committed_memories,
            evaluation=MemoryCommitEvaluation(
                candidate_count=len(candidates),
                committed_count=committed_count,
                discarded_count=len(candidates) - committed_count,
                persistence_enabled=request.persist_committed,
            ),
        )
        logger.info(
            "memory_commit_completed",
            extra={
                "turn_id": request.turn_input.turn_id,
                "candidate_count": response.evaluation.candidate_count,
                "committed_count": response.evaluation.committed_count,
            },
        )
        return response

    def _build_candidates(self, request: MemoryCommitRequest) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        turn = request.turn_input
        perception = request.perception_state
        critic = request.critic_review

        if perception.salience_signals.contains_preference:
            candidates.append(
                self._make_candidate(
                    candidate_type="preference",
                    key=f"preference.{self._slugify(perception.topic)}",
                    content=turn.message_text.strip(),
                    source_turn_id=turn.turn_id,
                    novelty_score=0.62,
                    importance_score=0.88,
                    repeat_score=0.55,
                    trust_score=0.9,
                    future_utility_score=0.87,
                    explicit_user_signal=True,
                    tags=["preference", "conversation"],
                    rationale="Explicit user preference signal detected in the turn.",
                )
            )

        if perception.salience_signals.contains_correction:
            candidates.append(
                self._make_candidate(
                    candidate_type="correction",
                    key=f"correction.{self._slugify(perception.topic)}",
                    content=turn.message_text.strip(),
                    source_turn_id=turn.turn_id,
                    novelty_score=0.58,
                    importance_score=0.92,
                    repeat_score=0.5,
                    trust_score=0.93,
                    future_utility_score=0.89,
                    explicit_user_signal=True,
                    tags=["correction", "alignment"],
                    rationale="User correction should be persisted for future alignment.",
                )
            )

        if perception.salience_signals.contains_commitment:
            candidates.append(
                self._make_candidate(
                    candidate_type="procedural",
                    key=f"procedural.{self._slugify(perception.topic)}",
                    content=request.working_memory_state.active_goal,
                    source_turn_id=turn.turn_id,
                    novelty_score=0.45,
                    importance_score=0.74,
                    repeat_score=0.48,
                    trust_score=0.82,
                    future_utility_score=0.8,
                    explicit_user_signal=True,
                    tags=["procedural", "commitment"],
                    rationale="The turn includes a direction-setting commitment that may guide future steps.",
                )
            )

        if critic and critic.passed and perception.primary_intent in {"ask_explanation", "request_action"}:
            candidates.append(
                self._make_candidate(
                    candidate_type="episodic",
                    key=f"episode.{self._slugify(perception.topic)}.{turn.turn_id}",
                    content=f"{perception.primary_intent}: {turn.message_text.strip()}",
                    source_turn_id=turn.turn_id,
                    novelty_score=0.38,
                    importance_score=0.52,
                    repeat_score=0.35,
                    trust_score=0.76,
                    future_utility_score=0.58,
                    explicit_user_signal=False,
                    tags=["episodic", perception.primary_intent],
                    rationale="Successful turn recorded as an episodic memory for future recall.",
                )
            )

        return [self._apply_decision(candidate) for candidate in candidates]

    def _make_candidate(
        self,
        *,
        candidate_type: str,
        key: str,
        content: str,
        source_turn_id: str,
        novelty_score: float,
        importance_score: float,
        repeat_score: float,
        trust_score: float,
        future_utility_score: float,
        explicit_user_signal: bool,
        tags: list[str],
        rationale: str,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            candidate_id=f"cand_{uuid4().hex[:12]}",
            candidate_type=candidate_type,
            content=content,
            source_turn_id=source_turn_id,
            key=key,
            novelty_score=novelty_score,
            importance_score=importance_score,
            repeat_score=repeat_score,
            trust_score=trust_score,
            future_utility_score=future_utility_score,
            explicit_user_signal=explicit_user_signal,
            commit_decision="discard",
            rationale=rationale,
            tags=tags,
        )

    def _apply_decision(self, candidate: MemoryCandidate) -> MemoryCandidate:
        commit_score = (
            0.2 * candidate.novelty_score
            + 0.25 * candidate.importance_score
            + 0.15 * candidate.repeat_score
            + 0.2 * candidate.trust_score
            + 0.2 * candidate.future_utility_score
        )
        should_commit = candidate.explicit_user_signal or commit_score >= 0.7
        return candidate.model_copy(
            update={
                "commit_decision": "commit" if should_commit else "discard",
                "rationale": f"{candidate.rationale} commit_score={commit_score:.2f}",
            }
        )

    def _slugify(self, text: str) -> str:
        cleaned = "".join(char.lower() if char.isalnum() else "." for char in text or "general")
        parts = [part for part in cleaned.split(".") if part]
        return ".".join(parts[:4]) or "general"
