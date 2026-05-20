from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from .models import (
    AttentionGateEvaluation,
    AttentionGateRequest,
    AttentionGateResponse,
    AttentionGateState,
    PerceptionState,
    TurnInput,
    WorkingMemoryState,
)

logger = logging.getLogger("attention_gate_service")


class AttentionGateHook(ABC):
    @abstractmethod
    def evaluate(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> AttentionGateState:
        raise NotImplementedError


class DefaultAttentionGateHook(AttentionGateHook):
    def evaluate(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> AttentionGateState:
        focus_targets = self._derive_focus_targets(turn, perception, current_state)
        suppressed_topics = self._derive_suppressed_topics(perception, current_state)
        salience_score = self._derive_salience_score(perception)
        attention_budget = self._derive_attention_budget(perception, salience_score)
        requires_recent_context = perception.references_prior_context or bool(turn.recent_context and turn.recent_context.unresolved_questions)
        memory_retrieval_mode = self._derive_memory_retrieval_mode(perception, requires_recent_context, attention_budget)
        primary_focus = focus_targets[0] if focus_targets else perception.topic

        return AttentionGateState(
            primary_focus=primary_focus or "general_conversation",
            focus_targets=focus_targets[:6],
            suppressed_topics=suppressed_topics[:4],
            salience_score=round(salience_score, 2),
            attention_budget=round(attention_budget, 2),
            requires_recent_context=requires_recent_context,
            memory_retrieval_mode=memory_retrieval_mode,
            debug_signals={
                "source_intent": perception.primary_intent,
                "source_topic": perception.topic,
                "references_prior_context": perception.references_prior_context,
            },
        )

    def _derive_focus_targets(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> list[str]:
        targets: list[str] = []
        if perception.topic:
            targets.append(perception.topic)
        targets.extend(entity.text for entity in perception.entities[:4])
        if perception.references_prior_context:
            targets.append("recent_context")
        if perception.salience_signals.contains_preference:
            targets.append("preference_signal")
        if perception.salience_signals.contains_correction:
            targets.append("correction_signal")
        if current_state:
            targets.extend(current_state.attention_targets[:2])
        if turn.recent_context and turn.recent_context.active_topic:
            targets.append(turn.recent_context.active_topic)
        return self._dedupe(targets)

    def _derive_suppressed_topics(
        self,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> list[str]:
        suppressed = list(current_state.suppressed_topics) if current_state else []
        if perception.salience_signals.contains_scope_shift:
            suppressed.append("previous_scope")
        return self._dedupe(suppressed)

    def _derive_salience_score(self, perception: PerceptionState) -> float:
        score = 0.32
        if perception.salience_signals.contains_correction:
            score += 0.2
        if perception.salience_signals.contains_preference:
            score += 0.16
        if perception.salience_signals.contains_commitment:
            score += 0.12
        if perception.references_prior_context:
            score += 0.08
        if perception.urgency == "high":
            score += 0.12
        if perception.ambiguity_score >= 0.45:
            score += 0.08
        return min(1.0, score)

    def _derive_attention_budget(self, perception: PerceptionState, salience_score: float) -> float:
        budget = 0.42 + (0.4 * salience_score)
        if perception.primary_intent in {"ask_explanation", "request_action", "brainstorming"}:
            budget += 0.08
        if perception.primary_intent == "social_message":
            budget -= 0.08
        return min(1.0, max(0.0, budget))

    def _derive_memory_retrieval_mode(
        self,
        perception: PerceptionState,
        requires_recent_context: bool,
        attention_budget: float,
    ) -> str:
        if perception.salience_signals.contains_correction or perception.salience_signals.contains_preference:
            return "focused_retrieval"
        if requires_recent_context:
            return "recent_context_priority"
        if attention_budget >= 0.7:
            return "broad_retrieval"
        return "minimal_retrieval"

    def _dedupe(self, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(normalized)
        return output


class AttentionGate:
    def __init__(self, hook: AttentionGateHook | None = None) -> None:
        self.hook = hook or DefaultAttentionGateHook()

    def evaluate(self, request: AttentionGateRequest) -> AttentionGateResponse:
        state = self.hook.evaluate(
            request.turn_input,
            request.perception_state,
            request.current_state,
        )
        evaluation = AttentionGateEvaluation(
            focus_target_count=len(state.focus_targets),
            suppressed_topic_count=len(state.suppressed_topics),
            references_recent_context=state.requires_recent_context,
            attention_budget_high=state.attention_budget >= 0.7,
        )
        logger.info(
            "attention_gate_evaluated",
            extra={
                "turn_id": request.turn_input.turn_id,
                "state": state.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return AttentionGateResponse(state=state, evaluation=evaluation)
