from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Any

from .models import (
    PerceptionState,
    TurnInput,
    WorkingMemoryEvaluation,
    WorkingMemoryState,
    WorkingMemoryUpdateRequest,
    WorkingMemoryUpdateResponse,
)

logger = logging.getLogger("working_memory_service")


class WorkingMemoryHook(ABC):
    @abstractmethod
    def update(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> WorkingMemoryState:
        raise NotImplementedError


class DefaultWorkingMemoryHook(WorkingMemoryHook):
    def update(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        current_state: WorkingMemoryState | None,
    ) -> WorkingMemoryState:
        previous = current_state or self._initial_state(turn, perception)
        active_goal = self._derive_active_goal(previous, perception)
        current_subgoal = self._derive_current_subgoal(perception)
        response_mode = self._derive_response_mode(perception)
        conversation_mode = self._derive_conversation_mode(turn, previous)
        active_entities = self._merge_entities(previous, turn, perception)
        temporary_assumptions = self._merge_temporary_assumptions(previous, perception)
        unresolved_questions = self._merge_unresolved_questions(previous, turn, perception)
        recent_turns_compact = self._update_recent_turns(previous, turn, perception)
        emotional_context = self._derive_emotional_context(perception)
        attention_targets = self._derive_attention_targets(perception, active_entities)
        suppressed_topics = self._derive_suppressed_topics(previous, perception)

        return WorkingMemoryState(
            active_goal=active_goal,
            current_subgoal=current_subgoal,
            conversation_mode=conversation_mode,
            response_mode=response_mode,
            active_entities=active_entities,
            temporary_assumptions=temporary_assumptions,
            unresolved_questions=unresolved_questions,
            recent_turns_compact=recent_turns_compact,
            emotional_context=emotional_context,
            attention_targets=attention_targets,
            suppressed_topics=suppressed_topics,
            debug_signals={
                "source_intent": perception.primary_intent,
                "source_topic": perception.topic,
                "source_goal": perception.possible_user_goal,
            },
        )

    def _initial_state(self, turn: TurnInput, perception: PerceptionState) -> WorkingMemoryState:
        return WorkingMemoryState(
            active_goal=perception.possible_user_goal,
            current_subgoal=f"handle the current turn about {perception.topic}",
            conversation_mode=turn.recent_context.conversation_mode if turn.recent_context else "general",
            response_mode="direct_response",
            active_entities=[],
            temporary_assumptions=[],
            unresolved_questions=[],
            recent_turns_compact=[],
            emotional_context="stable",
            attention_targets=[],
            suppressed_topics=[],
        )

    def _derive_active_goal(self, previous: WorkingMemoryState, perception: PerceptionState) -> str:
        if perception.salience_signals.contains_scope_shift or perception.salience_signals.contains_preference:
            return perception.possible_user_goal
        if perception.primary_intent in {"request_action", "ask_explanation", "brainstorming"}:
            return perception.possible_user_goal
        return previous.active_goal

    def _derive_current_subgoal(self, perception: PerceptionState) -> str:
        if perception.primary_intent == "ask_explanation":
            return f"clarify or expand on {perception.topic}"
        if perception.primary_intent == "request_action":
            return f"prepare the next action for {perception.topic}"
        if perception.primary_intent == "correction":
            return f"repair understanding related to {perception.topic}"
        if perception.primary_intent == "clarification":
            return f"resolve ambiguity around {perception.topic}"
        return f"continue the conversation around {perception.topic}"

    def _derive_response_mode(self, perception: PerceptionState) -> str:
        if perception.primary_intent == "ask_explanation":
            return "explain_clearly"
        if perception.primary_intent == "request_action":
            return "action_oriented"
        if perception.primary_intent == "brainstorming":
            return "explore_options"
        if perception.primary_intent == "correction":
            return "repair_and_align"
        if perception.primary_intent == "social_message":
            return "social_reply"
        return "direct_response"

    def _derive_conversation_mode(self, turn: TurnInput, previous: WorkingMemoryState) -> str:
        if turn.recent_context and turn.recent_context.conversation_mode:
            return turn.recent_context.conversation_mode
        return previous.conversation_mode

    def _merge_entities(
        self,
        previous: WorkingMemoryState,
        turn: TurnInput,
        perception: PerceptionState,
    ) -> list[str]:
        entities = [entity.text for entity in perception.entities]
        if perception.topic and perception.topic != "general_conversation":
            entities.extend(self._split_topic_entities(perception.topic))
        if turn.recent_context and turn.recent_context.active_topic:
            entities.append(turn.recent_context.active_topic)
        merged = previous.active_entities + entities
        return self._dedupe_preserve_order(merged)[:12]

    def _merge_temporary_assumptions(
        self,
        previous: WorkingMemoryState,
        perception: PerceptionState,
    ) -> list[str]:
        assumptions = list(previous.temporary_assumptions)
        if perception.references_prior_context:
            assumptions.append("The user expects continuity with recent context.")
        if perception.salience_signals.contains_preference:
            assumptions.append("The user is shaping system behavior or direction.")
        if perception.ambiguity_score >= 0.45:
            assumptions.append("The current turn may require extra care around ambiguity.")
        return self._dedupe_preserve_order(assumptions)[-6:]

    def _merge_unresolved_questions(
        self,
        previous: WorkingMemoryState,
        turn: TurnInput,
        perception: PerceptionState,
    ) -> list[str]:
        unresolved = list(previous.unresolved_questions)
        if turn.recent_context:
            unresolved.extend(turn.recent_context.unresolved_questions)
        if perception.primary_intent in {"ask_question", "clarification"}:
            unresolved.append(turn.message_text.strip())
        if perception.ambiguity_score >= 0.6 and turn.message_text.strip() not in unresolved:
            unresolved.append(f"Resolve ambiguity in: {turn.message_text.strip()}")
        return self._dedupe_preserve_order(unresolved)[-8:]

    def _update_recent_turns(
        self,
        previous: WorkingMemoryState,
        turn: TurnInput,
        perception: PerceptionState,
    ) -> list[str]:
        summary = f"{perception.primary_intent}: {turn.message_text.strip()}"
        return (previous.recent_turns_compact + [summary])[-6:]

    def _derive_emotional_context(self, perception: PerceptionState) -> str:
        if perception.emotion in {"anxiety", "disappointment"}:
            return "sensitive"
        if perception.emotion in {"curiosity", "confusion"}:
            return "engaged"
        if perception.tone == "frustrated":
            return "tense"
        return "stable"

    def _derive_attention_targets(
        self,
        perception: PerceptionState,
        active_entities: list[str],
    ) -> list[str]:
        targets = [perception.topic]
        if perception.references_prior_context:
            targets.append("recent_context")
        targets.extend(active_entities[:3])
        return self._dedupe_preserve_order([target for target in targets if target])[:6]

    def _derive_suppressed_topics(
        self,
        previous: WorkingMemoryState,
        perception: PerceptionState,
    ) -> list[str]:
        suppressed = list(previous.suppressed_topics)
        if perception.salience_signals.contains_scope_shift and perception.topic:
            suppressed.append("previous_scope")
        return self._dedupe_preserve_order(suppressed)[-4:]

    def _split_topic_entities(self, topic: str) -> list[str]:
        return [part for part in re.split(r"[\s/_-]+", topic) if part]

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
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


class WorkingMemoryManager:
    def __init__(self, hook: WorkingMemoryHook | None = None) -> None:
        self.hook = hook or DefaultWorkingMemoryHook()

    def update(self, request: WorkingMemoryUpdateRequest) -> WorkingMemoryUpdateResponse:
        next_state = self.hook.update(
            request.turn_input,
            request.perception_state,
            request.current_state,
        )
        previous = request.current_state
        evaluation = WorkingMemoryEvaluation(
            active_goal_changed=previous.active_goal != next_state.active_goal if previous else True,
            subgoal_changed=previous.current_subgoal != next_state.current_subgoal if previous else True,
            references_carried_forward=request.perception_state.references_prior_context,
            unresolved_questions_count=len(next_state.unresolved_questions),
            active_entities_count=len(next_state.active_entities),
        )
        logger.info(
            "working_memory_updated",
            extra={
                "turn_id": request.turn_input.turn_id,
                "state": next_state.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return WorkingMemoryUpdateResponse(state=next_state, evaluation=evaluation)
