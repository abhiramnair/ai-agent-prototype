from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from .models import (
    DialoguePlan,
    DialoguePlanEvaluation,
    DialoguePlanRequest,
    DialoguePlanResponse,
    DraftConstraints,
    PerceptionState,
    TurnInput,
    WorkingMemoryState,
)

logger = logging.getLogger("dialogue_planner_service")


class DialoguePlannerHook(ABC):
    @abstractmethod
    def create_plan(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
    ) -> DialoguePlan:
        raise NotImplementedError


class DefaultDialoguePlannerHook(DialoguePlannerHook):
    def create_plan(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
    ) -> DialoguePlan:
        response_mode = self._derive_response_mode(perception, working_memory)
        reasoning_style = self._derive_reasoning_style(perception, working_memory)
        tone = self._derive_tone(perception, working_memory)
        detail_level = self._derive_detail_level(perception)
        clarification_policy = self._derive_clarification_policy(perception)
        memory_use_policy = self._derive_memory_use_policy(perception)
        must_include = self._derive_must_include(perception, working_memory)
        must_avoid = self._derive_must_avoid(perception, working_memory)
        constraints = self._derive_constraints(perception, response_mode)

        return DialoguePlan(
            response_mode=response_mode,
            primary_goal=working_memory.active_goal,
            secondary_goal=working_memory.current_subgoal,
            reasoning_style=reasoning_style,
            tone=tone,
            detail_level=detail_level,
            clarification_policy=clarification_policy,
            memory_use_policy=memory_use_policy,
            must_include=must_include,
            must_avoid=must_avoid,
            draft_constraints=constraints,
            debug_signals={
                "source_intent": perception.primary_intent,
                "source_response_mode": working_memory.response_mode,
                "active_entities_count": len(working_memory.active_entities),
            },
        )

    def _derive_response_mode(self, perception: PerceptionState, working_memory: WorkingMemoryState) -> str:
        if perception.ambiguity_score >= 0.65:
            return "clarify_before_answering"
        if perception.primary_intent == "ask_explanation":
            return "structured_explanation"
        if perception.primary_intent == "request_action":
            return "execution_planning"
        if perception.primary_intent == "brainstorming":
            return "option_generation"
        if perception.primary_intent == "correction":
            return "alignment_repair"
        if perception.primary_intent == "social_message":
            return "social_reply"
        return working_memory.response_mode or "direct_response"

    def _derive_reasoning_style(self, perception: PerceptionState, working_memory: WorkingMemoryState) -> str:
        if perception.primary_intent in {"ask_explanation", "clarification"}:
            return "structured"
        if perception.primary_intent == "brainstorming":
            return "divergent"
        if "technical" in working_memory.conversation_mode.lower():
            return "stepwise"
        return "direct"

    def _derive_tone(self, perception: PerceptionState, working_memory: WorkingMemoryState) -> str:
        if working_memory.emotional_context == "tense":
            return "calm_supportive"
        if perception.tone == "collaborative":
            return "collaborative_technical"
        if perception.tone == "friendly":
            return "warm_clear"
        return "clear_direct"

    def _derive_detail_level(self, perception: PerceptionState) -> str:
        if perception.primary_intent in {"ask_explanation", "brainstorming"}:
            return "high"
        if perception.primary_intent == "social_message":
            return "low"
        return "medium"

    def _derive_clarification_policy(self, perception: PerceptionState) -> str:
        if perception.ambiguity_score >= 0.65:
            return "ask_one_targeted_question"
        if perception.ambiguity_score >= 0.45:
            return "answer_with_assumption_if_safe"
        return "do_not_ask_unless_blocked"

    def _derive_memory_use_policy(self, perception: PerceptionState) -> str:
        if perception.references_prior_context:
            return "prefer_recent_context"
        return "retrieve_minimal_relevant_context"

    def _derive_must_include(
        self,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
    ) -> list[str]:
        items = [working_memory.current_subgoal]
        if perception.references_prior_context:
            items.append("acknowledge recent context")
        if working_memory.unresolved_questions:
            items.append("address unresolved questions when relevant")
        if perception.topic:
            items.append(perception.topic)
        return self._dedupe(items)[:6]

    def _derive_must_avoid(
        self,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
    ) -> list[str]:
        items = ["unsupported claims", "scope drift"]
        if perception.ambiguity_score >= 0.45:
            items.append("overconfident assumptions")
        if working_memory.suppressed_topics:
            items.extend(working_memory.suppressed_topics)
        return self._dedupe(items)[:6]

    def _derive_constraints(self, perception: PerceptionState, response_mode: str) -> DraftConstraints:
        return DraftConstraints(
            avoid_repetition=True,
            avoid_overclaiming=True,
            avoid_scope_drift=True,
            prefer_examples=perception.primary_intent in {"ask_explanation", "brainstorming"},
            require_explicit_uncertainty=response_mode == "clarify_before_answering" or perception.ambiguity_score >= 0.45,
        )

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


class DialoguePlanner:
    def __init__(self, hook: DialoguePlannerHook | None = None) -> None:
        self.hook = hook or DefaultDialoguePlannerHook()

    def create_plan(self, request: DialoguePlanRequest) -> DialoguePlanResponse:
        plan = self.hook.create_plan(
            request.turn_input,
            request.perception_state,
            request.working_memory_state,
        )
        evaluation = DialoguePlanEvaluation(
            used_recent_context=request.perception_state.references_prior_context,
            asks_for_clarification=plan.clarification_policy == "ask_one_targeted_question",
            ambiguity_sensitive=request.perception_state.ambiguity_score >= 0.45,
            plan_has_required_fields=all(
                [
                    plan.response_mode,
                    plan.primary_goal,
                    plan.secondary_goal,
                    plan.reasoning_style,
                    plan.tone,
                    plan.detail_level,
                    plan.clarification_policy,
                    plan.memory_use_policy,
                ]
            ),
        )
        logger.info(
            "dialogue_plan_created",
            extra={
                "turn_id": request.turn_input.turn_id,
                "plan": plan.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return DialoguePlanResponse(plan=plan, evaluation=evaluation)
