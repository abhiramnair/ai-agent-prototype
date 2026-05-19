from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from .memory_retriever import MemoryRetriever
from .models import (
    DialoguePlan,
    MemoryRetrievalRequest,
    PerceptionState,
    PromptAssembly,
    PromptAssemblyEvaluation,
    PromptAssemblyRequest,
    PromptAssemblyResponse,
    RetrievedMemory,
    TurnInput,
    WorkingMemoryState,
)

logger = logging.getLogger("prompt_assembler_service")


class PromptAssemblerHook(ABC):
    @abstractmethod
    def assemble(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
        dialogue_plan: DialoguePlan,
        retrieved_memories: list[RetrievedMemory],
    ) -> PromptAssembly:
        raise NotImplementedError


class DefaultPromptAssemblerHook(PromptAssemblerHook):
    SYSTEM_ROLE = "You are the conversation generator in a modular cognitive system."

    def assemble(
        self,
        turn: TurnInput,
        perception: PerceptionState,
        working_memory: WorkingMemoryState,
        dialogue_plan: DialoguePlan,
        retrieved_memories: list[RetrievedMemory],
    ) -> PromptAssembly:
        relevant_recent_context = self._collect_recent_context(turn, working_memory)
        working_memory_snapshot = self._build_working_memory_snapshot(working_memory)
        perception_summary = self._build_perception_summary(perception)
        response_plan = self._build_response_plan(dialogue_plan)
        instructions = self._build_instructions(perception, dialogue_plan, retrieved_memories)
        rendered_prompt = self._render_prompt(
            turn=turn,
            relevant_recent_context=relevant_recent_context,
            retrieved_memories=retrieved_memories,
            working_memory=working_memory,
            perception_summary=perception_summary,
            response_plan=response_plan,
            instructions=instructions,
        )

        return PromptAssembly(
            system_role=self.SYSTEM_ROLE,
            current_user_message=turn.message_text.strip(),
            active_goal=working_memory.active_goal,
            current_subgoal=working_memory.current_subgoal,
            relevant_recent_context=relevant_recent_context,
            retrieved_memories=retrieved_memories,
            working_memory_snapshot=working_memory_snapshot,
            perception_summary=perception_summary,
            response_plan=response_plan,
            instructions=instructions,
            rendered_prompt=rendered_prompt,
            debug_signals={
                "recent_context_count": len(relevant_recent_context),
                "retrieved_memory_count": len(retrieved_memories),
                "instruction_count": len(instructions),
                "must_include_count": len(dialogue_plan.must_include),
            },
        )

    def _collect_recent_context(
        self,
        turn: TurnInput,
        working_memory: WorkingMemoryState,
    ) -> list[str]:
        items: list[str] = []
        if turn.recent_context:
            items.extend(turn.recent_context.recent_turn_summaries)
            if turn.recent_context.active_topic:
                items.append(f"active_topic: {turn.recent_context.active_topic}")
            if turn.recent_context.last_assistant_action:
                items.append(f"last_assistant_action: {turn.recent_context.last_assistant_action}")
        items.extend(working_memory.recent_turns_compact[-3:])
        return self._dedupe(items)[:6]

    def _build_working_memory_snapshot(self, working_memory: WorkingMemoryState) -> dict[str, object]:
        return {
            "conversation_mode": working_memory.conversation_mode,
            "response_mode": working_memory.response_mode,
            "active_entities": working_memory.active_entities[:6],
            "temporary_assumptions": working_memory.temporary_assumptions[:4],
            "unresolved_questions": working_memory.unresolved_questions[:4],
            "attention_targets": working_memory.attention_targets[:6],
            "suppressed_topics": working_memory.suppressed_topics[:4],
            "emotional_context": working_memory.emotional_context,
        }

    def _build_perception_summary(self, perception: PerceptionState) -> dict[str, object]:
        return {
            "intent": perception.primary_intent,
            "topic": perception.topic,
            "tone": perception.tone,
            "emotion": perception.emotion,
            "urgency": perception.urgency,
            "ambiguity_score": perception.ambiguity_score,
            "references_prior_context": perception.references_prior_context,
            "possible_user_goal": perception.possible_user_goal,
        }

    def _build_response_plan(self, dialogue_plan: DialoguePlan) -> dict[str, object]:
        return {
            "response_mode": dialogue_plan.response_mode,
            "primary_goal": dialogue_plan.primary_goal,
            "secondary_goal": dialogue_plan.secondary_goal,
            "reasoning_style": dialogue_plan.reasoning_style,
            "tone": dialogue_plan.tone,
            "detail_level": dialogue_plan.detail_level,
            "clarification_policy": dialogue_plan.clarification_policy,
            "memory_use_policy": dialogue_plan.memory_use_policy,
            "must_include": dialogue_plan.must_include,
            "must_avoid": dialogue_plan.must_avoid,
            "draft_constraints": dialogue_plan.draft_constraints.model_dump(mode="json"),
        }

    def _build_instructions(
        self,
        perception: PerceptionState,
        dialogue_plan: DialoguePlan,
        retrieved_memories: list[RetrievedMemory],
    ) -> list[str]:
        instructions = [
            "Answer the user directly.",
            "Stay within the active topic and current subgoal.",
            "Follow the response plan and draft constraints.",
        ]
        if dialogue_plan.must_include:
            instructions.append(f"Include: {', '.join(dialogue_plan.must_include)}.")
        if dialogue_plan.must_avoid:
            instructions.append(f"Avoid: {', '.join(dialogue_plan.must_avoid)}.")
        if dialogue_plan.draft_constraints.require_explicit_uncertainty:
            instructions.append("State uncertainty explicitly when assumptions are necessary.")
        if perception.references_prior_context:
            instructions.append("Maintain continuity with recent context.")
        if retrieved_memories:
            instructions.append("Use retrieved long-term memories only when they are relevant and supportive.")
        return instructions

    def _render_prompt(
        self,
        turn: TurnInput,
        relevant_recent_context: list[str],
        retrieved_memories: list[RetrievedMemory],
        working_memory: WorkingMemoryState,
        perception_summary: dict[str, object],
        response_plan: dict[str, object],
        instructions: list[str],
    ) -> str:
        sections = [
            f"SYSTEM ROLE\n{self.SYSTEM_ROLE}",
            f"CURRENT USER MESSAGE\n{turn.message_text.strip()}",
            f"ACTIVE GOAL\n{working_memory.active_goal}",
            f"CURRENT SUBGOAL\n{working_memory.current_subgoal}",
            "RELEVANT RECENT CONTEXT\n" + self._render_list(relevant_recent_context),
            "RETRIEVED LONG-TERM MEMORIES\n" + self._render_memories(retrieved_memories),
            "WORKING MEMORY SNAPSHOT\n" + self._render_dict(
                {
                    "conversation_mode": working_memory.conversation_mode,
                    "response_mode": working_memory.response_mode,
                    "attention_targets": working_memory.attention_targets,
                    "unresolved_questions": working_memory.unresolved_questions,
                    "emotional_context": working_memory.emotional_context,
                }
            ),
            "PERCEPTION SUMMARY\n" + self._render_dict(perception_summary),
            "RESPONSE PLAN\n" + self._render_dict(response_plan),
            "INSTRUCTIONS\n" + self._render_list(instructions),
        ]
        return "\n\n".join(sections)

    def _render_memories(self, memories: list[RetrievedMemory]) -> str:
        if not memories:
            return "- none"
        lines: list[str] = []
        for memory in memories:
            lines.append(
                f"- [{memory.memory_type}] {memory.key}: {memory.value} "
                f"(score={memory.retrieval_score}, reason={memory.retrieval_reason})"
            )
        return "\n".join(lines)

    def _render_list(self, items: list[object]) -> str:
        if not items:
            return "- none"
        return "\n".join(f"- {item}" for item in items)

    def _render_dict(self, data: dict[str, object]) -> str:
        if not data:
            return "- none"
        return "\n".join(f"- {key}: {value}" for key, value in data.items())

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


class PromptAssembler:
    def __init__(
        self,
        hook: PromptAssemblerHook | None = None,
        memory_retriever: MemoryRetriever | None = None,
    ) -> None:
        self.hook = hook or DefaultPromptAssemblerHook()
        self.memory_retriever = memory_retriever

    def assemble_prompt(self, request: PromptAssemblyRequest) -> PromptAssemblyResponse:
        retrieved_memories = self._retrieve_memories(request) if self.memory_retriever else []
        prompt = self.hook.assemble(
            request.turn_input,
            request.perception_state,
            request.working_memory_state,
            request.dialogue_plan,
            retrieved_memories,
        )
        evaluation = PromptAssemblyEvaluation(
            includes_recent_context=bool(prompt.relevant_recent_context),
            includes_retrieved_memories=bool(prompt.retrieved_memories),
            includes_constraints=bool(prompt.response_plan.get("draft_constraints")),
            includes_uncertainty_guidance=any(
                "uncertainty" in instruction.lower() for instruction in prompt.instructions
            ),
            rendered_prompt_nonempty=bool(prompt.rendered_prompt.strip()),
        )
        logger.info(
            "prompt_assembled",
            extra={
                "turn_id": request.turn_input.turn_id,
                "prompt": prompt.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return PromptAssemblyResponse(prompt=prompt, evaluation=evaluation)

    def _retrieve_memories(self, request: PromptAssemblyRequest) -> list[RetrievedMemory]:
        query_parts = [
            request.turn_input.message_text.strip(),
            request.perception_state.topic,
            request.perception_state.possible_user_goal,
        ]
        query_text = " ".join(part for part in query_parts if part).strip()
        tags: list[str] = []
        if request.perception_state.salience_signals.contains_preference:
            tags.append("preference")
        if request.perception_state.salience_signals.contains_correction:
            tags.append("correction")
        retrieval = self.memory_retriever.retrieve(
            MemoryRetrievalRequest(
                query_text=query_text,
                memory_types=[],
                tags=tags,
                limit=4,
            )
        )
        if retrieval.memories:
            return retrieval.memories

        fallback = self.memory_retriever.retrieve(
            MemoryRetrievalRequest(
                query_text="",
                memory_types=["preference", "procedural", "correction"],
                tags=tags,
                limit=3,
            )
        )
        return fallback.memories
