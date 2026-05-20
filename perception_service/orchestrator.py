from __future__ import annotations

import logging

from .attention_gate import AttentionGate
from .critic import ResponseCritic
from .dialogue_planner import DialoguePlanner
from .generator import BaseLLMGenerator
from .memory_committer import MemoryCommitter
from .models import (
    AgentRunEvaluation,
    AgentRunRequest,
    AgentRunResponse,
    AttentionGateRequest,
    CriticRequest,
    DialoguePlanRequest,
    GenerationRequest,
    MemoryCommitRequest,
    PromptAssemblyRequest,
    WorkingMemoryUpdateRequest,
)
from .pipeline import PerceptionPipeline
from .prompt_assembler import PromptAssembler
from .session_state import SessionStateStore
from .working_memory import WorkingMemoryManager

logger = logging.getLogger("orchestrator_service")


class AgentOrchestrator:
    def __init__(
        self,
        perception_pipeline: PerceptionPipeline,
        attention_gate: AttentionGate,
        working_memory_manager: WorkingMemoryManager,
        dialogue_planner: DialoguePlanner,
        prompt_assembler: PromptAssembler,
        generator: BaseLLMGenerator,
        critic: ResponseCritic,
        memory_committer: MemoryCommitter,
        session_state_store: SessionStateStore | None = None,
    ) -> None:
        self.perception_pipeline = perception_pipeline
        self.attention_gate = attention_gate
        self.working_memory_manager = working_memory_manager
        self.dialogue_planner = dialogue_planner
        self.prompt_assembler = prompt_assembler
        self.generator = generator
        self.critic = critic
        self.memory_committer = memory_committer
        self.session_state_store = session_state_store

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        prior_state = request.current_working_memory_state
        if prior_state is None and request.use_session_state and self.session_state_store is not None:
            session_lookup = self.session_state_store.get(request.turn_input.session_id)
            prior_state = session_lookup.session.working_memory_state if session_lookup.found and session_lookup.session else None

        perception = self.perception_pipeline.analyze(request.turn_input)
        attention_gate = self.attention_gate.evaluate(
            AttentionGateRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                current_state=prior_state,
            )
        )
        working_memory = self.working_memory_manager.update(
            WorkingMemoryUpdateRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                attention_state=attention_gate.state,
                current_state=prior_state,
            )
        )
        dialogue_plan = self.dialogue_planner.create_plan(
            DialoguePlanRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                attention_state=attention_gate.state,
                working_memory_state=working_memory.state,
            )
        )
        prompt_assembly = self.prompt_assembler.assemble_prompt(
            PromptAssemblyRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                attention_state=attention_gate.state,
                working_memory_state=working_memory.state,
                dialogue_plan=dialogue_plan.plan,
            )
        )
        generation = self.generator.generate(
            GenerationRequest(prompt=prompt_assembly.prompt)
        )
        critic = self.critic.review(
            CriticRequest(
                prompt=prompt_assembly.prompt,
                generation_output=generation.output,
            )
        )

        memory_commit = None
        if request.commit_memory:
            memory_commit = self.memory_committer.commit(
                MemoryCommitRequest(
                    turn_input=request.turn_input,
                    perception_state=perception,
                    working_memory_state=working_memory.state,
                    dialogue_plan=dialogue_plan.plan,
                    critic_review=critic.review,
                    persist_committed=request.persist_committed_memory,
                )
            )

        if request.use_session_state and self.session_state_store is not None:
            self.session_state_store.set(request.turn_input.session_id, working_memory.state)

        response = AgentRunResponse(
            perception=perception,
            attention_gate=attention_gate,
            working_memory=working_memory,
            dialogue_plan=dialogue_plan,
            prompt_assembly=prompt_assembly,
            generation=generation,
            critic=critic,
            memory_commit=memory_commit,
            evaluation=AgentRunEvaluation(
                used_memory_commit=request.commit_memory,
                critic_requires_revision=critic.evaluation.requires_revision,
                used_retrieved_memories=bool(prompt_assembly.prompt.retrieved_memories),
                provider_name=generation.output.metadata.provider_name,
                used_session_state=request.use_session_state,
            ),
        )
        logger.info(
            "agent_orchestrator_completed",
            extra={
                "turn_id": request.turn_input.turn_id,
                "provider_name": response.evaluation.provider_name,
                "critic_requires_revision": response.evaluation.critic_requires_revision,
            },
        )
        return response
