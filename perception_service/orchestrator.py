from __future__ import annotations

import logging

from .critic import ResponseCritic
from .dialogue_planner import DialoguePlanner
from .generator import BaseLLMGenerator
from .memory_committer import MemoryCommitter
from .models import (
    AgentRunEvaluation,
    AgentRunRequest,
    AgentRunResponse,
    CriticRequest,
    DialoguePlanRequest,
    GenerationRequest,
    MemoryCommitRequest,
    PromptAssemblyRequest,
    WorkingMemoryUpdateRequest,
)
from .pipeline import PerceptionPipeline
from .prompt_assembler import PromptAssembler
from .working_memory import WorkingMemoryManager

logger = logging.getLogger("orchestrator_service")


class AgentOrchestrator:
    def __init__(
        self,
        perception_pipeline: PerceptionPipeline,
        working_memory_manager: WorkingMemoryManager,
        dialogue_planner: DialoguePlanner,
        prompt_assembler: PromptAssembler,
        generator: BaseLLMGenerator,
        critic: ResponseCritic,
        memory_committer: MemoryCommitter,
    ) -> None:
        self.perception_pipeline = perception_pipeline
        self.working_memory_manager = working_memory_manager
        self.dialogue_planner = dialogue_planner
        self.prompt_assembler = prompt_assembler
        self.generator = generator
        self.critic = critic
        self.memory_committer = memory_committer

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        perception = self.perception_pipeline.analyze(request.turn_input)
        working_memory = self.working_memory_manager.update(
            WorkingMemoryUpdateRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                current_state=request.current_working_memory_state,
            )
        )
        dialogue_plan = self.dialogue_planner.create_plan(
            DialoguePlanRequest(
                turn_input=request.turn_input,
                perception_state=perception,
                working_memory_state=working_memory.state,
            )
        )
        prompt_assembly = self.prompt_assembler.assemble_prompt(
            PromptAssemblyRequest(
                turn_input=request.turn_input,
                perception_state=perception,
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
                    critic_review=critic.review,
                    persist_committed=request.persist_committed_memory,
                )
            )

        response = AgentRunResponse(
            perception=perception,
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
