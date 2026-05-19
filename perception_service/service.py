from __future__ import annotations

import logging

from fastapi import FastAPI

from .models import (
    AgentRunRequest,
    AgentRunResponse,
    CriticRequest,
    CriticResponse,
    DialoguePlanRequest,
    DialoguePlanResponse,
    GenerationRequest,
    GenerationResponse,
    MemoryCommitRequest,
    MemoryCommitResponse,
    MemoryArchiveRequest,
    MemoryArchiveResponse,
    MemoryDecayRequest,
    MemoryDecayResponse,
    MemoryMutationResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    MemoryUpsertRequest,
    PerceptionState,
    PromptAssemblyRequest,
    PromptAssemblyResponse,
    TurnInput,
    WorkingMemoryUpdateRequest,
    WorkingMemoryUpdateResponse,
)
from .critic import ResponseCritic
from .dialogue_planner import DialoguePlanner
from .generator import BaseLLMGenerator
from .memory_committer import MemoryCommitter
from .memory_retriever import MemoryRetriever
from .memory_store import MemoryStore
from .orchestrator import AgentOrchestrator
from .pipeline import PerceptionPipeline
from .prompt_assembler import PromptAssembler
from .working_memory import WorkingMemoryManager


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Perception Service", version="0.1.0")
    pipeline = PerceptionPipeline()
    working_memory = WorkingMemoryManager()
    dialogue_planner = DialoguePlanner()
    generator = BaseLLMGenerator()
    critic = ResponseCritic()
    memory_store = MemoryStore()
    memory_retriever = MemoryRetriever(memory_store)
    memory_committer = MemoryCommitter(memory_store)
    prompt_assembler = PromptAssembler(memory_retriever=memory_retriever)
    orchestrator = AgentOrchestrator(
        perception_pipeline=pipeline,
        working_memory_manager=working_memory,
        dialogue_planner=dialogue_planner,
        prompt_assembler=prompt_assembler,
        generator=generator,
        critic=critic,
        memory_committer=memory_committer,
    )

    @app.post("/perception/analyze", response_model=PerceptionState)
    def analyze_perception(turn: TurnInput) -> PerceptionState:
        return pipeline.analyze(turn)

    @app.post("/working-memory/update", response_model=WorkingMemoryUpdateResponse)
    def update_working_memory(request: WorkingMemoryUpdateRequest) -> WorkingMemoryUpdateResponse:
        return working_memory.update(request)

    @app.post("/dialogue-planner/plan", response_model=DialoguePlanResponse)
    def create_dialogue_plan(request: DialoguePlanRequest) -> DialoguePlanResponse:
        return dialogue_planner.create_plan(request)

    @app.post("/prompt-assembler/assemble", response_model=PromptAssemblyResponse)
    def assemble_prompt(request: PromptAssemblyRequest) -> PromptAssemblyResponse:
        return prompt_assembler.assemble_prompt(request)

    @app.post("/generator/generate", response_model=GenerationResponse)
    def generate_response(request: GenerationRequest) -> GenerationResponse:
        return generator.generate(request)

    @app.post("/critic/review", response_model=CriticResponse)
    def review_response(request: CriticRequest) -> CriticResponse:
        return critic.review(request)

    @app.post("/memory/upsert", response_model=MemoryMutationResponse)
    def upsert_memory(request: MemoryUpsertRequest) -> MemoryMutationResponse:
        return memory_store.upsert(request)

    @app.post("/memory/query", response_model=MemoryQueryResponse)
    def query_memory(request: MemoryQueryRequest) -> MemoryQueryResponse:
        return memory_store.query(request)

    @app.post("/memory/archive", response_model=MemoryArchiveResponse)
    def archive_memory(request: MemoryArchiveRequest) -> MemoryArchiveResponse:
        return memory_store.archive(request)

    @app.post("/memory/decay", response_model=MemoryDecayResponse)
    def decay_memory(request: MemoryDecayRequest) -> MemoryDecayResponse:
        return memory_store.decay(request)

    @app.post("/memory/retrieve", response_model=MemoryRetrievalResponse)
    def retrieve_memory(request: MemoryRetrievalRequest) -> MemoryRetrievalResponse:
        return memory_retriever.retrieve(request)

    @app.post("/memory/commit", response_model=MemoryCommitResponse)
    def commit_memory(request: MemoryCommitRequest) -> MemoryCommitResponse:
        return memory_committer.commit(request)

    @app.post("/agent/run", response_model=AgentRunResponse)
    def run_agent(request: AgentRunRequest) -> AgentRunResponse:
        return orchestrator.run(request)

    return app
