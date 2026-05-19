from __future__ import annotations

import logging

from fastapi import FastAPI

from .models import (
    CriticRequest,
    CriticResponse,
    DialoguePlanRequest,
    DialoguePlanResponse,
    GenerationRequest,
    GenerationResponse,
    MemoryArchiveRequest,
    MemoryArchiveResponse,
    MemoryMutationResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
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
from .memory_store import MemoryStore
from .pipeline import PerceptionPipeline
from .prompt_assembler import PromptAssembler
from .working_memory import WorkingMemoryManager


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Perception Service", version="0.1.0")
    pipeline = PerceptionPipeline()
    working_memory = WorkingMemoryManager()
    dialogue_planner = DialoguePlanner()
    prompt_assembler = PromptAssembler()
    generator = BaseLLMGenerator()
    critic = ResponseCritic()
    memory_store = MemoryStore()

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

    return app
