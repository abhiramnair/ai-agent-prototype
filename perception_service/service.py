from __future__ import annotations

import json
import logging
from urllib import error, request as urlrequest

from fastapi import FastAPI

from .models import (
    AgentRunRequest,
    AgentRunResponse,
    ConfigResponse,
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
    HealthResponse,
    PerceptionState,
    PromptAssemblyRequest,
    PromptAssemblyResponse,
    SessionStateResponse,
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
from .session_state import SessionStateStore
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
    session_state_store = SessionStateStore()
    prompt_assembler = PromptAssembler(memory_retriever=memory_retriever)
    orchestrator = AgentOrchestrator(
        perception_pipeline=pipeline,
        working_memory_manager=working_memory,
        dialogue_planner=dialogue_planner,
        prompt_assembler=prompt_assembler,
        generator=generator,
        critic=critic,
        memory_committer=memory_committer,
        session_state_store=session_state_store,
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

    @app.get("/session/{session_id}", response_model=SessionStateResponse)
    def get_session_state(session_id: str) -> SessionStateResponse:
        return session_state_store.get(session_id)

    @app.delete("/session/{session_id}", response_model=SessionStateResponse)
    def clear_session_state(session_id: str) -> SessionStateResponse:
        return session_state_store.clear(session_id)

    @app.get("/config", response_model=ConfigResponse)
    def get_config() -> ConfigResponse:
        provider_name = str(getattr(generator.provider, "provider_name", generator.provider.__class__.__name__))
        model_name = str(getattr(generator.provider, "model_name", "unknown"))
        base_url = str(getattr(generator.provider, "base_url", ""))
        return ConfigResponse(
            llm_provider=provider_name,
            llm_model=model_name,
            ollama_base_url=base_url,
            test_provider_override=provider_name == "mock",
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        provider_name = str(getattr(generator.provider, "provider_name", generator.provider.__class__.__name__))
        model_name = str(getattr(generator.provider, "model_name", "unknown"))
        memory_records_available = memory_store.query(
            MemoryQueryRequest(limit=1, include_archived=True)
        ).evaluation.total_matches >= 0
        ollama_reachable = provider_name != "ollama"

        if provider_name == "ollama":
            base_url = str(getattr(generator.provider, "base_url", "")).rstrip("/")
            try:
                req = urlrequest.Request(f"{base_url}/api/tags", method="GET")
                with urlrequest.urlopen(req, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                ollama_reachable = "models" in payload
            except (error.URLError, TimeoutError, json.JSONDecodeError):
                ollama_reachable = False

        return HealthResponse(
            status="ok" if ollama_reachable else "degraded",
            llm_provider=provider_name,
            llm_model=model_name,
            ollama_reachable=ollama_reachable,
            memory_records_available=memory_records_available,
        )

    return app
