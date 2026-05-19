from __future__ import annotations

import logging

from fastapi import FastAPI

from .models import (
    DialoguePlanRequest,
    DialoguePlanResponse,
    PerceptionState,
    PromptAssemblyRequest,
    PromptAssemblyResponse,
    TurnInput,
    WorkingMemoryUpdateRequest,
    WorkingMemoryUpdateResponse,
)
from .dialogue_planner import DialoguePlanner
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

    return app
