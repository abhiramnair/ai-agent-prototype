from __future__ import annotations

import logging

from fastapi import FastAPI

from .models import PerceptionState, TurnInput
from .pipeline import PerceptionPipeline


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Perception Service", version="0.1.0")
    pipeline = PerceptionPipeline()

    @app.post("/perception/analyze", response_model=PerceptionState)
    def analyze_perception(turn: TurnInput) -> PerceptionState:
        return pipeline.analyze(turn)

    return app
