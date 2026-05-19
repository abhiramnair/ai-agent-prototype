from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time

from .models import (
    GenerationEvaluation,
    GenerationMetadata,
    GenerationRequest,
    GenerationResponse,
    GeneratorOutput,
    PromptAssembly,
)

logger = logging.getLogger("generator_service")


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: PromptAssembly) -> GeneratorOutput:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    provider_name = "mock"
    model_name = "mock-conversation-generator-v1"

    def generate(self, prompt: PromptAssembly) -> GeneratorOutput:
        started = time.perf_counter()
        response_text = self._compose_response(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return GeneratorOutput(
            response_text=response_text,
            response_mode=str(prompt.response_plan.get("response_mode", "direct_response")),
            metadata=GenerationMetadata(
                provider_name=self.provider_name,
                model_name=self.model_name,
                finish_reason="completed",
                latency_ms=round(latency_ms, 2),
                token_usage={
                    "prompt_characters": len(prompt.rendered_prompt),
                    "response_characters": len(response_text),
                },
            ),
            debug_signals={
                "used_mock_provider": True,
                "instruction_count": len(prompt.instructions),
            },
        )

    def _compose_response(self, prompt: PromptAssembly) -> str:
        response_mode = str(prompt.response_plan.get("response_mode", "direct_response"))
        detail_level = str(prompt.response_plan.get("detail_level", "medium"))
        must_include = prompt.response_plan.get("must_include", [])
        uncertainty_required = bool(
            prompt.response_plan.get("draft_constraints", {}).get("require_explicit_uncertainty", False)
        )

        opening = self._opening_for_mode(response_mode)
        goal_line = f"I'm focusing on {prompt.current_subgoal}."
        detail_line = f"I'll keep the answer at a {detail_level} level of detail."
        include_line = ""
        if must_include:
            include_line = "I'll make sure to cover " + ", ".join(str(item) for item in must_include[:3]) + "."
        uncertainty_line = ""
        if uncertainty_required:
            uncertainty_line = "Some parts of this turn are ambiguous, so I'll call out assumptions explicitly."

        return " ".join(
            part
            for part in [opening, goal_line, detail_line, include_line, uncertainty_line]
            if part
        ).strip()

    def _opening_for_mode(self, response_mode: str) -> str:
        if response_mode == "structured_explanation":
            return "Here's a structured explanation."
        if response_mode == "execution_planning":
            return "Here's the next-step plan."
        if response_mode == "option_generation":
            return "Here are the strongest options."
        if response_mode == "alignment_repair":
            return "Let me realign the answer."
        if response_mode == "clarify_before_answering":
            return "Before locking an answer, I need to handle the ambiguity carefully."
        return "Here's the response."


class BaseLLMGenerator:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or MockLLMProvider()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = self.provider.generate(request.prompt)
        evaluation = GenerationEvaluation(
            response_nonempty=bool(output.response_text.strip()),
            includes_goal_alignment=request.prompt.current_subgoal.lower() in output.response_text.lower(),
            follows_uncertainty_guidance=(
                not request.prompt.response_plan.get("draft_constraints", {}).get("require_explicit_uncertainty", False)
                or "assumption" in output.response_text.lower()
                or "ambiguous" in output.response_text.lower()
            ),
        )
        logger.info(
            "generation_completed",
            extra={
                "response_mode": output.response_mode,
                "metadata": output.metadata.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return GenerationResponse(output=output, evaluation=evaluation)
