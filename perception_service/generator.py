from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
import time
from urllib import error, request

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


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))

    def generate(self, prompt: PromptAssembly) -> GeneratorOutput:
        started = time.perf_counter()
        body = {
            "model": self.model_name,
            "prompt": prompt.rendered_prompt,
            "stream": False,
        }
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc

        data = json.loads(raw)
        response_text = str(data.get("response", "")).strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return GeneratorOutput(
            response_text=response_text,
            response_mode=str(prompt.response_plan.get("response_mode", "direct_response")),
            metadata=GenerationMetadata(
                provider_name=self.provider_name,
                model_name=self.model_name,
                finish_reason=str(data.get("done_reason", "completed")),
                latency_ms=round(latency_ms, 2),
                token_usage={
                    "prompt_eval_count": int(data.get("prompt_eval_count", 0) or 0),
                    "eval_count": int(data.get("eval_count", 0) or 0),
                },
            ),
            debug_signals={
                "base_url": self.base_url,
                "instruction_count": len(prompt.instructions),
                "used_ollama_provider": True,
            },
        )


def build_default_provider() -> LLMProvider:
    provider_name = os.getenv("AI_AGENT_LLM_PROVIDER", "ollama").strip().lower()
    if provider_name == "mock":
        return MockLLMProvider()
    if provider_name == "ollama":
        return OllamaProvider()
    return MockLLMProvider()


class BaseLLMGenerator:
    def __init__(
        self,
        provider: LLMProvider | None = None,
        fallback_provider: LLMProvider | None = None,
    ) -> None:
        self.provider = provider or build_default_provider()
        self.fallback_provider = fallback_provider or MockLLMProvider()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = self._generate_with_fallback(request.prompt)
        evaluation = GenerationEvaluation(
            response_nonempty=bool(output.response_text.strip()),
            includes_goal_alignment=request.prompt.current_subgoal.lower() in output.response_text.lower(),
            follows_uncertainty_guidance=(
                not request.prompt.response_plan.get("draft_constraints", {}).get("require_explicit_uncertainty", False)
                or "assumption" in output.response_text.lower()
                or "ambiguous" in output.response_text.lower()
                or "uncertain" in output.response_text.lower()
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

    def _generate_with_fallback(self, prompt: PromptAssembly) -> GeneratorOutput:
        try:
            return self.provider.generate(prompt)
        except Exception as exc:
            logger.warning("primary_generation_provider_failed", extra={"error": str(exc)})
            fallback_output = self.fallback_provider.generate(prompt)
            fallback_output.debug_signals.update(
                {
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                    "preferred_provider": getattr(self.provider, "provider_name", self.provider.__class__.__name__),
                    "preferred_model": getattr(self.provider, "model_name", "unknown"),
                }
            )
            return fallback_output
