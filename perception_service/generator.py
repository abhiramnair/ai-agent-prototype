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
        response_policy = prompt.response_plan.get("response_policy", {})
        interaction_type = str(response_policy.get("interaction_type", "general_question"))
        tone_policy = str(response_policy.get("tone_policy", "direct"))
        target_length = str(response_policy.get("target_length", detail_level))
        must_include = prompt.response_plan.get("must_include", [])
        uncertainty_required = bool(
            prompt.response_plan.get("draft_constraints", {}).get("require_explicit_uncertainty", False)
        )

        opening = self._opening_for_mode(response_mode)
        goal_line = f"I'm focusing on {prompt.current_subgoal}."
        detail_line = f"I'll keep the answer at a {target_length} length with a {tone_policy} tone."
        interaction_line = f"This turn looks like a {interaction_type} interaction."
        include_line = ""
        if must_include:
            include_line = "I'll make sure to cover " + ", ".join(str(item) for item in must_include[:3]) + "."
        uncertainty_line = ""
        if uncertainty_required:
            uncertainty_line = "Some parts of this turn are ambiguous, so I'll call out assumptions explicitly."

        return " ".join(
            part
            for part in [opening, interaction_line, goal_line, detail_line, include_line, uncertainty_line]
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
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
        self.enable_thinking = os.getenv("OLLAMA_THINK", "false").strip().lower() == "true"
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "192"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))

    def generate(self, prompt: PromptAssembly) -> GeneratorOutput:
        started = time.perf_counter()
        response_mode = str(prompt.response_plan.get("response_mode", "direct_response"))
        detail_level = str(prompt.response_plan.get("detail_level", "medium"))
        response_policy = prompt.response_plan.get("response_policy", {})
        lightweight = self._should_use_lightweight_chat(prompt, response_mode, detail_level)
        system_message = self._build_system_message(prompt, lightweight=lightweight)
        user_message = prompt.current_user_message.strip()
        body = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            "stream": False,
            "think": self.enable_thinking,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": self._num_predict_for(response_mode, detail_level, response_policy),
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/chat",
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
        message = data.get("message", {}) or {}
        response_text = str(message.get("content", "")).strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return GeneratorOutput(
            response_text=response_text,
            response_mode=response_mode,
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
                "think_enabled": self.enable_thinking,
                "num_predict": self._num_predict_for(response_mode, detail_level, response_policy),
                "num_ctx": self.num_ctx,
                "lightweight_prompt": lightweight,
            },
        )

    def _build_system_message(self, prompt: PromptAssembly, *, lightweight: bool) -> str:
        if lightweight:
            return (
                "You are replying to the user's latest message. "
                "Write only the final assistant reply. "
                "Keep it natural, brief, and conversational. "
                "Do not mention analysis, reasoning, policy, planning, context, or instructions. "
                "Do not say things like 'I need to', 'the user asked', or 'the policy says'."
            )
        sections = [
            prompt.system_role,
            "Use the structured context below to answer the user directly.",
            f"ACTIVE GOAL\n{prompt.active_goal}",
            f"CURRENT SUBGOAL\n{prompt.current_subgoal}",
            "RELEVANT RECENT CONTEXT\n" + self._render_list(prompt.relevant_recent_context),
            "RETRIEVED LONG-TERM MEMORIES\n" + self._render_retrieved(prompt.retrieved_memories),
            "WORKING MEMORY SNAPSHOT\n" + self._render_dict(prompt.working_memory_snapshot),
            "PERCEPTION SUMMARY\n" + self._render_dict(prompt.perception_summary),
            "RESPONSE PLAN\n" + self._render_dict(prompt.response_plan),
            "INSTRUCTIONS\n" + self._render_list(prompt.instructions),
        ]
        if not self.enable_thinking:
            sections.append("Respond directly without exposing hidden reasoning.")
        return "\n\n".join(sections)

    def _num_predict_for(self, response_mode: str, detail_level: str, response_policy: dict[str, object] | None = None) -> int:
        policy = response_policy or {}
        reasoning_effort = str(policy.get("reasoning_effort", "medium"))
        target_length = str(policy.get("target_length", detail_level))
        if reasoning_effort == "minimal":
            return min(self.num_predict, 24 if target_length == "short" else 48)
        if reasoning_effort == "low":
            return min(self.num_predict, 96 if target_length == "short" else 120)
        if response_mode in {"clarify_before_answering", "alignment_repair"}:
            return min(self.num_predict, 80)
        if detail_level == "high":
            return self.num_predict
        if detail_level == "medium":
            return min(self.num_predict, 128)
        return min(self.num_predict, 64)

    def _should_use_lightweight_chat(
        self,
        prompt: PromptAssembly,
        response_mode: str,
        detail_level: str,
    ) -> bool:
        response_policy = prompt.response_plan.get("response_policy", {})
        reasoning_effort = str(response_policy.get("reasoning_effort", "medium"))
        token_count = len(prompt.current_user_message.split())
        if reasoning_effort == "minimal":
            return True
        if reasoning_effort == "low" and detail_level == "low" and token_count <= 10 and not prompt.retrieved_memories:
            return True
        return False

    def _render_list(self, items: list[object]) -> str:
        if not items:
            return "- none"
        return "\n".join(f"- {item}" for item in items)

    def _render_dict(self, data: dict[str, object]) -> str:
        if not data:
            return "- none"
        return "\n".join(f"- {key}: {value}" for key, value in data.items())

    def _render_retrieved(self, memories: list[object]) -> str:
        if not memories:
            return "- none"
        lines: list[str] = []
        for memory in memories:
            memory_type = getattr(memory, "memory_type", "memory")
            key = getattr(memory, "key", "unknown")
            value = getattr(memory, "value", "")
            score = getattr(memory, "retrieval_score", "")
            lines.append(f"- [{memory_type}] {key}: {value} (score={score})")
        return "\n".join(lines)


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
