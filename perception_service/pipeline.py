from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any
from uuid import uuid4

from .hooks import (
    DefaultEmotionHook,
    DefaultEntityHook,
    DefaultIntentHook,
    DefaultReferenceHook,
    DefaultSalienceHook,
    DefaultToneHook,
    DefaultTopicHook,
    DefaultUrgencyHook,
    EmotionHook,
    EntityHook,
    IntentHook,
    ReferenceHook,
    SalienceHook,
    ToneHook,
    TopicHook,
    UrgencyHook,
)
from .models import PerceptionState, SurfaceFeatures, TurnInput
from .taxonomy import IntentLabel

logger = logging.getLogger("perception_service")


@dataclass
class HookRegistry:
    intent: IntentHook
    topic: TopicHook
    tone: ToneHook
    emotion: EmotionHook
    urgency: UrgencyHook
    reference: ReferenceHook
    entity: EntityHook
    salience: SalienceHook


def default_hook_registry() -> HookRegistry:
    return HookRegistry(
        intent=DefaultIntentHook(),
        topic=DefaultTopicHook(),
        tone=DefaultToneHook(),
        emotion=DefaultEmotionHook(),
        urgency=DefaultUrgencyHook(),
        reference=DefaultReferenceHook(),
        entity=DefaultEntityHook(),
        salience=DefaultSalienceHook(),
    )


class PerceptionPipeline:
    def __init__(self, hooks: HookRegistry | None = None) -> None:
        self.hooks = hooks or default_hook_registry()

    def analyze(self, turn: TurnInput) -> PerceptionState:
        started = time.perf_counter()
        request_id = f"perception-{turn.turn_id}-{uuid4().hex[:8]}"

        normalized_text = self.normalize_input(turn.message_text)
        features = self.extract_surface_features(turn, normalized_text)
        hook_outputs = self.run_logic_hooks(turn, features)
        state = self.assemble_perception_state(turn, features, hook_outputs)

        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "perception_analysis_completed",
            extra={
                "request_id": request_id,
                "normalized_input": normalized_text,
                "hook_outputs": hook_outputs,
                "final_state": state.model_dump(mode="json"),
                "confidence_values": {
                    "intent_confidence": state.intent_confidence,
                    "topic_confidence": state.topic_confidence,
                    "tone_confidence": state.tone_confidence,
                    "emotion_confidence": state.emotion_confidence,
                    "ambiguity_score": state.ambiguity_score,
                },
                "latency_ms": latency_ms,
            },
        )

        state.debug_signals.update(
            {
                "request_id": request_id,
                "normalized_text": normalized_text,
                "latency_ms": round(latency_ms, 2),
            }
        )
        return state

    def normalize_input(self, message_text: str) -> str:
        collapsed = re.sub(r"\s+", " ", message_text).strip()
        return collapsed

    def extract_surface_features(self, turn: TurnInput, normalized_text: str) -> SurfaceFeatures:
        lower_text = normalized_text.lower()
        pronoun_references = [
            match.group(0)
            for match in re.finditer(r"\b(this|that|it|those|these)\b", lower_text)
        ]
        matched_keywords = {
            "question_words": re.findall(r"\b(what|why|how|when|where|who)\b", lower_text),
            "preference_words": re.findall(r"\b(prefer|want|like)\b", lower_text),
            "action_words": re.findall(r"\b(build|create|implement|add|set up)\b", lower_text),
        }
        return SurfaceFeatures(
            normalized_text=normalized_text,
            lower_text=lower_text,
            token_count=len(normalized_text.split()),
            sentence_count=max(1, len(re.findall(r"[.!?]+", normalized_text))),
            has_question_mark="?" in normalized_text,
            exclamation_count=normalized_text.count("!"),
            pronoun_references=pronoun_references,
            matched_keywords=matched_keywords,
            context_available=turn.recent_context is not None,
            attachment_count=len(turn.attachments),
        )

    def run_logic_hooks(self, turn: TurnInput, features: SurfaceFeatures) -> dict[str, Any]:
        intent, intent_confidence = self.hooks.intent.classify(turn, features)
        topic, topic_confidence = self.hooks.topic.classify(turn, features)
        tone, tone_confidence = self.hooks.tone.classify(turn, features)
        emotion, emotion_confidence = self.hooks.emotion.classify(turn, features)
        urgency = self.hooks.urgency.classify(turn, features)
        references_prior_context = self.hooks.reference.classify(turn, features)
        entities = self.hooks.entity.extract(turn, features)
        salience = self.hooks.salience.classify(turn, features)
        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "topic": topic,
            "topic_confidence": topic_confidence,
            "tone": tone,
            "tone_confidence": tone_confidence,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "urgency": urgency,
            "references_prior_context": references_prior_context,
            "entities": entities,
            "salience": salience,
        }

    def score_ambiguity(self, turn: TurnInput, features: SurfaceFeatures, hook_outputs: dict[str, Any]) -> float:
        ambiguity = 0.08
        text = features.lower_text
        if hook_outputs["intent_confidence"] < 0.6:
            ambiguity += 0.28
        if hook_outputs["topic_confidence"] < 0.6:
            ambiguity += 0.18
        if hook_outputs["references_prior_context"] and not turn.recent_context:
            ambiguity += 0.2
        if features.pronoun_references and not turn.recent_context:
            ambiguity += 0.24
        if features.token_count <= 3:
            ambiguity += 0.16
        if hook_outputs["intent"] in {IntentLabel.ASK_QUESTION, IntentLabel.REQUEST_ACTION} and any(
            phrase in text for phrase in ["this", "that", "it"]
        ):
            ambiguity += 0.12
        return min(1.0, round(ambiguity, 2))

    def infer_possible_user_goal(self, turn: TurnInput, hook_outputs: dict[str, Any]) -> str:
        intent = hook_outputs["intent"]
        topic = hook_outputs["topic"]
        if intent == IntentLabel.REQUEST_ACTION:
            return f"get the assistant to perform or prepare work related to {topic}"
        if intent == IntentLabel.ASK_EXPLANATION:
            return f"understand {topic} more clearly"
        if intent == IntentLabel.CORRECTION:
            return f"correct the assistant's understanding about {topic}"
        if intent == IntentLabel.PREFERENCE_STATEMENT:
            return f"set a durable preference related to {topic}"
        if intent == IntentLabel.BRAINSTORMING:
            return f"explore options related to {topic}"
        return f"advance the conversation around {topic}"

    def assemble_perception_state(
        self,
        turn: TurnInput,
        features: SurfaceFeatures,
        hook_outputs: dict[str, Any],
    ) -> PerceptionState:
        ambiguity_score = self.score_ambiguity(turn, features, hook_outputs)
        return PerceptionState(
            primary_intent=hook_outputs["intent"],
            intent_confidence=hook_outputs["intent_confidence"],
            topic=hook_outputs["topic"],
            topic_confidence=hook_outputs["topic_confidence"],
            tone=hook_outputs["tone"],
            tone_confidence=hook_outputs["tone_confidence"],
            emotion=hook_outputs["emotion"],
            emotion_confidence=hook_outputs["emotion_confidence"],
            urgency=hook_outputs["urgency"],
            ambiguity_score=ambiguity_score,
            references_prior_context=hook_outputs["references_prior_context"],
            entities=hook_outputs["entities"],
            possible_user_goal=self.infer_possible_user_goal(turn, hook_outputs),
            salience_signals=hook_outputs["salience"],
            debug_signals={
                "surface_features": features.model_dump(mode="json"),
                "hook_outputs": {
                    key: value.model_dump(mode="json") if hasattr(value, "model_dump") else str(value)
                    for key, value in hook_outputs.items()
                    if key not in {"entities"}
                }
                | {
                    "entities": [entity.model_dump(mode="json") for entity in hook_outputs["entities"]],
                },
            },
        )
