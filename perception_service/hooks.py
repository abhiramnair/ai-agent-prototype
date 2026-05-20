from __future__ import annotations

from abc import ABC, abstractmethod
import re

from .models import DetectedEntity, SalienceSignals, SurfaceFeatures, TurnInput
from .taxonomy import EmotionLabel, IntentLabel, ToneLabel, UrgencyLabel


class IntentHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[IntentLabel, float]:
        raise NotImplementedError


class TopicHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[str, float]:
        raise NotImplementedError


class ToneHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[ToneLabel, float]:
        raise NotImplementedError


class EmotionHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[EmotionLabel, float]:
        raise NotImplementedError


class UrgencyHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> UrgencyLabel:
        raise NotImplementedError


class ReferenceHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> bool:
        raise NotImplementedError


class EntityHook(ABC):
    @abstractmethod
    def extract(self, turn: TurnInput, features: SurfaceFeatures) -> list[DetectedEntity]:
        raise NotImplementedError


class SalienceHook(ABC):
    @abstractmethod
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> SalienceSignals:
        raise NotImplementedError


class DefaultIntentHook(IntentHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[IntentLabel, float]:
        text = features.lower_text
        if any(phrase in text for phrase in ["you are wrong", "that's wrong", "that is wrong", "correct this"]):
            return IntentLabel.CORRECTION, 0.94
        if text.startswith(("yes", "yep", "sure", "correct", "exactly")):
            return IntentLabel.CONFIRMATION, 0.88
        if text.startswith(("no", "nope", "not really")):
            return IntentLabel.REJECTION, 0.86
        if text.startswith(("hey", "hi", "hello")):
            return IntentLabel.SOCIAL_MESSAGE, 0.9
        if any(phrase in text for phrase in ["how are you", "how's it going", "hows it going", "what's up", "whats up"]):
            return IntentLabel.SOCIAL_MESSAGE, 0.88
        if any(phrase in text for phrase in ["i prefer", "i want", "i like", "please keep", "lets go with", "let's go with"]):
            return IntentLabel.PREFERENCE_STATEMENT, 0.83
        if any(phrase in text for phrase in ["brainstorm", "ideas", "options", "what if we"]) and features.has_question_mark:
            return IntentLabel.BRAINSTORMING, 0.8
        if any(phrase in text for phrase in ["explain", "go deeper", "simpler", "break down"]):
            return IntentLabel.ASK_EXPLANATION, 0.89
        if any(phrase in text for phrase in ["can you", "please implement", "build", "create", "add ", "set up"]):
            return IntentLabel.REQUEST_ACTION, 0.84
        if features.has_question_mark or text.startswith(("what", "why", "how", "when", "where", "who")):
            return IntentLabel.ASK_QUESTION, 0.81
        if any(phrase in text for phrase in ["hello", "hi", "hey", "thanks", "thank you", "how are you", "what's up", "whats up"]):
            return IntentLabel.SOCIAL_MESSAGE, 0.75
        if any(phrase in text for phrase in ["i'm worried", "i am worried", "frustrated", "upset", "confused"]):
            return IntentLabel.EMOTIONAL_EXPRESSION, 0.78
        if any(phrase in text for phrase in ["what do you mean", "which one", "can you clarify"]):
            return IntentLabel.CLARIFICATION, 0.77
        return IntentLabel.ASK_QUESTION, 0.42


class DefaultTopicHook(TopicHook):
    STOPWORDS = {
        "a", "an", "the", "and", "or", "to", "for", "of", "is", "it", "this", "that",
        "we", "you", "i", "on", "in", "with", "be", "do", "can", "will", "our",
    }

    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[str, float]:
        if turn.recent_context and turn.recent_context.active_topic and any(
            ref in features.pronoun_references for ref in ["this", "that", "it"]
        ):
            return turn.recent_context.active_topic, 0.84

        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]+\b", turn.message_text)
        filtered = [token for token in tokens if token.lower() not in self.STOPWORDS]
        if not filtered:
            return "general_conversation", 0.35
        topic = " ".join(filtered[:4]).lower()
        confidence = 0.7 if len(filtered) >= 2 else 0.52
        return topic, confidence


class DefaultToneHook(ToneHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[ToneLabel, float]:
        text = features.lower_text
        if features.exclamation_count >= 2 or "asap" in text or "urgent" in text:
            return ToneLabel.URGENT, 0.82
        if any(word in text for word in ["please", "let's", "lets", "together"]):
            return ToneLabel.COLLABORATIVE, 0.8
        if any(word in text for word in ["thanks", "thank you", "appreciate"]):
            return ToneLabel.FRIENDLY, 0.76
        if any(word in text for word in ["wrong", "bad", "frustrated", "annoying"]):
            return ToneLabel.FRUSTRATED, 0.78
        if len(turn.message_text.split()) <= 4:
            return ToneLabel.DIRECT, 0.68
        return ToneLabel.NEUTRAL, 0.58


class DefaultEmotionHook(EmotionHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> tuple[EmotionLabel, float]:
        text = features.lower_text
        if any(phrase in text for phrase in ["confused", "don't understand", "do not understand"]):
            return EmotionLabel.CONFUSION, 0.86
        if any(phrase in text for phrase in ["worried", "concerned", "anxious"]):
            return EmotionLabel.ANXIETY, 0.84
        if any(phrase in text for phrase in ["disappointed", "not happy", "unfortunately"]):
            return EmotionLabel.DISAPPOINTMENT, 0.8
        if any(phrase in text for phrase in ["great", "perfect", "sounds good"]):
            return EmotionLabel.SATISFACTION, 0.74
        if features.has_question_mark or any(phrase in text for phrase in ["how", "why", "what if", "deeper"]):
            return EmotionLabel.CURIOSITY, 0.7
        return EmotionLabel.NONE, 0.61


class DefaultUrgencyHook(UrgencyHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> UrgencyLabel:
        text = features.lower_text
        if any(term in text for term in ["asap", "immediately", "urgent", "right now"]):
            return UrgencyLabel.HIGH
        if any(term in text for term in ["soon", "today", "quickly"]):
            return UrgencyLabel.MEDIUM
        return UrgencyLabel.LOW


class DefaultReferenceHook(ReferenceHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> bool:
        if not turn.recent_context:
            return False
        if turn.recent_context.active_topic or turn.recent_context.recent_turn_summaries:
            return bool(features.pronoun_references or any(
                phrase in features.lower_text for phrase in ["again", "simpler", "same", "previous", "earlier"]
            ))
        return False


class DefaultEntityHook(EntityHook):
    ENTITY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,}|[a-z]+_[a-z0-9_]+)\b")

    def extract(self, turn: TurnInput, features: SurfaceFeatures) -> list[DetectedEntity]:
        entities: list[DetectedEntity] = []
        for match in self.ENTITY_PATTERN.finditer(turn.message_text):
            text = match.group(1)
            entity_type = "identifier" if "_" in text else "proper_noun"
            entities.append(
                DetectedEntity(
                    text=text,
                    type=entity_type,
                    confidence=0.72,
                    span_start=match.start(1),
                    span_end=match.end(1),
                )
            )
        return entities


class DefaultSalienceHook(SalienceHook):
    def classify(self, turn: TurnInput, features: SurfaceFeatures) -> SalienceSignals:
        text = features.lower_text
        return SalienceSignals(
            contains_correction=any(phrase in text for phrase in ["wrong", "correct this", "fix this", "instead"]),
            contains_preference=any(phrase in text for phrase in ["i prefer", "i want", "lets go with", "let's go with"]),
            contains_commitment=any(phrase in text for phrase in ["let's do", "lets do", "we will", "we'll", "lock this", "go with"]),
            contains_scope_shift=any(phrase in text for phrase in ["instead", "change direction", "different approach", "focus only"]),
            contains_rejection=text.startswith(("no", "nope", "not really")),
            contains_confirmation=text.startswith(("yes", "yep", "sure", "correct", "exactly")),
        )
