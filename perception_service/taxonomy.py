from enum import StrEnum


class IntentLabel(StrEnum):
    ASK_QUESTION = "ask_question"
    ASK_EXPLANATION = "ask_explanation"
    REQUEST_ACTION = "request_action"
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    CONFIRMATION = "confirmation"
    REJECTION = "rejection"
    PREFERENCE_STATEMENT = "preference_statement"
    BRAINSTORMING = "brainstorming"
    SOCIAL_MESSAGE = "social_message"
    EMOTIONAL_EXPRESSION = "emotional_expression"


class ToneLabel(StrEnum):
    NEUTRAL = "neutral"
    DIRECT = "direct"
    COLLABORATIVE = "collaborative"
    FRIENDLY = "friendly"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"


class EmotionLabel(StrEnum):
    NONE = "none"
    CURIOSITY = "curiosity"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"
    ANXIETY = "anxiety"


class UrgencyLabel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
