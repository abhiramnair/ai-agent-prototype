from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .taxonomy import EmotionLabel, IntentLabel, ToneLabel, UrgencyLabel


class RecentContext(BaseModel):
    recent_turn_summaries: list[str] = Field(default_factory=list)
    active_topic: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    conversation_mode: str | None = None
    last_assistant_action: str | None = None


class AttachmentStub(BaseModel):
    name: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnInput(BaseModel):
    turn_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    message_text: str
    channel: str | None = "chat"
    recent_context: RecentContext | None = None
    attachments: list[AttachmentStub] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_text")
    @classmethod
    def validate_message_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message_text must not be empty")
        return value


class DetectedEntity(BaseModel):
    text: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> "DetectedEntity":
        if self.span_end < self.span_start:
            raise ValueError("span_end must be >= span_start")
        return self


class SalienceSignals(BaseModel):
    contains_correction: bool = False
    contains_preference: bool = False
    contains_commitment: bool = False
    contains_scope_shift: bool = False
    contains_rejection: bool = False
    contains_confirmation: bool = False


class PerceptionState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    primary_intent: IntentLabel
    intent_confidence: float = Field(ge=0.0, le=1.0)
    topic: str
    topic_confidence: float = Field(ge=0.0, le=1.0)
    tone: ToneLabel
    tone_confidence: float = Field(ge=0.0, le=1.0)
    emotion: EmotionLabel
    emotion_confidence: float = Field(ge=0.0, le=1.0)
    urgency: UrgencyLabel
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    references_prior_context: bool
    entities: list[DetectedEntity] = Field(default_factory=list)
    possible_user_goal: str
    salience_signals: SalienceSignals
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class SurfaceFeatures(BaseModel):
    normalized_text: str
    lower_text: str
    token_count: int
    sentence_count: int
    has_question_mark: bool
    exclamation_count: int
    pronoun_references: list[str] = Field(default_factory=list)
    matched_keywords: dict[str, list[str]] = Field(default_factory=dict)
    context_available: bool = False
    attachment_count: int = 0


class PerceptionDebugRecord(BaseModel):
    request_id: str
    normalized_text: str
    hook_outputs: dict[str, Any]
    confidence_values: dict[str, float]
    latency_ms: float


class FixtureExpectation(BaseModel):
    primary_intent: IntentLabel | None = None
    references_prior_context: bool | None = None
    minimum_intent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_ambiguity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    salience_flags: dict[str, bool] = Field(default_factory=dict)


class LabeledFixture(BaseModel):
    name: str
    turn_input: TurnInput
    expected: FixtureExpectation


class FixtureResult(BaseModel):
    name: str
    passed: bool
    actual: PerceptionState
    mismatches: list[str] = Field(default_factory=list)


class WorkingMemoryState(BaseModel):
    active_goal: str
    current_subgoal: str
    conversation_mode: str
    response_mode: str
    active_entities: list[str] = Field(default_factory=list)
    temporary_assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recent_turns_compact: list[str] = Field(default_factory=list)
    emotional_context: str = "stable"
    attention_targets: list[str] = Field(default_factory=list)
    suppressed_topics: list[str] = Field(default_factory=list)
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryUpdateRequest(BaseModel):
    turn_input: TurnInput
    perception_state: PerceptionState
    current_state: WorkingMemoryState | None = None


class WorkingMemoryEvaluation(BaseModel):
    active_goal_changed: bool
    subgoal_changed: bool
    references_carried_forward: bool
    unresolved_questions_count: int
    active_entities_count: int


class WorkingMemoryUpdateResponse(BaseModel):
    state: WorkingMemoryState
    evaluation: WorkingMemoryEvaluation
