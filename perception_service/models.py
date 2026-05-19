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


class DraftConstraints(BaseModel):
    avoid_repetition: bool = True
    avoid_overclaiming: bool = True
    avoid_scope_drift: bool = True
    prefer_examples: bool = False
    require_explicit_uncertainty: bool = False


class DialoguePlan(BaseModel):
    response_mode: str
    primary_goal: str
    secondary_goal: str
    reasoning_style: str
    tone: str
    detail_level: str
    clarification_policy: str
    memory_use_policy: str
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    draft_constraints: DraftConstraints = Field(default_factory=DraftConstraints)
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class DialoguePlanRequest(BaseModel):
    turn_input: TurnInput
    perception_state: PerceptionState
    working_memory_state: WorkingMemoryState


class DialoguePlanEvaluation(BaseModel):
    used_recent_context: bool
    asks_for_clarification: bool
    ambiguity_sensitive: bool
    plan_has_required_fields: bool


class DialoguePlanResponse(BaseModel):
    plan: DialoguePlan
    evaluation: DialoguePlanEvaluation


class PromptAssembly(BaseModel):
    system_role: str
    current_user_message: str
    active_goal: str
    current_subgoal: str
    relevant_recent_context: list[str] = Field(default_factory=list)
    working_memory_snapshot: dict[str, Any] = Field(default_factory=dict)
    perception_summary: dict[str, Any] = Field(default_factory=dict)
    response_plan: dict[str, Any] = Field(default_factory=dict)
    instructions: list[str] = Field(default_factory=list)
    rendered_prompt: str
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class PromptAssemblyRequest(BaseModel):
    turn_input: TurnInput
    perception_state: PerceptionState
    working_memory_state: WorkingMemoryState
    dialogue_plan: DialoguePlan


class PromptAssemblyEvaluation(BaseModel):
    includes_recent_context: bool
    includes_constraints: bool
    includes_uncertainty_guidance: bool
    rendered_prompt_nonempty: bool


class PromptAssemblyResponse(BaseModel):
    prompt: PromptAssembly
    evaluation: PromptAssemblyEvaluation


class GenerationMetadata(BaseModel):
    provider_name: str
    model_name: str
    finish_reason: str
    latency_ms: float = Field(ge=0.0)
    token_usage: dict[str, int] = Field(default_factory=dict)


class GeneratorOutput(BaseModel):
    response_text: str
    response_mode: str
    metadata: GenerationMetadata
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    prompt: PromptAssembly


class GenerationEvaluation(BaseModel):
    response_nonempty: bool
    includes_goal_alignment: bool
    follows_uncertainty_guidance: bool


class GenerationResponse(BaseModel):
    output: GeneratorOutput
    evaluation: GenerationEvaluation


class CriticFinding(BaseModel):
    severity: str
    category: str
    message: str


class CriticScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    faithfulness_to_plan: float = Field(ge=0.0, le=1.0)
    tone_fit: float = Field(ge=0.0, le=1.0)
    hallucination_risk: float = Field(ge=0.0, le=1.0)


class CriticReview(BaseModel):
    passed: bool
    scores: CriticScores
    findings: list[CriticFinding] = Field(default_factory=list)
    recommended_edits: list[str] = Field(default_factory=list)
    debug_signals: dict[str, Any] = Field(default_factory=dict)


class CriticRequest(BaseModel):
    prompt: PromptAssembly
    generation_output: GeneratorOutput


class CriticEvaluation(BaseModel):
    has_findings: bool
    requires_revision: bool
    score_summary: float = Field(ge=0.0, le=1.0)


class CriticResponse(BaseModel):
    review: CriticReview
    evaluation: CriticEvaluation


class MemoryRecord(BaseModel):
    memory_id: str
    memory_type: str
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_turn_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    reinforcement_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    archived: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpsertRequest(BaseModel):
    memory_type: str
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_turn_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_id: str | None = None


class MemoryQueryRequest(BaseModel):
    memory_type: str | None = None
    query_text: str | None = None
    tags: list[str] = Field(default_factory=list)
    include_archived: bool = False
    limit: int = Field(default=10, ge=1, le=100)


class MemoryQueryEvaluation(BaseModel):
    total_matches: int = Field(ge=0)
    used_type_filter: bool
    used_text_filter: bool
    used_tag_filter: bool


class MemoryQueryResponse(BaseModel):
    memories: list[MemoryRecord] = Field(default_factory=list)
    evaluation: MemoryQueryEvaluation


class MemoryMutationResponse(BaseModel):
    memory: MemoryRecord
    created: bool


class MemoryArchiveRequest(BaseModel):
    memory_id: str


class MemoryArchiveResponse(BaseModel):
    memory: MemoryRecord
    archived: bool


class RetrievedMemory(BaseModel):
    memory_id: str
    memory_type: str
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source_turn_id: str | None = None
    retrieval_reason: str


class MemoryRetrievalRequest(BaseModel):
    query_text: str
    memory_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    include_archived: bool = False


class MemoryRetrievalEvaluation(BaseModel):
    retrieved_count: int = Field(ge=0)
    used_type_filters: bool
    used_tag_filters: bool
    top_score: float = Field(ge=0.0, le=1.0)


class MemoryRetrievalResponse(BaseModel):
    memories: list[RetrievedMemory] = Field(default_factory=list)
    evaluation: MemoryRetrievalEvaluation
