from datetime import datetime
import os

from fastapi.testclient import TestClient
import pytest

os.environ["AI_AGENT_LLM_PROVIDER"] = "mock"

from app import app
from perception_service.dialogue_planner import DialoguePlanner
from perception_service.generator import BaseLLMGenerator, LLMProvider
from perception_service.memory_retriever import MemoryRetriever
from perception_service.memory_store import MemoryStore
from perception_service.models import (
    CriticReview,
    CriticScores,
    DialoguePlanRequest,
    GenerationMetadata,
    GenerationRequest,
    GeneratorOutput,
    MemoryUpsertRequest,
    PromptAssembly,
    RecentContext,
    TurnInput,
)


client = TestClient(app)


def make_turn(**overrides):
    base = {
        "turn_id": "turn-1",
        "session_id": "session-1",
        "user_id": "user-1",
        "timestamp": datetime(2026, 5, 19, 10, 0, 0).isoformat(),
        "message_text": "Can you explain this more simply?",
    }
    base.update(overrides)
    return base


def test_rejects_missing_required_fields():
    response = client.post("/perception/analyze", json={"message_text": "hi"})
    assert response.status_code == 422


def test_chat_ui_root_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Talk to the orchestrated agent runtime." in response.text


def test_chat_ui_static_assets_load():
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "agent.run" not in response.text.lower() or "fetch(\"/agent/run\"" in response.text.lower()


def test_rejects_empty_message_text():
    with pytest.raises(ValueError):
        TurnInput(
            turn_id="turn-1",
            session_id="session-1",
            user_id="user-1",
            timestamp=datetime(2026, 5, 19, 10, 0, 0),
            message_text="   ",
        )


def test_contract_response_shape_and_enums():
    response = client.post("/perception/analyze", json=make_turn())
    assert response.status_code == 200
    payload = response.json()
    assert payload["primary_intent"] == "ask_explanation"
    assert 0.0 <= payload["intent_confidence"] <= 1.0
    assert payload["tone"] in {"neutral", "direct", "collaborative", "friendly", "frustrated", "urgent"}
    assert "ambiguity_score" in payload
    assert isinstance(payload["entities"], list)


def test_context_reference_case_again():
    response = client.post(
        "/perception/analyze",
        json=make_turn(
            message_text="Do that again.",
            recent_context={
                "recent_turn_summaries": ["Assistant proposed a Perception blueprint."],
                "active_topic": "perception module design",
                "unresolved_questions": [],
                "conversation_mode": "technical_collaboration",
                "last_assistant_action": "summarized architecture",
            },
        ),
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["references_prior_context"] is True
    assert payload["topic"] == "perception module design"


def test_low_confidence_still_returns_valid_state():
    response = client.post("/perception/analyze", json=make_turn(message_text="This?"))
    payload = response.json()
    assert response.status_code == 200
    assert payload["primary_intent"]
    assert payload["ambiguity_score"] >= 0.0


def test_unsupported_attachment_metadata_is_accepted():
    response = client.post(
        "/perception/analyze",
        json=make_turn(
            attachments=[
                {
                    "name": "design.png",
                    "content_type": "image/png",
                    "metadata": {"note": "reserved for later"},
                }
            ]
        ),
    )
    assert response.status_code == 200


def test_recent_context_model_parses():
    turn = TurnInput(
        turn_id="turn-2",
        session_id="session-2",
        user_id="user-2",
        timestamp=datetime(2026, 5, 19, 10, 0, 0),
        message_text="Make it simpler.",
        recent_context=RecentContext(
            recent_turn_summaries=["Previous explanation about module interfaces."],
            active_topic="module interfaces",
            unresolved_questions=["How the planner chooses a mode"],
            conversation_mode="design",
            last_assistant_action="explained the blueprint",
        ),
    )
    assert turn.recent_context.active_topic == "module interfaces"


def test_working_memory_update_endpoint_returns_state():
    perception_response = client.post(
        "/perception/analyze",
        json=make_turn(
            message_text="Can you explain the working memory manager more clearly?",
            recent_context={
                "recent_turn_summaries": ["We finished the Perception module."],
                "active_topic": "working memory manager",
                "unresolved_questions": ["What should active state contain?"],
                "conversation_mode": "technical_collaboration",
                "last_assistant_action": "implemented perception",
            },
        ),
    )
    assert perception_response.status_code == 200

    response = client.post(
        "/working-memory/update",
        json={
            "turn_input": make_turn(
                turn_id="turn-2",
                message_text="Can you explain the working memory manager more clearly?",
                recent_context={
                    "recent_turn_summaries": ["We finished the Perception module."],
                    "active_topic": "working memory manager",
                    "unresolved_questions": ["What should active state contain?"],
                    "conversation_mode": "technical_collaboration",
                    "last_assistant_action": "implemented perception",
                },
            ),
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["active_goal"]
    assert payload["state"]["current_subgoal"]
    assert payload["state"]["conversation_mode"] == "technical_collaboration"
    assert "working memory manager" in " ".join(payload["state"]["attention_targets"]).lower()
    assert payload["evaluation"]["unresolved_questions_count"] >= 1


def test_attention_gate_returns_focus_state():
    turn = make_turn(
        turn_id="turn-attn-1",
        message_text="Can you explain the working memory manager more clearly?",
        recent_context={
            "recent_turn_summaries": ["We finished the Perception module."],
            "active_topic": "working memory manager",
            "unresolved_questions": ["What should active state contain?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented perception",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    response = client.post(
        "/attention-gate/evaluate",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["primary_focus"]
    assert payload["state"]["focus_targets"]
    assert payload["evaluation"]["focus_target_count"] >= 1


def test_working_memory_carries_forward_existing_state():
    current_state = {
        "active_goal": "build the conversational architecture",
        "current_subgoal": "finish the perception module",
        "conversation_mode": "technical_collaboration",
        "response_mode": "action_oriented",
        "active_entities": ["Perception"],
        "temporary_assumptions": ["The user wants continuity."],
        "unresolved_questions": ["How should perception output be scored?"],
        "recent_turns_compact": ["request_action: implement the perception module"],
        "emotional_context": "stable",
        "attention_targets": ["Perception"],
        "suppressed_topics": [],
        "debug_signals": {},
    }

    perception_response = client.post(
        "/perception/analyze",
        json=make_turn(
            turn_id="turn-3",
            message_text="Make it simpler.",
            recent_context={
                "recent_turn_summaries": ["Assistant explained the working memory manager."],
                "active_topic": "working memory manager",
                "unresolved_questions": [],
                "conversation_mode": "technical_collaboration",
                "last_assistant_action": "explained the design",
            },
        ),
    )
    assert perception_response.status_code == 200

    response = client.post(
        "/working-memory/update",
        json={
            "turn_input": make_turn(
                turn_id="turn-3",
                message_text="Make it simpler.",
                recent_context={
                    "recent_turn_summaries": ["Assistant explained the working memory manager."],
                    "active_topic": "working memory manager",
                    "unresolved_questions": [],
                    "conversation_mode": "technical_collaboration",
                    "last_assistant_action": "explained the design",
                },
            ),
            "perception_state": perception_response.json(),
            "current_state": current_state,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["active_goal"]
    assert "Perception" in payload["state"]["active_entities"]
    assert payload["state"]["recent_turns_compact"]


def test_dialogue_planner_returns_structured_plan():
    turn = make_turn(
        turn_id="turn-4",
        message_text="Can you explain the dialogue planner with examples?",
        recent_context={
            "recent_turn_summaries": ["Working memory manager is now implemented."],
            "active_topic": "dialogue planner",
            "unresolved_questions": ["How should it choose response strategy?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented working memory",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["plan"]["response_mode"] == "structured_explanation"
    assert payload["plan"]["detail_level"] == "high"
    assert payload["plan"]["response_policy"]["interaction_type"] == "knowledge_explanation"
    assert payload["plan"]["response_policy"]["reasoning_effort"] == "high"
    assert payload["plan"]["draft_constraints"]["prefer_examples"] is True
    assert payload["evaluation"]["plan_has_required_fields"] is True


def test_dialogue_planner_handles_ambiguous_turns():
    turn = make_turn(
        turn_id="turn-5",
        message_text="This?",
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["plan"]["clarification_policy"] in {
        "ask_one_targeted_question",
        "answer_with_assumption_if_safe",
    }
    assert payload["plan"]["draft_constraints"]["require_explicit_uncertainty"] is True


def test_prompt_assembler_returns_structured_prompt():
    turn = make_turn(
        turn_id="turn-6",
        message_text="Can you explain the prompt assembler with a concrete example?",
        recent_context={
            "recent_turn_summaries": ["Dialogue planner was just implemented."],
            "active_topic": "prompt assembler",
            "unresolved_questions": ["What should the model actually receive?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented dialogue planner",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["prompt"]["system_role"]
    assert payload["prompt"]["rendered_prompt"]
    assert "SYSTEM ROLE" in payload["prompt"]["rendered_prompt"]
    assert "CURRENT USER MESSAGE" in payload["prompt"]["rendered_prompt"]
    assert "ATTENTION GATE" in payload["prompt"]["rendered_prompt"]
    assert "ASSISTANT RESPONSE" in payload["prompt"]["rendered_prompt"]
    assert payload["evaluation"]["includes_recent_context"] is True
    assert payload["evaluation"]["includes_constraints"] is True


def test_prompt_assembler_includes_uncertainty_guidance_when_needed():
    turn = make_turn(
        turn_id="turn-7",
        message_text="This?",
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["includes_uncertainty_guidance"] is True


def test_generator_returns_response_from_assembled_prompt():
    turn = make_turn(
        turn_id="turn-8",
        message_text="Can you explain how the base LLM generator should work?",
        recent_context={
            "recent_turn_summaries": ["Prompt assembler was just added."],
            "active_topic": "base llm generator",
            "unresolved_questions": ["How do we swap providers later?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented prompt assembly",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    prompt_response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    assert prompt_response.status_code == 200

    response = client.post(
        "/generator/generate",
        json={"prompt": prompt_response.json()["prompt"]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["output"]["response_text"]
    assert payload["output"]["metadata"]["provider_name"] in {"mock", "ollama"}
    assert payload["evaluation"]["response_nonempty"] is True


def test_generator_marks_uncertainty_guidance_for_ambiguous_prompt():
    turn = make_turn(
        turn_id="turn-9",
        message_text="This?",
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    prompt_response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    assert prompt_response.status_code == 200

    response = client.post(
        "/generator/generate",
        json={"prompt": prompt_response.json()["prompt"]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["follows_uncertainty_guidance"] is True


def test_generator_repairs_meta_reasoning_output():
    class LeakyProvider(LLMProvider):
        def generate(self, prompt: PromptAssembly) -> GeneratorOutput:
            return GeneratorOutput(
                response_text='Okay, the user asked "how are you". I need to respond naturally.',
                response_mode="social_reply",
                metadata=GenerationMetadata(
                    provider_name="test-provider",
                    model_name="test-model",
                    finish_reason="completed",
                    latency_ms=1.0,
                    token_usage={},
                ),
                debug_signals={},
            )

        def rewrite_to_final(self, prompt: PromptAssembly, draft: str) -> GeneratorOutput | None:
            return GeneratorOutput(
                response_text="I'm doing well, thanks for asking. How are you?",
                response_mode="social_reply",
                metadata=GenerationMetadata(
                    provider_name="test-provider",
                    model_name="test-model",
                    finish_reason="rewritten_final",
                    latency_ms=1.0,
                    token_usage={},
                ),
                debug_signals={},
            )

    prompt = PromptAssembly(
        system_role="You are the conversation generator in a modular cognitive system.",
        current_user_message="how are you",
        active_goal="continue the conversation",
        current_subgoal="reply naturally to the user",
        relevant_recent_context=[],
        retrieved_memories=[],
        working_memory_snapshot={},
        perception_summary={},
        response_plan={
            "response_mode": "social_reply",
            "detail_level": "low",
            "draft_constraints": {"require_explicit_uncertainty": False},
            "response_policy": {
                "interaction_type": "social_exchange",
                "reasoning_effort": "minimal",
                "target_length": "short",
                "tone_policy": "warm",
                "retrieval_policy": "minimal_retrieval",
                "example_policy": "examples_optional",
                "confidence_policy": "answer_directly",
                "adaptation_hints": [],
                "learning_objective": "Learn what social responses work best.",
            },
        },
        instructions=[],
        rendered_prompt="test prompt",
        debug_signals={},
    )

    generator = BaseLLMGenerator(provider=LeakyProvider())
    response = generator.generate(GenerationRequest(prompt=prompt))

    assert response.output.response_text == "I'm doing well, thanks for asking. How are you?"
    assert response.output.metadata.finish_reason == "rewritten_final"
    assert response.output.debug_signals["runtime_repair_applied"] is True


def test_generator_uses_adaptive_policy_for_simple_social_turn():
    turn = make_turn(
        turn_id="turn-9b",
        message_text="how are you",
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200
    assert perception_response.json()["primary_intent"] == "social_message"

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200
    assert planner_response.json()["plan"]["response_mode"] == "social_reply"
    assert planner_response.json()["plan"]["response_policy"]["reasoning_effort"] == "minimal"
    assert planner_response.json()["plan"]["response_policy"]["target_length"] == "short"
    assert planner_response.json()["plan"]["must_include"] == []

    prompt_response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    assert prompt_response.status_code == 200

    response = client.post(
        "/generator/generate",
        json={"prompt": prompt_response.json()["prompt"]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["output"]["response_text"]


def test_critic_reviews_generated_response():
    turn = make_turn(
        turn_id="turn-10",
        message_text="Can you explain how the response critic should work?",
        recent_context={
            "recent_turn_summaries": ["Base LLM generator was just added."],
            "active_topic": "response critic",
            "unresolved_questions": ["How should draft quality be scored?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented base generation",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    prompt_response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    assert prompt_response.status_code == 200

    generation_response = client.post(
        "/generator/generate",
        json={"prompt": prompt_response.json()["prompt"]},
    )
    assert generation_response.status_code == 200

    response = client.post(
        "/critic/review",
        json={
            "prompt": prompt_response.json()["prompt"],
            "generation_output": generation_response.json()["output"],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert "scores" in payload["review"]
    assert payload["evaluation"]["score_summary"] >= 0.0


def test_critic_flags_missing_uncertainty_when_required():
    prompt_payload = {
        "system_role": "You are the conversation generator in a modular cognitive system.",
        "current_user_message": "This?",
        "active_goal": "advance the conversation around general_conversation",
        "current_subgoal": "continue the conversation around general_conversation",
        "relevant_recent_context": [],
        "working_memory_snapshot": {},
        "perception_summary": {},
        "response_plan": {
            "response_mode": "clarify_before_answering",
            "primary_goal": "advance the conversation around general_conversation",
            "secondary_goal": "continue the conversation around general_conversation",
            "reasoning_style": "direct",
            "tone": "clear_direct",
            "detail_level": "medium",
            "clarification_policy": "ask_one_targeted_question",
            "memory_use_policy": "retrieve_minimal_relevant_context",
            "must_include": ["continue the conversation around general_conversation"],
            "must_avoid": ["unsupported claims"],
            "draft_constraints": {
                "avoid_repetition": True,
                "avoid_overclaiming": True,
                "avoid_scope_drift": True,
                "prefer_examples": False,
                "require_explicit_uncertainty": True,
            },
            "response_policy": {
                "interaction_type": "ambiguity_resolution",
                "reasoning_effort": "medium",
                "target_length": "medium",
                "tone_policy": "direct",
                "retrieval_policy": "minimal_retrieval",
                "example_policy": "examples_optional",
                "confidence_policy": "surface_uncertainty_and_clarify",
                "adaptation_hints": ["State uncertainty explicitly when assumptions are necessary."],
                "learning_objective": "Learn whether ambiguity_resolution turns work best with medium reasoning effort and medium responses.",
            },
        },
        "instructions": ["State uncertainty explicitly when assumptions are necessary."],
        "rendered_prompt": "test prompt",
        "debug_signals": {},
    }
    generation_output = {
        "response_text": "Here is a short answer with no caution.",
        "response_mode": "clarify_before_answering",
        "metadata": {
            "provider_name": "mock",
            "model_name": "mock-conversation-generator-v1",
            "finish_reason": "completed",
            "latency_ms": 1.0,
            "token_usage": {},
        },
        "debug_signals": {},
    }

    response = client.post(
        "/critic/review",
        json={
            "prompt": prompt_payload,
            "generation_output": generation_output,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["requires_revision"] is True
    assert any(finding["category"] == "uncertainty" for finding in payload["review"]["findings"])


def test_memory_store_upsert_and_query_round_trip():
    upsert_response = client.post(
        "/memory/upsert",
        json={
            "memory_type": "preference",
            "key": "answer_style.depth",
            "value": "deep",
            "confidence": 0.88,
            "source_turn_id": "turn-pref-1",
            "tags": ["preferences", "style"],
            "evidence": ["User asked to go deeper into architecture."],
            "metadata": {"scope": "conversation"},
        },
    )
    upsert_payload = upsert_response.json()
    assert upsert_response.status_code == 200
    assert upsert_payload["created"] is True
    assert upsert_payload["memory"]["reinforcement_count"] == 1

    query_response = client.post(
        "/memory/query",
        json={
            "memory_type": "preference",
            "query_text": "answer_style.depth deep",
            "tags": ["preferences"],
            "limit": 5,
        },
    )
    query_payload = query_response.json()
    assert query_response.status_code == 200
    assert query_payload["memories"]
    assert query_payload["memories"][0]["key"] == "answer_style.depth"
    assert query_payload["evaluation"]["used_type_filter"] is True


def test_memory_store_reinforces_existing_record():
    first = client.post(
        "/memory/upsert",
        json={
            "memory_type": "procedural",
            "key": "planning.next_step",
            "value": "move to memory store after the first conversational loop",
            "confidence": 0.81,
            "source_turn_id": "turn-proc-1",
            "tags": ["planning"],
            "evidence": ["Conversation sequence locked after response critic."],
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/memory/upsert",
        json={
            "memory_type": "procedural",
            "key": "planning.next_step",
            "value": "move to memory store after the first conversational loop",
            "confidence": 0.9,
            "source_turn_id": "turn-proc-2",
            "tags": ["planning", "memory"],
            "evidence": ["User confirmed memory store is the next step."],
        },
    )
    payload = second.json()
    assert second.status_code == 200
    assert payload["created"] is False
    assert payload["memory"]["reinforcement_count"] >= 2
    assert "memory" in payload["memory"]["tags"]


def test_memory_store_archive_hides_records_by_default():
    create_response = client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "milestone.first_loop",
            "value": "Completed the first conversational loop through response critic.",
            "confidence": 0.93,
            "source_turn_id": "turn-archive-1",
            "tags": ["milestone"],
            "evidence": ["Perception through critic is implemented."],
        },
    )
    memory_id = create_response.json()["memory"]["memory_id"]

    archive_response = client.post(
        "/memory/archive",
        json={"memory_id": memory_id},
    )
    assert archive_response.status_code == 200
    assert archive_response.json()["archived"] is True

    hidden_query = client.post(
        "/memory/query",
        json={
            "memory_type": "episodic",
            "query_text": "milestone first conversational loop",
            "limit": 5,
        },
    )
    assert hidden_query.status_code == 200
    assert hidden_query.json()["memories"] == []

    visible_query = client.post(
        "/memory/query",
        json={
            "memory_type": "episodic",
            "query_text": "milestone first conversational loop",
            "include_archived": True,
            "limit": 5,
        },
    )
    assert visible_query.status_code == 200
    assert visible_query.json()["memories"]


def test_memory_retriever_returns_ranked_prompt_ready_memories():
    client.post(
        "/memory/upsert",
        json={
            "memory_type": "preference",
            "key": "answer_style.depth",
            "value": "deep architectural discussion",
            "confidence": 0.91,
            "source_turn_id": "turn-retrieve-1",
            "tags": ["preferences", "style"],
            "evidence": ["User asked to go deeper several times."],
        },
    )
    client.post(
        "/memory/upsert",
        json={
            "memory_type": "procedural",
            "key": "planning.sequence",
            "value": "build memory store before memory retriever",
            "confidence": 0.82,
            "source_turn_id": "turn-retrieve-2",
            "tags": ["planning", "memory"],
            "evidence": ["Module order was agreed in the conversation."],
        },
    )

    response = client.post(
        "/memory/retrieve",
        json={
            "query_text": "deep architectural discussion",
            "memory_types": ["preference", "procedural"],
            "tags": ["preferences"],
            "limit": 3,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["memories"]
    assert payload["memories"][0]["memory_type"] == "preference"
    assert payload["memories"][0]["retrieval_score"] >= 0.0
    assert payload["evaluation"]["used_type_filters"] is True


def test_memory_retriever_respects_archived_filtering():
    create_response = client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "retrieval.hidden.case",
            "value": "This archived memory should not appear by default.",
            "confidence": 0.77,
            "source_turn_id": "turn-retrieve-archive",
            "tags": ["hidden"],
            "evidence": ["Archive filtering test."],
        },
    )
    memory_id = create_response.json()["memory"]["memory_id"]
    client.post("/memory/archive", json={"memory_id": memory_id})

    hidden = client.post(
        "/memory/retrieve",
        json={
            "query_text": "archived memory",
            "memory_types": ["episodic"],
            "limit": 5,
        },
    )
    assert hidden.status_code == 200
    assert hidden.json()["memories"] == []

    visible = client.post(
        "/memory/retrieve",
        json={
            "query_text": "archived memory",
            "memory_types": ["episodic"],
            "include_archived": True,
            "limit": 5,
        },
    )
    assert visible.status_code == 200
    assert visible.json()["memories"]


def test_memory_committer_persists_explicit_preference_and_commitment():
    turn = make_turn(
        turn_id="turn-commit-1",
        message_text="Let's go with deeper technical explanations from now on.",
        recent_context={
            "recent_turn_summaries": ["We completed the first conversational loop."],
            "active_topic": "response style",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "summarized the current architecture",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    response = client.post(
        "/memory/commit",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "critic_review": None,
            "persist_committed": True,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["candidate_count"] >= 1
    assert payload["evaluation"]["committed_count"] >= 1
    assert payload["committed_memories"]

    retrieved = client.post(
        "/memory/retrieve",
        json={
            "query_text": "deeper technical explanations",
            "memory_types": ["preference", "procedural"],
            "limit": 5,
        },
    )
    assert retrieved.status_code == 200
    assert retrieved.json()["memories"]


def test_memory_committer_persists_successful_response_policy():
    turn = make_turn(
        turn_id="turn-commit-policy-1",
        message_text="how are you",
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    response = client.post(
        "/memory/commit",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
            "critic_review": {
                "passed": True,
                "scores": {
                    "relevance": 0.8,
                    "clarity": 0.8,
                    "faithfulness_to_plan": 0.8,
                    "tone_fit": 0.8,
                    "hallucination_risk": 0.1,
                },
                "findings": [],
                "recommended_edits": [],
                "debug_signals": {},
            },
            "persist_committed": True,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["committed_count"] >= 1
    assert any("response_policy." in memory["key"] for memory in payload["committed_memories"])


def test_dialogue_planner_reuses_learned_response_policy():
    store = MemoryStore()
    retriever = MemoryRetriever(store)
    planner = DialoguePlanner(memory_retriever=retriever)

    store.upsert(
        MemoryUpsertRequest(
            memory_type="procedural",
            key="response_policy.social_exchange",
            value='{"interaction_type":"social_exchange","reasoning_effort":"low","target_length":"medium","tone_policy":"warm","retrieval_policy":"minimal_retrieval","example_policy":"examples_optional","confidence_policy":"answer_directly"}',
            confidence=0.91,
            source_turn_id="memory-seed-1",
            tags=["procedural", "response_policy", "social_exchange"],
            evidence=["Successful prior social exchange."],
        )
    )

    turn = TurnInput(
        turn_id="turn-policy-reuse-1",
        session_id="session-policy-reuse-1",
        user_id="user-policy-reuse-1",
        timestamp=datetime(2026, 5, 19, 11, 0, 0),
        message_text="how are you",
    )
    perception_response = client.post("/perception/analyze", json=turn.model_dump(mode="json"))
    assert perception_response.status_code == 200
    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn.model_dump(mode="json"),
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    plan_response = planner.create_plan(
        DialoguePlanRequest(
            turn_input=turn,
            perception_state=perception_response.json(),
            working_memory_state=working_memory_response.json()["state"],
        )
    )
    assert plan_response.plan.debug_signals["policy_source"] == "memory_reuse"
    assert plan_response.plan.response_policy.reasoning_effort == "low"
    assert plan_response.plan.response_policy.target_length == "medium"


def test_memory_committer_can_dry_run_without_persisting():
    turn = make_turn(
        turn_id="turn-commit-2",
        message_text="That is wrong, correct this assumption.",
        recent_context={
            "recent_turn_summaries": ["The assistant made an incorrect assumption."],
            "active_topic": "assumption correction",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "made a draft recommendation",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    response = client.post(
        "/memory/commit",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "critic_review": None,
            "persist_committed": False,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["candidate_count"] >= 1
    assert payload["evaluation"]["persistence_enabled"] is False
    assert payload["committed_memories"] == []


def test_prompt_assembler_includes_retrieved_long_term_memories():
    client.post(
        "/memory/upsert",
        json={
            "memory_type": "preference",
            "key": "preference.response.style",
            "value": "The user prefers deeper technical explanations with architecture detail.",
            "confidence": 0.94,
            "source_turn_id": "turn-prompt-memory-1",
            "tags": ["preference", "style"],
            "evidence": ["Stored preference for response depth."],
        },
    )

    turn = make_turn(
        turn_id="turn-prompt-memory-2",
        message_text="Can you explain the architecture in more technical depth?",
        recent_context={
            "recent_turn_summaries": ["We implemented the memory committer."],
            "active_topic": "architecture depth",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "implemented learning logic",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["includes_retrieved_memories"] is True
    assert payload["prompt"]["retrieved_memories"]
    assert "RETRIEVED LONG-TERM MEMORIES" in payload["prompt"]["rendered_prompt"]


def test_prompt_assembler_still_works_without_matching_memories():
    turn = make_turn(
        turn_id="turn-prompt-memory-3",
        message_text="What is the next clean module to build?",
        recent_context={
            "recent_turn_summaries": ["The current architecture is stable."],
            "active_topic": "next module",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "summarized progress",
        },
    )
    perception_response = client.post("/perception/analyze", json=turn)
    assert perception_response.status_code == 200

    working_memory_response = client.post(
        "/working-memory/update",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "current_state": None,
        },
    )
    assert working_memory_response.status_code == 200

    planner_response = client.post(
        "/dialogue-planner/plan",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
        },
    )
    assert planner_response.status_code == 200

    response = client.post(
        "/prompt-assembler/assemble",
        json={
            "turn_input": turn,
            "perception_state": perception_response.json(),
            "working_memory_state": working_memory_response.json()["state"],
            "dialogue_plan": planner_response.json()["plan"],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert "RETRIEVED LONG-TERM MEMORIES" in payload["prompt"]["rendered_prompt"]


def test_memory_decay_archives_weak_stale_memories():
    create_response = client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "decay.weak.case",
            "value": "Low-value stale memory.",
            "confidence": 0.05,
            "source_turn_id": "turn-decay-1",
            "tags": ["decay"],
            "evidence": ["Weak memory for decay test."],
            "metadata": {},
        },
    )
    memory_id = create_response.json()["memory"]["memory_id"]

    query_response = client.post(
        "/memory/query",
        json={
            "memory_type": "episodic",
            "query_text": "Low-value stale memory",
            "limit": 1,
        },
    )
    assert query_response.status_code == 200

    decay_response = client.post(
        "/memory/decay",
        json={
            "threshold": 0.4,
            "max_idle_days": 0,
        },
    )
    payload = decay_response.json()
    assert decay_response.status_code == 200
    matching = [result for result in payload["results"] if result["memory_id"] == memory_id]
    assert matching
    assert matching[0]["archived"] is True


def test_memory_decay_keeps_reinforced_strong_memories_active():
    first = client.post(
        "/memory/upsert",
        json={
            "memory_type": "preference",
            "key": "decay.strong.case",
            "value": "User prefers detailed architectural answers.",
            "confidence": 0.9,
            "source_turn_id": "turn-decay-2",
            "tags": ["preference", "decay"],
            "evidence": ["Strong durable preference."],
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/memory/upsert",
        json={
            "memory_type": "preference",
            "key": "decay.strong.case",
            "value": "User prefers detailed architectural answers.",
            "confidence": 0.92,
            "source_turn_id": "turn-decay-3",
            "tags": ["preference", "decay"],
            "evidence": ["Repeated preference signal."],
        },
    )
    memory_id = second.json()["memory"]["memory_id"]

    decay_response = client.post(
        "/memory/decay",
        json={
            "threshold": 0.4,
            "max_idle_days": 365,
        },
    )
    payload = decay_response.json()
    assert decay_response.status_code == 200
    matching = [result for result in payload["results"] if result["memory_id"] == memory_id]
    assert matching
    assert matching[0]["archived"] is False


def test_memory_consolidation_creates_semantic_summary_from_episodic_group():
    client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "episode.working.memory.turn1",
            "value": "ask_explanation: explain the working memory manager",
            "confidence": 0.74,
            "source_turn_id": "turn-consolidate-1",
            "tags": ["episodic", "working-memory"],
            "evidence": ["First successful working memory explanation."],
        },
    )
    client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "episode.working.memory.turn2",
            "value": "ask_explanation: clarify the working memory state shape",
            "confidence": 0.79,
            "source_turn_id": "turn-consolidate-2",
            "tags": ["episodic", "working-memory"],
            "evidence": ["Second successful working memory explanation."],
        },
    )

    response = client.post(
        "/memory/consolidate",
        json={
            "memory_type": "episodic",
            "tags": ["working-memory"],
            "min_group_size": 2,
            "persist_consolidated": True,
            "archive_sources": False,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["evaluation"]["candidate_count"] >= 1
    assert payload["evaluation"]["consolidated_count"] >= 1
    assert any(memory["key"].startswith("consolidated.episode.working.memory") for memory in payload["consolidated_memories"])


def test_memory_consolidation_can_archive_source_memories():
    first = client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "episode.consolidate.archive.turn1",
            "value": "First source memory.",
            "confidence": 0.7,
            "source_turn_id": "turn-consolidate-archive-1",
            "tags": ["archive-consolidation"],
            "evidence": ["First archival consolidation source."],
        },
    ).json()["memory"]["memory_id"]
    second = client.post(
        "/memory/upsert",
        json={
            "memory_type": "episodic",
            "key": "episode.consolidate.archive.turn2",
            "value": "Second source memory.",
            "confidence": 0.72,
            "source_turn_id": "turn-consolidate-archive-2",
            "tags": ["archive-consolidation"],
            "evidence": ["Second archival consolidation source."],
        },
    ).json()["memory"]["memory_id"]

    response = client.post(
        "/memory/consolidate",
        json={
            "memory_type": "episodic",
            "tags": ["archive-consolidation"],
            "min_group_size": 2,
            "persist_consolidated": True,
            "archive_sources": True,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert first in payload["archived_source_ids"]
    assert second in payload["archived_source_ids"]

    archived_query = client.post(
        "/memory/query",
        json={
            "memory_type": "episodic",
            "query_text": "source",
            "tags": ["archive-consolidation"],
            "include_archived": True,
            "limit": 10,
        },
    )
    archived_payload = archived_query.json()
    archived_ids = {memory["memory_id"] for memory in archived_payload["memories"] if memory["archived"]}
    assert first in archived_ids
    assert second in archived_ids


def test_agent_orchestrator_runs_full_pipeline():
    turn = make_turn(
        turn_id="turn-orchestrator-1",
        message_text="Let's go with deeper technical explanations and explain the orchestrator.",
        recent_context={
            "recent_turn_summaries": ["The long-term memory loop is now working."],
            "active_topic": "orchestrator design",
            "unresolved_questions": ["How do all modules run together?"],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "finished memory decay",
        },
    )

    response = client.post(
        "/agent/run",
        json={
            "turn_input": turn,
            "current_working_memory_state": None,
            "commit_memory": True,
            "persist_committed_memory": True,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["perception"]["primary_intent"]
    assert payload["attention_gate"]["state"]["primary_focus"]
    assert payload["working_memory"]["state"]["active_goal"]
    assert payload["dialogue_plan"]["plan"]["response_mode"]
    assert payload["prompt_assembly"]["prompt"]["rendered_prompt"]
    assert payload["generation"]["output"]["response_text"]
    assert "scores" in payload["critic"]["review"]
    assert payload["memory_commit"] is not None
    assert payload["evaluation"]["used_memory_commit"] is True


def test_agent_orchestrator_supports_dry_run_without_memory_persistence():
    turn = make_turn(
        turn_id="turn-orchestrator-2",
        message_text="That is wrong, correct the orchestrator assumption.",
        recent_context={
            "recent_turn_summaries": ["The orchestrator made a bad assumption."],
            "active_topic": "orchestrator correction",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "returned a draft answer",
        },
    )

    response = client.post(
        "/agent/run",
        json={
            "turn_input": turn,
            "current_working_memory_state": None,
            "commit_memory": True,
            "persist_committed_memory": False,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["memory_commit"] is not None
    assert payload["memory_commit"]["committed_memories"] == []
    assert payload["evaluation"]["used_memory_commit"] is True


def test_session_state_persists_across_agent_runs():
    session_id = "session-orchestrator-state"
    first_turn = make_turn(
        turn_id="turn-session-1",
        session_id=session_id,
        message_text="Let's go with a technical collaboration mode.",
    )
    first_response = client.post(
        "/agent/run",
        json={
            "turn_input": first_turn,
            "commit_memory": False,
            "persist_committed_memory": False,
            "use_session_state": True,
        },
    )
    assert first_response.status_code == 200

    session_lookup = client.get(f"/session/{session_id}")
    assert session_lookup.status_code == 200
    assert session_lookup.json()["found"] is True

    second_turn = make_turn(
        turn_id="turn-session-2",
        session_id=session_id,
        message_text="Make it simpler.",
        recent_context={
            "recent_turn_summaries": ["The assistant explained the collaboration mode."],
            "active_topic": "technical collaboration mode",
            "unresolved_questions": [],
            "conversation_mode": "technical_collaboration",
            "last_assistant_action": "answered directly",
        },
    )
    second_response = client.post(
        "/agent/run",
        json={
            "turn_input": second_turn,
            "commit_memory": False,
            "persist_committed_memory": False,
            "use_session_state": True,
        },
    )
    payload = second_response.json()
    assert second_response.status_code == 200
    assert payload["evaluation"]["used_session_state"] is True
    assert payload["working_memory"]["state"]["recent_turns_compact"]


def test_config_and_health_endpoints_return_runtime_information():
    config_response = client.get("/config")
    assert config_response.status_code == 200
    config_payload = config_response.json()
    assert config_payload["llm_provider"] in {"mock", "ollama"}
    assert config_payload["llm_model"]

    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["status"] in {"ok", "degraded"}
    assert "llm_provider" in health_payload
