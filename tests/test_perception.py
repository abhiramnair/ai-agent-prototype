from datetime import datetime

from fastapi.testclient import TestClient
import pytest

from app import app
from perception_service.models import RecentContext, TurnInput


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
    assert payload["output"]["metadata"]["provider_name"] == "mock"
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
