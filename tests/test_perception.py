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
