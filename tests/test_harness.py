from datetime import datetime

from perception_service.harness import evaluate_fixtures
from perception_service.models import FixtureExpectation, LabeledFixture, TurnInput
from perception_service.taxonomy import IntentLabel


def make_input(message_text: str, **overrides) -> TurnInput:
    data = {
        "turn_id": overrides.pop("turn_id", "turn-fixture"),
        "session_id": overrides.pop("session_id", "session-fixture"),
        "user_id": overrides.pop("user_id", "user-fixture"),
        "timestamp": overrides.pop("timestamp", datetime(2026, 5, 19, 10, 0, 0)),
        "message_text": message_text,
    }
    data.update(overrides)
    return TurnInput(**data)


def test_fixture_harness_compares_expected_outputs():
    fixtures = [
        LabeledFixture(
            name="correction-turn",
            turn_input=make_input("That is wrong, correct this please."),
            expected=FixtureExpectation(
                primary_intent=IntentLabel.CORRECTION,
                minimum_intent_confidence=0.8,
                salience_flags={"contains_correction": True},
            ),
        ),
        LabeledFixture(
            name="preference-turn",
            turn_input=make_input("Let's go with code-first hooks."),
            expected=FixtureExpectation(
                primary_intent=IntentLabel.PREFERENCE_STATEMENT,
                salience_flags={"contains_preference": True, "contains_commitment": True},
            ),
        ),
    ]

    results = evaluate_fixtures(fixtures)

    assert all(result.passed for result in results), [result.mismatches for result in results]
