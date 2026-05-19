from __future__ import annotations

from .models import FixtureResult, LabeledFixture
from .pipeline import PerceptionPipeline


def evaluate_fixtures(
    fixtures: list[LabeledFixture],
    pipeline: PerceptionPipeline | None = None,
) -> list[FixtureResult]:
    analyzer = pipeline or PerceptionPipeline()
    results: list[FixtureResult] = []
    for fixture in fixtures:
        actual = analyzer.analyze(fixture.turn_input)
        mismatches: list[str] = []
        expected = fixture.expected

        if expected.primary_intent and actual.primary_intent != expected.primary_intent:
            mismatches.append(
                f"primary_intent expected {expected.primary_intent} but got {actual.primary_intent}"
            )
        if (
            expected.references_prior_context is not None
            and actual.references_prior_context != expected.references_prior_context
        ):
            mismatches.append(
                "references_prior_context expected "
                f"{expected.references_prior_context} but got {actual.references_prior_context}"
            )
        if (
            expected.minimum_intent_confidence is not None
            and actual.intent_confidence < expected.minimum_intent_confidence
        ):
            mismatches.append(
                f"intent_confidence expected >= {expected.minimum_intent_confidence} "
                f"but got {actual.intent_confidence}"
            )
        if (
            expected.maximum_ambiguity_score is not None
            and actual.ambiguity_score > expected.maximum_ambiguity_score
        ):
            mismatches.append(
                f"ambiguity_score expected <= {expected.maximum_ambiguity_score} "
                f"but got {actual.ambiguity_score}"
            )
        for flag, expected_value in expected.salience_flags.items():
            actual_value = getattr(actual.salience_signals, flag)
            if actual_value != expected_value:
                mismatches.append(f"salience {flag} expected {expected_value} but got {actual_value}")

        results.append(
            FixtureResult(
                name=fixture.name,
                passed=not mismatches,
                actual=actual,
                mismatches=mismatches,
            )
        )
    return results
