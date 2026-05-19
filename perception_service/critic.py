from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from .models import (
    CriticEvaluation,
    CriticFinding,
    CriticRequest,
    CriticResponse,
    CriticReview,
    CriticScores,
)

logger = logging.getLogger("critic_service")


class CriticHook(ABC):
    @abstractmethod
    def review(self, request: CriticRequest) -> CriticReview:
        raise NotImplementedError


class DefaultCriticHook(CriticHook):
    def review(self, request: CriticRequest) -> CriticReview:
        response_text = request.generation_output.response_text
        response_lower = response_text.lower()
        findings: list[CriticFinding] = []
        recommended_edits: list[str] = []

        must_include = [str(item).lower() for item in request.prompt.response_plan.get("must_include", [])]
        must_avoid = [str(item).lower() for item in request.prompt.response_plan.get("must_avoid", [])]
        uncertainty_required = bool(
            request.prompt.response_plan.get("draft_constraints", {}).get("require_explicit_uncertainty", False)
        )
        expected_tone = str(request.prompt.response_plan.get("tone", "clear_direct"))

        matched_required = sum(1 for item in must_include if item and item in response_lower)
        violated_avoid = [item for item in must_avoid if item and item in response_lower]

        relevance = min(
            1.0,
            0.45 + 0.18 * matched_required + (0.12 if request.prompt.current_subgoal.lower() in response_lower else 0.0),
        )
        faithfulness = min(1.0, 0.4 + 0.2 * matched_required)
        clarity = 0.88 if len(response_text.split()) >= 8 else 0.62
        tone_fit = 0.9 if expected_tone in {"clear_direct", "collaborative_technical", "warm_clear"} else 0.76
        hallucination_risk = 0.12

        if matched_required < min(2, len(must_include)) and must_include:
            findings.append(
                CriticFinding(
                    severity="medium",
                    category="coverage",
                    message="The draft does not clearly cover enough required plan elements.",
                )
            )
            recommended_edits.append("Incorporate more of the required plan elements explicitly.")
            relevance = max(0.0, relevance - 0.22)
            faithfulness = max(0.0, faithfulness - 0.25)

        if violated_avoid:
            findings.append(
                CriticFinding(
                    severity="high",
                    category="constraint_violation",
                    message=f"The draft appears to include avoided content: {', '.join(violated_avoid[:3])}.",
                )
            )
            recommended_edits.append("Remove content that conflicts with the avoid-list constraints.")
            faithfulness = max(0.0, faithfulness - 0.18)
            hallucination_risk = min(1.0, hallucination_risk + 0.18)

        if uncertainty_required and not any(term in response_lower for term in ["assumption", "ambiguous", "uncertain"]):
            findings.append(
                CriticFinding(
                    severity="medium",
                    category="uncertainty",
                    message="The draft should acknowledge uncertainty but does not do so explicitly.",
                )
            )
            recommended_edits.append("Add an explicit uncertainty or assumption statement.")
            faithfulness = max(0.0, faithfulness - 0.12)
            clarity = max(0.0, clarity - 0.08)

        if len(response_text.split()) < 6:
            findings.append(
                CriticFinding(
                    severity="low",
                    category="clarity",
                    message="The draft may be too short to fully satisfy the plan.",
                )
            )
            recommended_edits.append("Expand the draft to better explain the intended response.")
            clarity = max(0.0, clarity - 0.12)

        passed = not any(finding.severity == "high" for finding in findings) and faithfulness >= 0.5 and relevance >= 0.5
        return CriticReview(
            passed=passed,
            scores=CriticScores(
                relevance=round(relevance, 2),
                clarity=round(clarity, 2),
                faithfulness_to_plan=round(faithfulness, 2),
                tone_fit=round(tone_fit, 2),
                hallucination_risk=round(hallucination_risk, 2),
            ),
            findings=findings,
            recommended_edits=recommended_edits,
            debug_signals={
                "matched_required_count": matched_required,
                "required_count": len(must_include),
                "violated_avoid_count": len(violated_avoid),
            },
        )


class ResponseCritic:
    def __init__(self, hook: CriticHook | None = None) -> None:
        self.hook = hook or DefaultCriticHook()

    def review(self, request: CriticRequest) -> CriticResponse:
        review = self.hook.review(request)
        score_summary = (
            review.scores.relevance
            + review.scores.clarity
            + review.scores.faithfulness_to_plan
            + review.scores.tone_fit
            + (1.0 - review.scores.hallucination_risk)
        ) / 5.0
        evaluation = CriticEvaluation(
            has_findings=bool(review.findings),
            requires_revision=not review.passed,
            score_summary=round(score_summary, 2),
        )
        logger.info(
            "critic_review_completed",
            extra={
                "review": review.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
        )
        return CriticResponse(review=review, evaluation=evaluation)
