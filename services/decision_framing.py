"""Decision detection and framing (Phase 3A)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from services.decision_framing_schemas import (
    DecisionFrame,
    DecisionFramingMetrics,
    DecisionFramingResult,
)
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

_FRAMING_SYSTEM_PROMPT = """You are a decision-framing analyst for a research system.

Your job is ONLY to determine whether the user is asking for a DECISION and, if so, extract a structured DecisionFrame.

DECISION-ORIENTED queries ask for a choice, recommendation context, or whether to take an action.
Examples: "Should we acquire Company X?", "Which CRM should we choose?", "Should we use OpenAI or Anthropic?"

NOT decision-oriented (set decision_oriented=false, decision_frame=null):
- Pure research or comparison without a choice: "Compare OpenAI and Anthropic."
- Exploratory information: "Tell me about entering the Japanese market."
- Factual lookups.

RULES FOR decision_frame WHEN decision_oriented=true:

1. decision — state what must be decided in plain language.

2. options — each option has label and origin:
   - origin="explicit" ONLY when the user named that alternative in the query (preserve exact wording).
   - origin="implied" ONLY for minimal binary defaults clearly required by the decision (e.g. "acquire Company X" → "Do not acquire Company X" as implied).
   - Do NOT invent extra strategic alternatives or third vendors.

3. criteria — each criterion has label and origin:
   - origin="explicit" ONLY when the user stated that dimension matters (e.g. "cost is our most important consideration" → Cost).
   - origin="inferred" for reasonable high-level evaluation dimensions the user did not name (e.g. risk, strategic fit).
   - Do NOT score, rank, or weight criteria.

4. constraints — list of strings, EXPLICIT USER REQUIREMENTS ONLY.
   - Include only hard limits the user stated (budget caps, must integrate with X, geography).
   - Do NOT infer constraints from model judgment.

5. time_horizon — the period over which the user intends to make or evaluate the decision.
   - Populate ONLY for explicit decision horizons (e.g. "over the next three years", "for the next 18 months", "before Q4", "this year" when it clearly scopes the decision).
   - Otherwise null.
   - Do NOT use metric or pricing cadence as time_horizon (e.g. "$20,000 per year", "per month", "annual subscription", "per-user pricing" belong in constraints/metrics, NOT time_horizon).

6. missing_decision_context — important unspecified DECISION inputs (price range, strategic objective, capacity).
   - Do NOT invent values.
   - These are NOT factual research unknowns.

7. explicit_assumptions — ONLY assumptions the user explicitly stated (e.g. "assuming rates stay above 4%").
   - Do NOT generate speculative assumptions.

8. decision_type — metadata only: market_entry, vendor_selection, acquisition, investment, buy_vs_not_buy, hiring, prioritization, product_strategy, or other.

9. Do NOT recommend, score options, or add pros/cons.

10. When uncertain whether the query is a decision, set decision_oriented=false."""


def _clean_strings(values: list[str]) -> list[str]:
    return [v.strip() for v in values if v and v.strip()]


_METRIC_CADENCE_HORIZON = re.compile(
    r"^(?:per\s+(?:year|month|week|day|user|seat)s?|"
    r"(?:annual|monthly|yearly)(?:\s+(?:recurring|subscription))?)$",
    re.IGNORECASE,
)


def _sanitize_time_horizon(value: str | None) -> str | None:
    """Drop metric/pricing cadence mistakenly placed in time_horizon."""
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if _METRIC_CADENCE_HORIZON.fullmatch(stripped):
        return None
    return stripped


def _validate_frame(result: DecisionFramingResult) -> DecisionFrame | None:
    """Fail open on invalid frames."""
    if not result.decision_oriented:
        return None
    if result.decision_frame is None:
        return None
    frame = result.decision_frame
    if not frame.decision or not frame.decision.strip():
        return None

    options = [opt for opt in frame.options if opt.label.strip()]
    criteria = [crit for crit in frame.criteria if crit.label.strip()]

    return DecisionFrame(
        decision=frame.decision.strip(),
        decision_type=frame.decision_type,
        options=options,
        criteria=criteria,
        constraints=_clean_strings(frame.constraints),
        time_horizon=_sanitize_time_horizon(frame.time_horizon),
        missing_decision_context=_clean_strings(frame.missing_decision_context),
        explicit_assumptions=_clean_strings(frame.explicit_assumptions),
    )


def _metrics_from_frame(
    frame: DecisionFrame | None,
    *,
    framing_failed: bool = False,
    failure_reason: str | None = None,
    framing_llm_calls: int = 0,
    framing_time_ms: float = 0.0,
) -> DecisionFramingMetrics:
    metrics = DecisionFramingMetrics(
        decision_detected=frame is not None,
        framing_llm_calls=framing_llm_calls,
        framing_time_ms=framing_time_ms,
        framing_failed=framing_failed,
        failure_reason=failure_reason,
    )
    if frame is None:
        return metrics

    metrics.decision_type = frame.decision_type.value
    metrics.option_count = len(frame.options)
    metrics.explicit_option_count = sum(1 for o in frame.options if o.origin == "explicit")
    metrics.implied_option_count = sum(1 for o in frame.options if o.origin == "implied")
    metrics.criteria_count = len(frame.criteria)
    metrics.explicit_criterion_count = sum(1 for c in frame.criteria if c.origin == "explicit")
    metrics.inferred_criterion_count = sum(1 for c in frame.criteria if c.origin == "inferred")
    metrics.constraint_count = len(frame.constraints)
    metrics.missing_context_count = len(frame.missing_decision_context)
    metrics.explicit_assumption_count = len(frame.explicit_assumptions)
    return metrics


async def frame_decision_query(
    query: str,
    *,
    llm: Any | None = None,
) -> tuple[DecisionFrame | None, DecisionFramingMetrics]:
    """
    Detect decision orientation and extract DecisionFrame in one structured LLM call.

    Fails open: returns (None, metrics) on any error or non-decision query.
    """
    start = time.monotonic()
    if not query or not query.strip():
        return None, _metrics_from_frame(
            None,
            framing_failed=True,
            failure_reason="empty_query",
            framing_time_ms=(time.monotonic() - start) * 1000,
        )

    if llm is None:
        from langchain_anthropic import ChatAnthropic

        from config import settings

        llm = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        )

    elapsed_ms = 0.0
    try:
        with trace_llm_call("decision_framer", "detect_and_frame") as span:
            span.set_input({"query": query[:500]})
            structured = llm.with_structured_output(DecisionFramingResult)
            result: DecisionFramingResult = await structured.ainvoke(
                [
                    {"role": "system", "content": _FRAMING_SYSTEM_PROMPT},
                    {"role": "user", "content": query.strip()},
                ]
            )
            span.set_output(result.model_dump(mode="json"))
        elapsed_ms = (time.monotonic() - start) * 1000
        frame = _validate_frame(result)
        if result.decision_oriented and frame is None:
            return None, _metrics_from_frame(
                None,
                framing_failed=True,
                failure_reason="invalid_frame",
                framing_llm_calls=1,
                framing_time_ms=elapsed_ms,
            )
        return frame, _metrics_from_frame(
            frame,
            framing_llm_calls=1,
            framing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning("Decision framing failed open: %s", exc)
        return None, _metrics_from_frame(
            None,
            framing_failed=True,
            failure_reason=str(exc),
            framing_llm_calls=1,
            framing_time_ms=elapsed_ms,
        )
