"""Early evidence sufficiency for SIMPLE_FACT fast path."""

import re
from dataclasses import dataclass

from domain.models import Evidence, Source
from services.fact_target import AnswerTarget, entity_match_tokens
from services.fact_value import (
    StructuredFactValue,
    detect_value_conflicts,
    extract_fact_value,
    validate_fact_value_in_evidence,
)
from services.source_authority import is_source_adequate_for_domain


@dataclass(frozen=True)
class FactSufficiencyResult:
    """Whether decisive evidence is sufficient to answer a factual target."""

    is_sufficient: bool
    reason: str
    decisive_evidence_id: int | None = None
    corroboration_count: int = 0
    fact_value: StructuredFactValue | None = None


def _evidence_addresses_target(evidence_text: str, target: AnswerTarget) -> bool:
    text_lower = evidence_text.lower()

    attr_patterns: dict[str, list[str]] = {
        "capital": ["capital"],
        "revenue": ["revenue", "sales", "net sales", "billion", "million"],
        "winner": [
            "won", "champion", "title", "victory", "secured", "clinched",
            "first", "secures",
        ],
        "ceo": ["ceo", "chief executive"],
        "date": ["released", "founded", "established", "born", "launched"],
        "identity": [],
        "eps": ["eps", "earnings per share"],
        "net income": ["net income", "profit"],
    }
    patterns = attr_patterns.get(target.attribute, [])
    attr_match = any(p in text_lower for p in patterns) if patterns else True

    entity_match = entity_match_tokens(target.entity, evidence_text)

    if target.temporal_scope:
        if target.temporal_scope not in evidence_text:
            if not any(
                y in evidence_text
                for y in __import__("re").findall(
                    r"\b(19|20)\d{2}\b", target.temporal_scope
                )
            ):
                return False

    return entity_match and attr_match


def check_fact_sufficiency(
    target: AnswerTarget,
    evidence: Evidence,
    source: Source,
    *,
    existing_evidence: list[Evidence] | None = None,
) -> FactSufficiencyResult:
    """Check if evidence decisively answers the target with extractable value."""
    fact_value = extract_fact_value(evidence.exact_text, target)

    if fact_value is None and not _evidence_addresses_target(
        evidence.exact_text, target
    ):
        return FactSufficiencyResult(
            False, "Evidence does not directly address the answer target"
        )

    if not is_source_adequate_for_domain(source, target.domain):
        return FactSufficiencyResult(
            False,
            f"Source inadequate for {target.domain.value} ({source.url})",
        )

    if fact_value is None:
        fact_value = extract_fact_value(evidence.exact_text, target)
    if fact_value is None:
        return FactSufficiencyResult(
            False, "Could not extract structured target value from evidence"
        )

    valid, reason = validate_fact_value_in_evidence(fact_value, evidence.exact_text)
    if not valid:
        return FactSufficiencyResult(False, reason)

    if target.temporal_scope:
        scope = target.temporal_scope
        in_evidence = scope in evidence.exact_text or re.search(
            r"\b" + re.escape(scope), evidence.exact_text
        )
        in_source = scope in (source.title or "") or scope in source.url
        if not in_evidence and not in_source:
            return FactSufficiencyResult(
                False, f"Temporal scope '{scope}' not matched in evidence or source"
            )

    if existing_evidence:
        prior_values = []
        for ev in existing_evidence:
            if ev.id != evidence.id:
                fv = extract_fact_value(ev.exact_text, target)
                if fv:
                    prior_values.append(fv)
        if prior_values:
            conflict = detect_value_conflicts(prior_values + [fact_value], target)
            if conflict:
                return FactSufficiencyResult(False, conflict)

    corroboration = len(existing_evidence or []) if existing_evidence else 0

    return FactSufficiencyResult(
        True,
        "Decisive evidence with validated target value",
        decisive_evidence_id=evidence.id,
        corroboration_count=corroboration,
        fact_value=fact_value,
    )


def detect_conflicting_values(
    evidence_list: list[Evidence], target: AnswerTarget
) -> str | None:
    """Detect conflicting target values across evidence items."""
    values: list[StructuredFactValue] = []
    for ev in evidence_list:
        fv = extract_fact_value(ev.exact_text, target)
        if fv:
            valid, _ = validate_fact_value_in_evidence(fv, ev.exact_text)
            if valid:
                values.append(fv)
    return detect_value_conflicts(values, target)
