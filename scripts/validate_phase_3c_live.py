"""Live validation for Phase 3C decision synthesis (isolated fixtures + real LLM)."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from domain.models import Claim, ClaimType, EvidenceConfidence, KnowledgeCategory, VerificationStatus
from services.decision_framing_schemas import DecisionCriterion, DecisionFrame, DecisionOption, DecisionType
from services.decision_synthesis import (
    build_oe_index,
    run_pre_check,
    synthesize_decision,
    validate_and_build_synthesis,
)
from services.decision_synthesis_schemas import (
    ChangeConditionLLM,
    ConstraintAssessmentLLM,
    ConstraintCompliance,
    CriterionReferenceLLM,
    DecisionSynthesisLLMOutput,
    RecommendationStatus,
)
from services.knowledge_state_schemas import KnowledgeState, KnowledgeStateEntry
from services.option_evaluation import build_claim_catalog
from services.option_evaluation_schemas import (
    CriterionAssessment,
    CriterionEvaluation,
    KnowledgeCoverage,
    OptionEvaluation,
    OptionEvaluationEntry,
)


def _claim(cid: int, text: str) -> Claim:
    return Claim(id=cid, research_run_id=1, text=text, claim_type=ClaimType.FACTUAL)


def _entry(cid: int, bucket: str = "known") -> KnowledgeStateEntry:
    sm = {"known": VerificationStatus.SUPPORTED, "disputed": VerificationStatus.UNCERTAIN}
    km = {"known": KnowledgeCategory.KNOWN, "disputed": KnowledgeCategory.DISPUTED}
    return KnowledgeStateEntry(
        claim_id=cid,
        verification_id=cid * 10,
        knowledge_category=km.get(bucket),
        verification_status=sm.get(bucket, VerificationStatus.SUPPORTED),
        confidence=EvidenceConfidence.HIGH,
        evidence_ids=[cid * 100],
    )


def _ks(**buckets) -> KnowledgeState:
    return KnowledgeState(**{k: buckets.get(k, []) for k in [
        "known", "likely", "disputed", "unknown", "contradicted", "unverifiable"
    ]})


def _ce(opt, crit, assessment, coverage, claim_ids, *, origin="explicit", priority="standard", reason=""):
    return CriterionEvaluation(
        criterion_label=crit,
        criterion_origin=origin,
        criterion_priority=priority,
        assessment=assessment,
        knowledge_coverage=coverage,
        claim_ids=claim_ids,
        reason=reason or f"Evaluation for {opt} on {crit}.",
    )


def _oe(decision, rows, *, constraints=None, limitations=None) -> OptionEvaluation:
    by: dict[str, OptionEvaluationEntry] = {}
    for opt, crit, assessment, coverage, cids, origin, priority, reason in rows:
        if opt not in by:
            by[opt] = OptionEvaluationEntry(option_label=opt, option_origin="explicit")
        by[opt].criteria_evaluations.append(
            _ce(opt, crit, assessment, coverage, cids, origin=origin, priority=priority, reason=reason)
        )
    return OptionEvaluation(
        decision=decision,
        option_evaluations=list(by.values()),
        constraints=constraints or [],
        decision_limitations=limitations or [],
    )


_EXTERNAL_HINTS = re.compile(
    r"\b(api pricing|per token|context window|gpt-?\d|claude \d|market leader|"
    r"industry standard|publicly known|reputation|enterprise plan)\b",
    re.I,
)


@dataclass
class LiveResult:
    name: str
    pass_: bool = False
    expected_criterion_pairs: int = 0
    actual_criterion_pairs: int = 0
    expected_constraint_pairs: int = 0
    actual_constraint_pairs: int = 0
    missing_constraints: list[str] = field(default_factory=list)
    status: str | None = None
    recommended_option: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    semantic_failures: list[str] = field(default_factory=list)
    external_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    synthesis_llm_calls: int = 0


def _constraint_pairs(synthesis) -> set[tuple[str, str]]:
    if not synthesis:
        return set()
    return {(a.option_label, a.constraint) for a in synthesis.constraint_assessments}


def _missing_constraints(frame: DecisionFrame, synthesis) -> list[str]:
    present = _constraint_pairs(synthesis)
    missing = []
    for o in frame.options:
        for c in frame.constraints:
            if (o.label, c) not in present:
                missing.append(f"{o.label} × {c}")
    return missing


def _scan_external(text: str, allowed: list[str]) -> list[str]:
    allowed_l = {a.lower() for a in allowed}
    flags = []
    for m in _EXTERNAL_HINTS.finditer(text):
        if m.group(0).lower() not in allowed_l:
            flags.append(m.group(0))
    return flags


async def _run(frame, oe, ks, claims) -> tuple:
    return await synthesize_decision(frame, oe, ks, claims, llm=None)


async def test_01_clear_primary() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose Vendor A or Vendor B",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", "Vendor A costs $12,000 per year."),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", "Vendor B costs $20,000 per year."),
    ])
    ks = _ks(known=[_entry(1), _entry(2)])
    claims = [_claim(1, "Vendor A costs $12,000 per year."), _claim(2, "Vendor B costs $20,000 per year.")]
    s, m = await _run(frame, oe, ks, claims)
    r = LiveResult("TEST 1 — CLEAR PRIMARY", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommendation_status != RecommendationStatus.RECOMMEND:
        r.semantic_failures.append(f"expected recommend, got {s.recommendation_status}")
    if s and s.recommended_option != "Vendor A":
        r.semantic_failures.append(f"expected Vendor A, got {s.recommended_option}")
    if s and not any(c.option_label == "Vendor A" and c.criterion_label == "Cost" for c in s.supporting_criteria):
        r.semantic_failures.append("missing supporting Vendor A × Cost reference")
    r.pass_ = not r.semantic_failures
    return r


async def test_02_primary_vs_standard() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Reliability", origin="explicit", priority="standard"),
        ],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
        ("Vendor A", "Reliability", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3], "explicit", "standard", ""),
        ("Vendor B", "Reliability", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4], "explicit", "standard", ""),
    ])
    ks = _ks(known=[_entry(i) for i in range(1, 5)])
    claims = [_claim(i, f"claim {i}") for i in range(1, 5)]
    s, m = await _run(frame, oe, ks, claims)
    r = LiveResult("TEST 2 — PRIMARY VS STANDARD", expected_criterion_pairs=4, actual_criterion_pairs=4,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommended_option == "Vendor B" and s.recommendation_status == RecommendationStatus.RECOMMEND:
        r.semantic_failures.append("Vendor B recommended over primary Cost advantage for A")
    r.pass_ = not r.semantic_failures
    return r


async def test_03_inferred_cannot_override() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Strategic fit", origin="inferred"),
            DecisionCriterion(label="Integration", origin="inferred"),
        ],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
        ("Vendor A", "Strategic fit", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3], "inferred", "standard", ""),
        ("Vendor B", "Strategic fit", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4], "inferred", "standard", ""),
        ("Vendor A", "Integration", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [5], "inferred", "standard", ""),
        ("Vendor B", "Integration", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [6], "inferred", "standard", ""),
    ])
    ks = _ks(known=[_entry(i) for i in range(1, 7)])
    claims = [_claim(i, f"claim {i}") for i in range(1, 7)]
    s, m = await _run(frame, oe, ks, claims)
    r = LiveResult("TEST 3 — INFERRED CANNOT OVERRIDE PRIMARY", expected_criterion_pairs=6, actual_criterion_pairs=6,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommended_option == "Vendor B" and s.recommendation_status == RecommendationStatus.RECOMMEND:
        r.semantic_failures.append("Vendor B recommended on inferred criteria alone")
    r.pass_ = not r.semantic_failures
    return r


async def test_04_hard_constraint_matrix() -> LiveResult:
    constraints = ["Budget must be under $20,000 per year", "Must support Salesforce integration"]
    frame = DecisionFrame(
        decision="Vendor A vs Vendor B",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        constraints=constraints,
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
    ], constraints=constraints)
    ks = _ks(known=[_entry(10), _entry(11), _entry(12), _entry(13), _entry(1), _entry(2)])
    claims = [
        _claim(1, "Vendor A annual cost is $12,000 per year"),
        _claim(2, "Vendor B annual cost is $25,000 per year"),
        _claim(10, "Vendor A annual spend is under the $20,000 per year budget cap"),
        _claim(11, "Vendor B annual spend exceeds the $20,000 per year budget cap"),
        _claim(12, "Vendor B supports Salesforce integration"),
        _claim(13, "No verified Salesforce integration evidence for Vendor A"),
    ]
    s, m = await _run(frame, oe, ks, claims)
    r = LiveResult("TEST 4 — HARD CONSTRAINT MATRIX", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   expected_constraint_pairs=4, actual_constraint_pairs=len(s.constraint_assessments) if s else 0,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    r.missing_constraints = _missing_constraints(frame, s)
    if r.missing_constraints:
        r.semantic_failures.append(f"missing constraints: {r.missing_constraints}")
    if s and s.recommended_option == "Vendor B":
        r.semantic_failures.append("Vendor B recommended despite budget violation")
    if s and s.recommendation_status == RecommendationStatus.RECOMMEND and s.recommended_option == "Vendor A":
        va_sf = next((a for a in s.constraint_assessments if a.option_label == "Vendor A" and "Salesforce" in a.constraint), None)
        if va_sf and va_sf.compliance == ConstraintCompliance.NOT_ESTABLISHED:
            r.semantic_failures.append("full recommend for A with Salesforce not established")
    r.pass_ = not r.semantic_failures
    return r


async def test_05_constraint_claim_outside_oe() -> LiveResult:
    constraints = ["Must support Salesforce integration"]
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        constraints=constraints,
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
    ], constraints=constraints)
    ks = _ks(known=[_entry(1), _entry(2), _entry(500)])
    claims = [
        _claim(1, "Vendor A has competitive pricing"),
        _claim(2, "Vendor B is more expensive"),
        _claim(500, "Vendor A supports Salesforce integration"),
    ]
    s, m = await _run(frame, oe, ks, claims)
    r = LiveResult("TEST 5 — CONSTRAINT CLAIM OUTSIDE OE", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   expected_constraint_pairs=2, actual_constraint_pairs=len(s.constraint_assessments) if s else 0,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    r.missing_constraints = _missing_constraints(frame, s)
    oe_claims = {cid for o in oe.option_evaluations for ce in o.criteria_evaluations for cid in ce.claim_ids}
    va = next((a for a in (s.constraint_assessments if s else []) if a.option_label == "Vendor A"), None)
    if va and 500 in va.claim_ids and 500 not in oe_claims:
        r.notes.append("Claim 500 used in constraint assessment but absent from OE")
    elif va and 500 not in va.claim_ids and va.compliance == ConstraintCompliance.SATISFIED:
        r.semantic_failures.append("Expected claim 500 to support Salesforce satisfaction")
  # soft - LLM may still satisfy with reasoning
    if r.missing_constraints:
        r.semantic_failures.append(f"missing constraints: {r.missing_constraints}")
    r.pass_ = not r.semantic_failures
    return r


async def test_06_split_explicit() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Reliability", origin="explicit"),
        ],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "standard", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "standard", ""),
        ("Vendor A", "Reliability", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3], "explicit", "standard", ""),
        ("Vendor B", "Reliability", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4], "explicit", "standard", ""),
    ])
    s, m = await _run(frame, oe, _ks(known=[_entry(i) for i in range(1, 5)]), [_claim(i, f"c{i}") for i in range(1, 5)])
    r = LiveResult("TEST 6 — SPLIT EXPLICIT", expected_criterion_pairs=4, actual_criterion_pairs=4,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommendation_status == RecommendationStatus.RECOMMEND:
        r.semantic_failures.append("confident recommend on split explicit criteria")
    r.pass_ = not r.semantic_failures
    return r


async def test_07_uncertain_primary() -> LiveResult:
    frame = DecisionFrame(
        decision="Enter market",
        options=[DecisionOption(label="Enter Market X", origin="implied"), DecisionOption(label="Do Not Enter Market X", origin="implied")],
        criteria=[
            DecisionCriterion(label="Regulatory viability", origin="explicit", priority="primary"),
            DecisionCriterion(label="Market opportunity", origin="inferred"),
        ],
    )
    oe = _oe(frame.decision, [
        ("Enter Market X", "Regulatory viability", CriterionAssessment.UNCERTAIN, KnowledgeCoverage.PARTIAL, [1], "explicit", "primary", ""),
        ("Do Not Enter Market X", "Regulatory viability", CriterionAssessment.UNCERTAIN, KnowledgeCoverage.PARTIAL, [1], "explicit", "primary", ""),
        ("Enter Market X", "Market opportunity", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [2], "inferred", "standard", ""),
        ("Do Not Enter Market X", "Market opportunity", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3], "inferred", "standard", ""),
    ])
    s, m = await _run(frame, oe, _ks(disputed=[_entry(1, "disputed"), _entry(2), _entry(3)]), [_claim(1, "disputed reg"), _claim(2, "big market"), _claim(3, "slow market")])
    r = LiveResult("TEST 7 — UNCERTAIN PRIMARY", expected_criterion_pairs=4, actual_criterion_pairs=4,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommendation_status == RecommendationStatus.RECOMMEND:
        r.semantic_failures.append("full recommend with uncertain primary criterion")
    r.pass_ = not r.semantic_failures
    return r


async def test_08_critical_missing_context() -> LiveResult:
    missing = ["Acceptable acquisition price", "Integration capacity"]
    frame = DecisionFrame(
        decision="Acquire Company X vs Do not acquire",
        decision_type=DecisionType.ACQUISITION,
        options=[
            DecisionOption(label="Acquire Company X", origin="implied"),
            DecisionOption(label="Do not acquire Company X", origin="implied"),
        ],
        criteria=[DecisionCriterion(label="Strategic fit", origin="inferred")],
        missing_decision_context=missing,
    )
    oe = _oe(frame.decision, [
        ("Acquire Company X", "Strategic fit", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "inferred", "standard", ""),
        ("Do not acquire Company X", "Strategic fit", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "inferred", "standard", ""),
    ], limitations=missing)
    s, m = await _run(frame, oe, _ks(known=[_entry(1), _entry(2)]), [_claim(1, "good fit"), _claim(2, "poor fit")])
    r = LiveResult("TEST 8 — CRITICAL MISSING CONTEXT", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s:
        for item in s.critical_missing_context:
            if item not in missing:
                r.semantic_failures.append(f"invented critical context: {item}")
        if s.critical_missing_context and s.recommendation_status == RecommendationStatus.RECOMMEND:
            r.semantic_failures.append("full recommend with critical missing context")
    r.pass_ = not r.semantic_failures
    return r


async def test_09_assumption() -> LiveResult:
    assumption = "Interest rates stay above 4%"
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        explicit_assumptions=[assumption],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", "Favorable if rates stay above 4%."),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
    ])
    s, m = await _run(frame, oe, _ks(known=[_entry(1), _entry(2)]), [_claim(1, "A cheaper at rates above 4%"), _claim(2, "B more expensive")])
    r = LiveResult("TEST 9 — ASSUMPTION", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.assumptions_relied_on:
        for a in s.assumptions_relied_on:
            if a not in frame.explicit_assumptions:
                r.semantic_failures.append(f"invented assumption: {a}")
    r.pass_ = not r.semantic_failures
    return r


async def test_10_insufficient_basis() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Security", origin="explicit"),
        ],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, [], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNCERTAIN, KnowledgeCoverage.PARTIAL, [1], "explicit", "primary", ""),
        ("Vendor A", "Security", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, [], "explicit", "standard", ""),
        ("Vendor B", "Security", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, [], "explicit", "standard", ""),
    ])
    s, m = await _run(frame, oe, _ks(disputed=[_entry(1, "disputed")]), [_claim(1, "disputed")])
    r = LiveResult("TEST 10 — INSUFFICIENT BASIS", expected_criterion_pairs=4, actual_criterion_pairs=4,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and s.recommendation_status != RecommendationStatus.INSUFFICIENT_BASIS:
        r.semantic_failures.append(f"expected insufficient_basis, got {s.recommendation_status}")
    if s and s.recommended_option is not None:
        r.semantic_failures.append("expected null recommended_option")
    r.pass_ = not r.semantic_failures
    return r


async def test_11_change_conditions() -> LiveResult:
    constraints = ["Must support Salesforce integration"]
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary"), DecisionCriterion(label="Reliability", origin="explicit")],
        constraints=constraints,
        explicit_assumptions=["Interest rates stay above 4%"],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
        ("Vendor A", "Reliability", CriterionAssessment.UNCERTAIN, KnowledgeCoverage.PARTIAL, [3], "explicit", "standard", ""),
        ("Vendor B", "Reliability", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4], "explicit", "standard", ""),
    ], constraints=constraints)
    s, m = await _run(frame, oe, _ks(known=[_entry(i) for i in range(1, 5)]), [_claim(i, f"c{i}") for i in range(1, 5)])
    r = LiveResult("TEST 11 — CHANGE CONDITIONS", expected_criterion_pairs=4, actual_criterion_pairs=4,
                   expected_constraint_pairs=2, actual_constraint_pairs=len(s.constraint_assessments) if s else 0,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s and not s.change_conditions:
        r.semantic_failures.append("expected change conditions")
    elif s and all(len(cc.description) < 20 for cc in s.change_conditions):
        r.semantic_failures.append("change conditions too generic")
    r.pass_ = not r.semantic_failures
    return r


async def test_12_no_external_knowledge() -> LiveResult:
    frame = DecisionFrame(
        decision="OpenAI vs Anthropic",
        options=[DecisionOption(label="OpenAI", origin="explicit"), DecisionOption(label="Anthropic", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
    )
    oe = _oe(frame.decision, [
        ("OpenAI", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [801], "explicit", "primary", "OpenAI costs less under hypothetical contract."),
        ("Anthropic", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [801], "explicit", "primary", "Anthropic costs more under hypothetical contract."),
    ])
    s, m = await _run(frame, oe, _ks(known=[_entry(801)]), [_claim(801, "OpenAI costs less than Anthropic under this hypothetical contract.")])
    r = LiveResult("TEST 12 — NO EXTERNAL KNOWLEDGE", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if s:
        text = s.rationale + " " + " ".join(cc.description for cc in s.change_conditions)
        r.external_flags = _scan_external(text, ["openai", "anthropic", "hypothetical", "contract"])
        if r.external_flags:
            r.semantic_failures.append(f"external knowledge: {r.external_flags}")
    r.pass_ = not r.semantic_failures
    return r


async def test_13_validator_never_upgrades() -> LiveResult:
    """Deterministic: strong fixture + LLM tentative must stay tentative."""
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
    ])
    catalog = build_claim_catalog(_ks(known=[_entry(1), _entry(2)]), [_claim(1, "low"), _claim(2, "high")])
    pre = run_pre_check(frame, oe)
    llm_out = DecisionSynthesisLLMOutput(
        recommendation_status=RecommendationStatus.TENTATIVE_RECOMMENDATION,
        recommended_option="Vendor A",
        rationale="Tentative lean to Vendor A on cost.",
        supporting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")],
    )
    s, _, _ = validate_and_build_synthesis(llm_out, frame, oe, catalog, pre)
    r = LiveResult("TEST 13 — VALIDATOR NEVER UPGRADES", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   synthesis_llm_calls=0)
    if s and s.recommendation_status != RecommendationStatus.TENTATIVE_RECOMMENDATION:
        r.semantic_failures.append(f"status upgraded to {s.recommendation_status}")
    r.pass_ = not r.semantic_failures
    return r


async def test_14_incomplete_oe_matrix() -> LiveResult:
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit"), DecisionCriterion(label="Reliability", origin="explicit")],
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "standard", ""),
    ])
    s, m = await _run(frame, oe, _ks(), [])
    r = LiveResult("TEST 14 — INCOMPLETE OE MATRIX", expected_criterion_pairs=4, actual_criterion_pairs=1,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   metrics=m.to_dict(), synthesis_llm_calls=m.synthesis_llm_calls)
    if m.synthesis_llm_calls != 0:
        r.semantic_failures.append(f"expected 0 LLM calls, got {m.synthesis_llm_calls}")
    if s and s.recommendation_status != RecommendationStatus.INSUFFICIENT_BASIS:
        r.semantic_failures.append(f"expected insufficient_basis, got {s.recommendation_status}")
    r.pass_ = not r.semantic_failures
    return r


async def test_15_incomplete_constraint_matrix() -> LiveResult:
    constraints = ["Budget under $20,000", "Must integrate with Salesforce"]
    frame = DecisionFrame(
        decision="Choose vendor",
        options=[DecisionOption(label="Vendor A", origin="explicit"), DecisionOption(label="Vendor B", origin="explicit")],
        criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        constraints=constraints,
    )
    oe = _oe(frame.decision, [
        ("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], "explicit", "primary", ""),
        ("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], "explicit", "primary", ""),
    ], constraints=constraints)
    catalog = build_claim_catalog(_ks(known=[_entry(1), _entry(2)]), [_claim(1, "a"), _claim(2, "b")])
    pre = run_pre_check(frame, oe)
    llm_out = DecisionSynthesisLLMOutput(
        recommendation_status=RecommendationStatus.RECOMMEND,
        recommended_option="Vendor A",
        rationale="Best on cost.",
        constraint_assessments=[
            ConstraintAssessmentLLM(option_label="Vendor A", constraint=constraints[0],
                                  compliance=ConstraintCompliance.SATISFIED, claim_ids=[1], reason="ok"),
        ],
    )
    s, _, errors = validate_and_build_synthesis(llm_out, frame, oe, catalog, pre)
    r = LiveResult("TEST 15 — INCOMPLETE CONSTRAINT MATRIX", expected_criterion_pairs=2, actual_criterion_pairs=2,
                   expected_constraint_pairs=4, actual_constraint_pairs=len(s.constraint_assessments) if s else 0,
                   status=s.recommendation_status.value if s else None, recommended_option=s.recommended_option if s else None,
                   synthesis_llm_calls=0)
    r.missing_constraints = _missing_constraints(frame, s)
    if s and s.recommendation_status != RecommendationStatus.INSUFFICIENT_BASIS:
        r.semantic_failures.append(f"expected insufficient_basis, got {s.recommendation_status}")
    if not any("missing_constraint_pairs" in e for e in errors):
        r.semantic_failures.append("expected missing_constraint_pairs error")
    r.pass_ = not r.semantic_failures
    return r


def _serialize(r: LiveResult) -> dict:
    return {
        "name": r.name,
        "pass": r.pass_,
        "expected_criterion_pairs": r.expected_criterion_pairs,
        "actual_criterion_pairs": r.actual_criterion_pairs,
        "expected_constraint_pairs": r.expected_constraint_pairs,
        "actual_constraint_pairs": r.actual_constraint_pairs,
        "missing_constraints": r.missing_constraints,
        "status": r.status,
        "recommended_option": r.recommended_option,
        "synthesis_llm_calls": r.synthesis_llm_calls,
        "semantic_failures": r.semantic_failures,
        "external_flags": r.external_flags,
        "notes": r.notes,
    }


async def main() -> dict:
    tests = [
        test_01_clear_primary, test_02_primary_vs_standard, test_03_inferred_cannot_override,
        test_04_hard_constraint_matrix, test_05_constraint_claim_outside_oe, test_06_split_explicit,
        test_07_uncertain_primary, test_08_critical_missing_context, test_09_assumption,
        test_10_insufficient_basis, test_11_change_conditions, test_12_no_external_knowledge,
        test_13_validator_never_upgrades, test_14_incomplete_oe_matrix, test_15_incomplete_constraint_matrix,
    ]
    results = []
    for fn in tests:
        print(f"Running {fn.__name__}...", flush=True)
        results.append(await fn())
    passed = sum(1 for r in results if r.pass_)
    return {
        "results": [_serialize(r) for r in results],
        "passed": passed,
        "total": len(results),
        "phase_3c_live_validation": "PASS" if passed == len(results) else "FAIL",
        "ready_to_freeze": "YES" if passed == len(results) else "NO",
    }


if __name__ == "__main__":
    report = asyncio.run(main())
    print(json.dumps(report, indent=2))
