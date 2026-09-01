"""Live validation for Phase 3B option evaluation (isolated fixtures + optional E2E)."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from domain.models import Claim, ClaimType, EvidenceConfidence, KnowledgeCategory, VerificationStatus
from services.decision_framing_schemas import DecisionCriterion, DecisionFrame, DecisionOption, DecisionType
from services.knowledge_state_schemas import KnowledgeState, KnowledgeStateEntry
from services.option_evaluation import (
    _has_recommendation_leakage,
    build_claim_catalog,
    evaluate_options,
    format_claim_catalog,
    format_decision_frame_for_prompt,
    validate_and_build_evaluation,
)
from services.option_evaluation_schemas import OptionEvaluationLLMOutput

_RECOMMENDATION_PATTERNS = [
    r"\bwe recommend\b",
    r"\bi recommend\b",
    r"\bour recommendation\b",
    r"\bthe best (choice|option)\b",
    r"\byou should (choose|select|pick)\b",
    r"\bthe winner is\b",
    r"\bwins\b",
    r"\bshould (choose|select|pick|go with)\b",
]

_EXTERNAL_KNOWLEDGE_HINTS = [
    r"\bapi pricing\b",
    r"\bcontext window\b",
    r"\bgpt-?\d\b",
    r"\bclaude\b",
    r"\bsoc\s*2\b",
    r"\benterprise (features|plan)\b",
    r"\bpublicly\b",
    r"\bmarket leader\b",
    r"\bindustry standard\b",
    r"\$\d+/(?:token|million|request)\b",
    r"\bper token\b",
    r"\bopenai'?s (?:pricing|api)\b",
    r"\banthropic'?s (?:pricing|api)\b",
]


def _claim(cid: int, text: str) -> Claim:
    return Claim(id=cid, research_run_id=1, text=text, claim_type=ClaimType.FACTUAL)


def _entry(
    cid: int,
    bucket: str,
    *,
    status: VerificationStatus | None = None,
    vid: int | None = None,
) -> KnowledgeStateEntry:
    status_map = {
        "known": VerificationStatus.SUPPORTED,
        "likely": VerificationStatus.PARTIALLY_SUPPORTED,
        "disputed": VerificationStatus.UNCERTAIN,
        "unknown": VerificationStatus.INSUFFICIENT_EVIDENCE,
        "contradicted": VerificationStatus.CONTRADICTED,
        "unverifiable": VerificationStatus.UNVERIFIABLE,
    }
    cat_map = {
        "known": KnowledgeCategory.KNOWN,
        "likely": KnowledgeCategory.LIKELY,
        "disputed": KnowledgeCategory.DISPUTED,
        "unknown": KnowledgeCategory.UNKNOWN,
        "contradicted": None,
        "unverifiable": None,
    }
    return KnowledgeStateEntry(
        claim_id=cid,
        verification_id=vid or cid * 10,
        knowledge_category=cat_map.get(bucket),
        verification_status=status or status_map[bucket],
        confidence=EvidenceConfidence.HIGH,
        evidence_ids=[cid * 100],
    )


def _ks(**buckets: list[KnowledgeStateEntry]) -> KnowledgeState:
    return KnowledgeState(
        known=buckets.get("known", []),
        likely=buckets.get("likely", []),
        disputed=buckets.get("disputed", []),
        unknown=buckets.get("unknown", []),
        contradicted=buckets.get("contradicted", []),
        unverifiable=buckets.get("unverifiable", []),
    )


@dataclass
class PairResult:
    option: str
    criterion: str
    assessment: str | None = None
    coverage: str | None = None
    claim_ids: list[int] = field(default_factory=list)
    reason: str | None = None
    option_origin: str | None = None
    criterion_origin: str | None = None


@dataclass
class TestResult:
    name: str
    expected_pairs: int
    llm_returned_rows: int
    validated_pairs: int
    missing_pairs: list[str]
    pairs: list[PairResult]
    metrics: dict[str, Any]
    evaluation_failed: bool
    evaluation_skipped: bool
    semantic_failures: list[str] = field(default_factory=list)
    epistemic_notes: list[str] = field(default_factory=list)
    recommendation_leakage: list[str] = field(default_factory=list)
    external_knowledge_flags: list[str] = field(default_factory=list)
    provenance_ok: bool | None = None
    pass_: bool = False


def _expected_pair_labels(frame: DecisionFrame) -> list[tuple[str, str]]:
    return [(o.label, c.label) for o in frame.options for c in frame.criteria]


def _extract_pairs(evaluation) -> list[PairResult]:
    if evaluation is None:
        return []
    out: list[PairResult] = []
    for opt in evaluation.option_evaluations:
        for ce in opt.criteria_evaluations:
            out.append(
                PairResult(
                    option=opt.option_label,
                    criterion=ce.criterion_label,
                    assessment=ce.assessment.value,
                    coverage=ce.knowledge_coverage.value,
                    claim_ids=list(ce.claim_ids),
                    reason=ce.reason,
                    option_origin=opt.option_origin,
                    criterion_origin=ce.criterion_origin,
                )
            )
    return out


def _missing_pairs(frame: DecisionFrame, pairs: list[PairResult]) -> list[str]:
    present = {(p.option, p.criterion) for p in pairs}
    return [f"{o} × {c}" for o, c in _expected_pair_labels(frame) if (o, c) not in present]


def _find_pair(pairs: list[PairResult], option: str, criterion: str) -> PairResult | None:
    for p in pairs:
        if p.option == option and p.criterion == criterion:
            return p
    return None


def _check_recommendation(text: str) -> list[str]:
    hits = []
    for pat in _RECOMMENDATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    if _has_recommendation_leakage(text):
        hits.append("validator_leakage_regex")
    return hits


def _check_external_knowledge(text: str, allowed_terms: list[str] | None = None) -> list[str]:
    allowed = {t.lower() for t in (allowed_terms or [])}
    flags = []
    for pat in _EXTERNAL_KNOWLEDGE_HINTS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.group(0).lower() not in allowed:
            flags.append(m.group(0))
    return flags


async def _run_live(frame: DecisionFrame, ks: KnowledgeState, claims: list[Claim]) -> tuple:
    """Run live LLM eval and capture raw + validated output."""
    from langchain_anthropic import ChatAnthropic

    from config import settings

    if not frame.options:
        return await evaluate_options(frame, ks, claims, llm=None)

    catalog = build_claim_catalog(ks, claims)
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )
    user_prompt = (
        f"DECISION FRAME:\n{format_decision_frame_for_prompt(frame)}\n\n"
        f"CLAIM CATALOG:\n{format_claim_catalog(catalog)}\n\n"
        "Produce one evaluation row for every option × criterion pair."
    )
    from services.option_evaluation import _EVALUATION_SYSTEM_PROMPT

    structured = llm.with_structured_output(OptionEvaluationLLMOutput)
    llm_output: OptionEvaluationLLMOutput = await structured.ainvoke(
        [
            {"role": "system", "content": _EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    evaluation, metrics = validate_and_build_evaluation(llm_output, frame, catalog)
    metrics.evaluation_llm_calls = 1
    return evaluation, metrics, llm_output, catalog


def _base_result(name: str, frame: DecisionFrame, evaluation, metrics, llm_rows: int) -> TestResult:
    pairs = _extract_pairs(evaluation)
    expected = len(frame.options) * len(frame.criteria)
    missing = _missing_pairs(frame, pairs)
    return TestResult(
        name=name,
        expected_pairs=expected,
        llm_returned_rows=llm_rows,
        validated_pairs=len(pairs),
        missing_pairs=missing,
        pairs=pairs,
        metrics=metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics),
        evaluation_failed=getattr(metrics, "evaluation_failed", False),
        evaluation_skipped=getattr(metrics, "evaluation_skipped", False),
    )


async def test_01_clear_grounded_direction() -> TestResult:
    frame = DecisionFrame(
        decision="Which vendor should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="Vendor A", origin="explicit"),
            DecisionOption(label="Vendor B", origin="explicit"),
        ],
        criteria=[DecisionCriterion(label="Cost", origin="explicit")],
    )
    claims = [
        _claim(101, "At the stated usage level, Vendor A costs $12,000 per year."),
        _claim(102, "At the stated usage level, Vendor B costs $20,000 per year."),
    ]
    ks = _ks(known=[_entry(101, "known"), _entry(102, "known")])
    evaluation, metrics, llm_out, catalog = await _run_live(frame, ks, claims)
    r = _base_result("TEST 1 — CLEAR GROUNDED DIRECTION", frame, evaluation, metrics, len(llm_out.evaluations))

    va = _find_pair(r.pairs, "Vendor A", "Cost")
    vb = _find_pair(r.pairs, "Vendor B", "Cost")
    if va is None or vb is None:
        r.semantic_failures.append("Missing Vendor A or Vendor B Cost evaluation")
    else:
        if va.assessment != "favorable":
            r.semantic_failures.append(f"Vendor A Cost expected favorable, got {va.assessment}")
        if vb.assessment != "unfavorable":
            r.semantic_failures.append(f"Vendor B Cost expected unfavorable, got {vb.assessment}")
        if va.coverage != "grounded":
            r.semantic_failures.append(f"Vendor A Cost expected grounded, got {va.coverage}")
        for p in (va, vb):
            if any(cid not in catalog for cid in p.claim_ids):
                r.semantic_failures.append(f"Invalid claim IDs in {p.option}×{p.criterion}: {p.claim_ids}")
            if p.reason:
                r.recommendation_leakage.extend(_check_recommendation(p.reason))
                if re.search(r"\b(discount|switching cost|contract term)\b", p.reason, re.I):
                    r.semantic_failures.append(f"Invented terms in reason: {p.reason[:120]}")

    r.pass_ = not r.semantic_failures and not r.missing_pairs and not r.evaluation_failed
    return r


async def test_02_multi_criterion_matrix() -> TestResult:
    frame = DecisionFrame(
        decision="Which platform should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="Platform A", origin="explicit"),
            DecisionOption(label="Platform B", origin="explicit"),
        ],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Reliability", origin="explicit"),
            DecisionCriterion(label="Integration", origin="inferred"),
        ],
    )
    claims = [
        _claim(201, "Platform A annual cost is $8,000 at stated usage."),
        _claim(202, "Platform B annual cost is $15,000 at stated usage."),
        _claim(203, "Platform A uptime in the reference period was 99.1%."),
        _claim(204, "Platform B uptime in the reference period was 99.9%."),
    ]
    ks = _ks(known=[_entry(201, "known"), _entry(202, "known"), _entry(203, "known"), _entry(204, "known")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 2 — MULTI-CRITERION COMPLETE MATRIX", frame, evaluation, metrics, len(llm_out.evaluations))

    expectations = {
        ("Platform A", "Cost"): "favorable",
        ("Platform B", "Cost"): "unfavorable",
        ("Platform A", "Reliability"): "unfavorable",
        ("Platform B", "Reliability"): "favorable",
        ("Platform A", "Integration"): "insufficient_information",
        ("Platform B", "Integration"): "insufficient_information",
    }
    for (opt, crit), exp in expectations.items():
        p = _find_pair(r.pairs, opt, crit)
        if p is None:
            continue
        if crit == "Integration":
            if p.assessment != "insufficient_information":
                r.semantic_failures.append(f"{opt}×Integration expected insufficient_information, got {p.assessment}")
            if p.claim_ids:
                r.semantic_failures.append(f"{opt}×Integration should have empty claim_ids, got {p.claim_ids}")
        else:
            if p.assessment != exp:
                r.semantic_failures.append(f"{opt}×{crit} expected {exp}, got {p.assessment}")
        if p.reason:
            r.recommendation_leakage.extend(_check_recommendation(p.reason))

    if r.missing_pairs:
        r.semantic_failures.append(f"Systematic omission suspected: missing {r.missing_pairs}")

    r.pass_ = not r.semantic_failures and r.validated_pairs == 6 and not r.evaluation_failed
    return r


async def test_03_disputed() -> TestResult:
    frame = DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[
            DecisionOption(label="Enter Market X", origin="implied"),
            DecisionOption(label="Do Not Enter Market X", origin="implied"),
        ],
        criteria=[DecisionCriterion(label="Regulatory environment", origin="inferred")],
    )
    claims = [_claim(301, "Regulatory requirements for Market X are subject to pending legislation and disputed among analysts.")]
    ks = _ks(disputed=[_entry(301, "disputed")])
    evaluation, metrics, llm_out, catalog = await _run_live(frame, ks, claims)
    r = _base_result("TEST 3 — DISPUTED KNOWLEDGE", frame, evaluation, metrics, len(llm_out.evaluations))

    for p in r.pairs:
        if p.assessment in ("favorable", "unfavorable"):
            r.semantic_failures.append(f"{p.option}×{p.criterion} must not be confidently directional, got {p.assessment}")
        if p.assessment not in ("uncertain", "mixed", "neutral", "insufficient_information"):
            r.epistemic_notes.append(f"{p.option}×{p.criterion}: assessment={p.assessment} (acceptable if not directional)")
        if 301 in p.claim_ids and p.assessment in ("uncertain",):
            r.epistemic_notes.append("Validator enforced uncertain on disputed claim")
        if p.claim_ids and any(cid not in catalog for cid in p.claim_ids):
            r.semantic_failures.append(f"Invalid claim refs: {p.claim_ids}")
        if p.reason and re.search(r"\b(law|regulation|permit)\b.*\b(requires|mandates)\b", p.reason, re.I):
            if "disputed" not in p.reason.lower() and p.assessment not in ("uncertain",):
                r.semantic_failures.append(f"Possible invented regulatory fact: {p.reason[:100]}")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_04_contradicted() -> TestResult:
    frame = DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[
            DecisionOption(label="Enter Market X", origin="implied"),
            DecisionOption(label="Do Not Enter Market X", origin="implied"),
        ],
        criteria=[DecisionCriterion(label="Market growth", origin="inferred")],
    )
    claims = [_claim(401, "Market X growth estimates conflict across sources.")]
    ks = _ks(contradicted=[_entry(401, "contradicted")])
    evaluation, metrics, llm_out, catalog = await _run_live(frame, ks, claims)
    r = _base_result("TEST 4 — CONTRADICTED KNOWLEDGE", frame, evaluation, metrics, len(llm_out.evaluations))

    for p in r.pairs:
        if p.assessment in ("favorable", "unfavorable"):
            r.semantic_failures.append(f"Contradicted-only should not stay directional: {p.assessment}")
        if p.assessment != "insufficient_information" and 401 in p.claim_ids:
            r.semantic_failures.append(f"Expected insufficient_information with claim 401, got {p.assessment}")
        if p.verification_ids if hasattr(p, "verification_ids") else False:
            pass
        if p.claim_ids:
            r.epistemic_notes.append(f"{p.option}: assessment={p.assessment}, claims={p.claim_ids}")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_05_unverifiable() -> TestResult:
    frame = DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[DecisionOption(label="Enter Market X", origin="implied")],
        criteria=[DecisionCriterion(label="Competitive intensity", origin="inferred")],
    )
    claims = [_claim(501, "Competitive dynamics in Market X cannot be verified from available sources.")]
    ks = _ks(unverifiable=[_entry(501, "unverifiable")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 5 — UNVERIFIABLE KNOWLEDGE", frame, evaluation, metrics, len(llm_out.evaluations))

    p = _find_pair(r.pairs, "Enter Market X", "Competitive intensity")
    if p and p.assessment in ("favorable", "unfavorable"):
        r.semantic_failures.append(f"Unverifiable claim must not yield confident directional: {p.assessment}")
    if p:
        r.epistemic_notes.append(f"assessment={p.assessment}, coverage={p.coverage}")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_06_mixed() -> TestResult:
    frame = DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[DecisionOption(label="Enter Market X", origin="implied")],
        criteria=[DecisionCriterion(label="Market opportunity", origin="inferred")],
    )
    claims = [
        _claim(601, "Market X has a large addressable market."),
        _claim(602, "Demand growth in Market X has recently slowed materially."),
    ]
    ks = _ks(known=[_entry(601, "known"), _entry(602, "known")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 6 — MIXED EVIDENCE", frame, evaluation, metrics, len(llm_out.evaluations))

    p = _find_pair(r.pairs, "Enter Market X", "Market opportunity")
    if p:
        if p.assessment != "mixed":
            r.semantic_failures.append(f"Expected mixed, got {p.assessment}")
        if not ({601, 602} & set(p.claim_ids)):
            r.semantic_failures.append(f"Expected both claims referenced, got {p.claim_ids}")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_07_no_relevant_info() -> TestResult:
    frame = DecisionFrame(
        decision="Which vendor should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[DecisionOption(label="Vendor A", origin="explicit")],
        criteria=[DecisionCriterion(label="Implementation cost", origin="inferred")],
    )
    claims = [
        _claim(701, "Vendor A has strong reliability metrics."),
        _claim(702, "Vendor A meets stated security requirements."),
        _claim(703, "Vendor A holds significant market share."),
    ]
    ks = _ks(known=[_entry(701, "known"), _entry(702, "known"), _entry(703, "known")])
    evaluation, metrics, llm_out, catalog = await _run_live(frame, ks, claims)
    r = _base_result("TEST 7 — NO RELEVANT INFORMATION", frame, evaluation, metrics, len(llm_out.evaluations))

    p = _find_pair(r.pairs, "Vendor A", "Implementation cost")
    if p:
        if p.assessment != "insufficient_information":
            r.semantic_failures.append(f"Expected insufficient_information, got {p.assessment}")
        if p.claim_ids:
            r.semantic_failures.append(f"Should not cite unrelated claims, got {p.claim_ids}")
        if p.reason:
            r.external_knowledge_flags.extend(_check_external_knowledge(p.reason))

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_08_no_external_knowledge() -> TestResult:
    frame = DecisionFrame(
        decision="Which LLM provider should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="OpenAI", origin="explicit"),
            DecisionOption(label="Anthropic", origin="explicit"),
        ],
        criteria=[DecisionCriterion(label="Cost", origin="explicit")],
    )
    claims = [_claim(801, "OpenAI costs less than Anthropic under this hypothetical contract.")]
    ks = _ks(known=[_entry(801, "known")])
    evaluation, metrics, llm_out, catalog = await _run_live(frame, ks, claims)
    r = _base_result("TEST 8 — NO EXTERNAL KNOWLEDGE", frame, evaluation, metrics, len(llm_out.evaluations))

    allowed = ["openai", "anthropic", "hypothetical", "contract", "cost", "less"]
    for p in r.pairs:
        if p.reason:
            flags = _check_external_knowledge(p.reason, allowed_terms=allowed)
            r.external_knowledge_flags.extend(flags)
            if flags:
                r.semantic_failures.append(f"External knowledge in reason: {flags} — {p.reason[:150]}")
        for cid in p.claim_ids:
            if cid not in catalog:
                r.semantic_failures.append(f"Claim {cid} not in catalog")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_09_binary_no_mirror() -> TestResult:
    frame = DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[
            DecisionOption(label="Enter Market X", origin="implied"),
            DecisionOption(label="Do Not Enter Market X", origin="implied"),
        ],
        criteria=[DecisionCriterion(label="Strategic fit", origin="inferred")],
    )
    claims = [_claim(901, "Entering Market X would strain the company's existing strategic focus and core competencies.")]
    ks = _ks(known=[_entry(901, "known")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 9 — BINARY OPTIONS NOT MIRRORED", frame, evaluation, metrics, len(llm_out.evaluations))

    enter = _find_pair(r.pairs, "Enter Market X", "Strategic fit")
    dont = _find_pair(r.pairs, "Do Not Enter Market X", "Strategic fit")
    if enter and dont:
        if enter.assessment == "unfavorable" and dont.assessment == "favorable":
            r.semantic_failures.append(
                "Suspicious perfect mirror: Enter=unfavorable AND Do Not Enter=favorable without independent reasoning"
            )
        r.epistemic_notes.append(f"Enter={enter.assessment}, DoNotEnter={dont.assessment}")

    r.pass_ = not r.semantic_failures and not r.evaluation_failed
    return r


async def test_10_recommendation_leakage() -> TestResult:
    frame = DecisionFrame(
        decision="Which vendor should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="Vendor A", origin="explicit"),
            DecisionOption(label="Vendor B", origin="explicit"),
        ],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Security", origin="explicit"),
        ],
    )
    claims = [
        _claim(1001, "Vendor A costs $5,000 per year at stated usage."),
        _claim(1002, "Vendor B costs $18,000 per year at stated usage."),
        _claim(1003, "Vendor A meets all stated security requirements."),
        _claim(1004, "Vendor B has unresolved security compliance gaps."),
    ]
    ks = _ks(known=[_entry(1001, "known"), _entry(1002, "known"), _entry(1003, "known"), _entry(1004, "known")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 10 — RECOMMENDATION LEAKAGE", frame, evaluation, metrics, len(llm_out.evaluations))

    for row in llm_out.evaluations:
        if _has_recommendation_leakage(row.reason):
            r.recommendation_leakage.append(f"RAW rejected candidate: {row.reason[:120]}")

    for p in r.pairs:
        if p.reason:
            r.recommendation_leakage.extend(_check_recommendation(p.reason))

    if r.evaluation_failed and metrics.failure_reason and "recommendation" in metrics.failure_reason:
        r.epistemic_notes.append(f"Validator rejected output: {metrics.failure_reason}")
    elif not r.recommendation_leakage:
        r.epistemic_notes.append("No recommendation language in validated output")

    r.pass_ = not any("recommend" in x.lower() or "best" in x.lower() or "winner" in x.lower() or "should choose" in x.lower() for x in r.recommendation_leakage if "RAW rejected" not in x)
    return r


async def test_11_provenance() -> TestResult:
    frame = DecisionFrame(
        decision="Which vendor should we use?",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[DecisionOption(label="Vendor A", origin="explicit")],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Implementation ease", origin="inferred"),
        ],
    )
    claims = [
        _claim(1101, "Vendor A pricing is competitive."),
        _claim(1102, "Vendor A deployment requires moderate engineering effort."),
    ]
    ks = _ks(known=[_entry(1101, "known"), _entry(1102, "known")])
    evaluation, metrics, llm_out, _ = await _run_live(frame, ks, claims)
    r = _base_result("TEST 11 — PROVENANCE", frame, evaluation, metrics, len(llm_out.evaluations))

    provenance_ok = True
    for p in r.pairs:
        if p.option_origin != "explicit":
            provenance_ok = False
            r.semantic_failures.append(f"option_origin expected explicit, got {p.option_origin}")
        if p.criterion == "Cost" and p.criterion_origin != "explicit":
            provenance_ok = False
            r.semantic_failures.append(f"Cost origin expected explicit, got {p.criterion_origin}")
        if p.criterion == "Implementation ease" and p.criterion_origin != "inferred":
            provenance_ok = False
            r.semantic_failures.append(f"Implementation ease origin expected inferred, got {p.criterion_origin}")

    r.provenance_ok = provenance_ok
    r.pass_ = provenance_ok and not r.evaluation_failed
    return r


async def test_12_empty_options() -> TestResult:
    frame = DecisionFrame(
        decision="Which CRM to choose",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[],
        criteria=[DecisionCriterion(label="Fit", origin="inferred")],
    )
    claims = [_claim(1201, "Some fact")]
    ks = _ks(known=[_entry(1201, "known")])
    evaluation, metrics = await evaluate_options(frame, ks, claims, llm=None)
    r = _base_result("TEST 12 — EMPTY OPTIONS", frame, evaluation, metrics, 0)

    if evaluation is not None:
        r.semantic_failures.append("Expected option_evaluation=None")
    if not metrics.evaluation_skipped:
        r.semantic_failures.append("Expected evaluation_skipped=True")
    if metrics.evaluation_skipped_reason != "no_concrete_options":
        r.semantic_failures.append(f"Expected skip reason no_concrete_options, got {metrics.evaluation_skipped_reason}")
    if metrics.evaluation_failed:
        r.semantic_failures.append("Should not be evaluation_failed")
    if metrics.evaluation_llm_calls != 0:
        r.semantic_failures.append(f"Expected 0 LLM calls, got {metrics.evaluation_llm_calls}")

    r.pass_ = not r.semantic_failures
    return r


async def optional_e2e() -> dict:
    from graph import create_graph, create_run_config
    from services.pipeline_init import create_initial_state, finalize_from_state

    query = "Should our company enter the Japanese EV charging market?"
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    final_state = await app.ainvoke(initial_state, config=create_run_config())
    await finalize_from_state(final_state, ctx)

    frame = final_state.get("decision_frame") or {}
    ks = final_state.get("knowledge_state") or {}
    oe = final_state.get("option_evaluation")
    metrics = final_state.get("option_evaluation_metrics") or {}

    pairs = []
    if oe:
        for opt in oe.get("option_evaluations", []):
            for ce in opt.get("criteria_evaluations", []):
                pairs.append({
                    "option": opt.get("option_label"),
                    "criterion": ce.get("criterion_label"),
                    "assessment": ce.get("assessment"),
                    "coverage": ce.get("knowledge_coverage"),
                    "claim_ids": ce.get("claim_ids"),
                    "reason": (ce.get("reason") or "")[:200],
                })

    return {
        "query": query,
        "options": [o.get("label") for o in frame.get("options", [])],
        "criteria": [c.get("label") for c in frame.get("criteria", [])],
        "material_claims": len(final_state.get("material_claims") or []),
        "catalog_buckets": {k: len(ks.get(k, [])) for k in ["known", "likely", "disputed", "unknown", "contradicted", "unverifiable"]},
        "option_evaluation_present": oe is not None,
        "metrics": metrics,
        "validated_pairs": len(pairs),
        "expected_pairs": len(frame.get("options", [])) * len(frame.get("criteria", [])),
        "sample_pairs": pairs[:6],
        "evaluation_failed": metrics.get("evaluation_failed"),
        "evaluation_skipped": metrics.get("evaluation_skipped"),
        "invalid_refs": metrics.get("invalid_reference_count", 0),
        "rejected_rows": metrics.get("rejected_row_count", 0),
    }


def _serialize_result(r: TestResult) -> dict:
    return {
        "name": r.name,
        "pass": r.pass_,
        "expected_pairs": r.expected_pairs,
        "llm_returned_rows": r.llm_returned_rows,
        "validated_pairs": r.validated_pairs,
        "missing_pairs": r.missing_pairs,
        "pairs": [
            {
                "option": p.option,
                "criterion": p.criterion,
                "assessment": p.assessment,
                "coverage": p.coverage,
                "claim_ids": p.claim_ids,
                "option_origin": p.option_origin,
                "criterion_origin": p.criterion_origin,
                "reason": (p.reason or "")[:250],
            }
            for p in r.pairs
        ],
        "metrics": r.metrics,
        "evaluation_failed": r.evaluation_failed,
        "evaluation_skipped": r.evaluation_skipped,
        "semantic_failures": r.semantic_failures,
        "epistemic_notes": r.epistemic_notes,
        "recommendation_leakage": r.recommendation_leakage,
        "external_knowledge_flags": r.external_knowledge_flags,
        "provenance_ok": r.provenance_ok,
    }


async def main(run_e2e: bool = True) -> dict:
    tests = [
        test_01_clear_grounded_direction,
        test_02_multi_criterion_matrix,
        test_03_disputed,
        test_04_contradicted,
        test_05_unverifiable,
        test_06_mixed,
        test_07_no_relevant_info,
        test_08_no_external_knowledge,
        test_09_binary_no_mirror,
        test_10_recommendation_leakage,
        test_11_provenance,
        test_12_empty_options,
    ]

    results: list[TestResult] = []
    for fn in tests:
        print(f"Running {fn.__name__}...", flush=True)
        results.append(await fn())

    matrix_summary = []
    systematic_omission = False
    for r in results:
        if r.expected_pairs > 0:
            matrix_summary.append({
                "test": r.name,
                "expected": r.expected_pairs,
                "llm_returned": r.llm_returned_rows,
                "validated": r.validated_pairs,
                "missing": r.missing_pairs,
            })
            if r.missing_pairs and r.expected_pairs >= 4:
                systematic_omission = True

    isolated_pass = all(r.pass_ for r in results)
    e2e_result = None
    if run_e2e and isolated_pass:
        print("Running optional E2E...", flush=True)
        try:
            e2e_result = await optional_e2e()
        except Exception as exc:
            e2e_result = {"error": str(exc)}

    report = {
        "isolated_results": [_serialize_result(r) for r in results],
        "matrix_coverage_summary": matrix_summary,
        "systematic_pair_omission_detected": systematic_omission,
        "isolated_pass_count": sum(1 for r in results if r.pass_),
        "isolated_total": len(results),
        "optional_e2e": e2e_result,
        "phase_3b_live_validation": "PASS" if isolated_pass else "FAIL",
        "ready_to_freeze": "YES" if isolated_pass and not systematic_omission else "NO",
    }
    return report


if __name__ == "__main__":
    run_e2e = "--no-e2e" not in sys.argv
    report = asyncio.run(main(run_e2e=run_e2e))
    print(json.dumps(report, indent=2))
