"""Evidence-grounded option evaluation (Phase 3B)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from domain.models import Claim
from services.decision_framing_schemas import DecisionFrame
from services.knowledge_state_schemas import KnowledgeState, KnowledgeStateEntry
from services.option_evaluation_schemas import (
    ClaimCatalogEntry,
    CriterionAssessment,
    CriterionEvaluation,
    CriterionEvaluationLLM,
    KnowledgeCoverage,
    OptionEvaluation,
    OptionEvaluationEntry,
    OptionEvaluationLLMOutput,
    OptionEvaluationMetrics,
)
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

_DIRECTIONAL = frozenset({
    CriterionAssessment.FAVORABLE,
    CriterionAssessment.UNFAVORABLE,
})
_SUPPORT_BUCKETS = frozenset({"known", "likely"})
_WEAK_BUCKETS = frozenset({"disputed", "unknown", "contradicted", "unverifiable"})

_RECOMMENDATION_LEAKAGE = re.compile(
    r"\b("
    r"we recommend|i recommend|our recommendation|"
    r"the best choice|best choice is|you should choose|"
    r"should choose|the winner is|recommend choosing|"
    r"we should (?:choose|select|pick|go with)"
    r")\b",
    re.IGNORECASE,
)

_BUCKET_FIELDS = (
    "known",
    "likely",
    "disputed",
    "unknown",
    "contradicted",
    "unverifiable",
)

_SCOPE_API_PRICING = re.compile(
    r"\b(api|token|per[\s-]?million|per[\s-]?1m|inference|pricing page|price per)\b",
    re.IGNORECASE,
)
_SCOPE_SUBSCRIPTION = re.compile(
    r"\b(subscription|monthly plan|claude code|consumer plan|pro plan|seat)\b",
    re.IGNORECASE,
)

_EVALUATION_SYSTEM_PROMPT = """You are an evidence-grounded option evaluator for a research system.

You receive:
1. A DecisionFrame (decision, options, criteria with labels only)
2. A claim catalog from verified research (claim_id, epistemic bucket, status, claim text)

Your job: for EACH combination of option × criterion in the DecisionFrame, assess what the
supplied claims imply for that option on that criterion.

RULES:
- Use ONLY claim_ids from the catalog. Do not invent facts or use outside knowledge.
- Output one evaluation row per option × criterion pair.
- option_label and criterion_label must EXACTLY match labels in the DecisionFrame.
- assessment must be one of: favorable, unfavorable, mixed, neutral, uncertain, insufficient_information
- favorable/unfavorable: only when known/likely claims support a clear directional implication
- mixed: usable claims point in opposing directions on the same criterion
- neutral: relevant claims exist but no clear positive/negative direction
- uncertain: relevant claims exist but are disputed/conflicted/low-confidence
- insufficient_information: no usable relevant claims for this option×criterion
- reason: brief explanation of the implication from cited claims — NO recommendation language
- Do NOT say "we recommend", "best choice", "winner", "you should choose", or rank options
- Evaluate each option independently — do not mirror assessments across binary options
- When comparing options on a criterion, do NOT infer relative superiority from evidence that is not materially comparable in scope, product, unit, or use case (e.g. API token pricing vs consumer subscription plans). If comparable evidence is unavailable for one option, prefer uncertain or insufficient_information rather than directional comparative language.
- Do NOT add options or criteria not in the DecisionFrame"""


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def build_claim_catalog(
    knowledge_state: KnowledgeState,
    material_claims: list[Claim],
) -> dict[int, ClaimCatalogEntry]:
    """Full material knowledge catalog with epistemic bucket per claim."""
    claim_text = {c.id: c.text for c in material_claims if c.id is not None}
    catalog: dict[int, ClaimCatalogEntry] = {}

    for bucket in _BUCKET_FIELDS:
        entries: list[KnowledgeStateEntry] = getattr(knowledge_state, bucket, [])
        for entry in entries:
            if entry.claim_id in catalog:
                continue
            kc = entry.knowledge_category.value if entry.knowledge_category else None
            catalog[entry.claim_id] = ClaimCatalogEntry(
                claim_id=entry.claim_id,
                bucket=bucket,
                verification_id=entry.verification_id,
                verification_status=entry.verification_status.value,
                knowledge_category=kc,
                claim_text=claim_text.get(entry.claim_id, ""),
            )

    return catalog


def format_claim_catalog(catalog: dict[int, ClaimCatalogEntry]) -> str:
    if not catalog:
        return "(No claims in knowledge state.)"
    lines = []
    for cid in sorted(catalog):
        e = catalog[cid]
        kc = e.knowledge_category or e.bucket
        text = e.claim_text[:500] if e.claim_text else "(no text)"
        lines.append(
            f"Claim {cid} [{e.bucket}, {e.verification_status}, category={kc}]: {text}"
        )
    return "\n".join(lines)


def format_decision_frame_for_prompt(frame: DecisionFrame) -> str:
    options = "\n".join(f"- {o.label} (origin={o.origin})" for o in frame.options)
    criteria = "\n".join(
        f"- {c.label} (origin={c.origin}, priority={c.priority})" for c in frame.criteria
    )
    return (
        f"Decision: {frame.decision}\n"
        f"Type: {frame.decision_type.value}\n"
        f"Options:\n{options or '(none)'}\n"
        f"Criteria:\n{criteria or '(none)'}\n"
        f"Constraints: {frame.constraints or []}\n"
        f"Missing decision context: {frame.missing_decision_context or []}"
    )


def _has_recommendation_leakage(reason: str) -> bool:
    return bool(_RECOMMENDATION_LEAKAGE.search(reason))


def _buckets_for_claims(
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> set[str]:
    return {catalog[cid].bucket for cid in claim_ids if cid in catalog}


def _compute_knowledge_coverage(
    assessment: CriterionAssessment,
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> KnowledgeCoverage:
    if assessment == CriterionAssessment.INSUFFICIENT_INFORMATION or not claim_ids:
        return KnowledgeCoverage.INSUFFICIENT
    buckets = _buckets_for_claims(claim_ids, catalog)
    if buckets & _SUPPORT_BUCKETS:
        return KnowledgeCoverage.GROUNDED
    if buckets:
        return KnowledgeCoverage.PARTIAL
    return KnowledgeCoverage.INSUFFICIENT


def _downgrade_assessment(
    assessment: CriterionAssessment,
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> CriterionAssessment:
    if not claim_ids:
        return CriterionAssessment.INSUFFICIENT_INFORMATION

    buckets = _buckets_for_claims(claim_ids, catalog)
    if not buckets:
        return CriterionAssessment.INSUFFICIENT_INFORMATION

    if buckets == {"contradicted"}:
        return CriterionAssessment.INSUFFICIENT_INFORMATION

    if assessment not in _DIRECTIONAL:
        return assessment

    if buckets <= _WEAK_BUCKETS:
        if buckets == {"unknown"}:
            return CriterionAssessment.INSUFFICIENT_INFORMATION
        return CriterionAssessment.UNCERTAIN

    if assessment in _DIRECTIONAL and not (buckets & _SUPPORT_BUCKETS):
        if "disputed" in buckets:
            return CriterionAssessment.UNCERTAIN
        return CriterionAssessment.UNCERTAIN

    return assessment


def _scope_tags_for_claims(
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> set[str]:
    tags: set[str] = set()
    for cid in claim_ids:
        entry = catalog.get(cid)
        if not entry or not entry.claim_text:
            continue
        text = entry.claim_text.lower()
        if _SCOPE_API_PRICING.search(text):
            tags.add("api_pricing")
        if _SCOPE_SUBSCRIPTION.search(text):
            tags.add("subscription")
    return tags


def _apply_cross_option_scope_guard(
    by_option: dict[str, OptionEvaluationEntry],
    criterion_label: str,
    catalog: dict[int, ClaimCatalogEntry],
) -> None:
    """
    Downgrade directional assessments when options cite obviously non-comparable evidence scopes.
    """
    crit_norm = _normalize_label(criterion_label)
    scoped: dict[str, tuple[CriterionEvaluation, set[str]]] = {}
    for entry in by_option.values():
        ce = next(
            (row for row in entry.criteria_evaluations if _normalize_label(row.criterion_label) == crit_norm),
            None,
        )
        if ce is None:
            continue
        scoped[entry.option_label] = (ce, _scope_tags_for_claims(ce.claim_ids, catalog))

    if len(scoped) < 2:
        return

    all_tags = set().union(*(tags for _, tags in scoped.values()))
    if "api_pricing" not in all_tags or "subscription" not in all_tags:
        return

    directional_opts = {
        opt
        for opt, (ce, tags) in scoped.items()
        if ce.assessment in _DIRECTIONAL and tags
    }
    api_opts = {opt for opt, (_, tags) in scoped.items() if "api_pricing" in tags}
    sub_opts = {opt for opt, (_, tags) in scoped.items() if "subscription" in tags}
    if not directional_opts or not api_opts or not sub_opts:
        return
    if api_opts == sub_opts:
        return

    for opt in directional_opts:
        ce, _ = scoped[opt]
        if ce.assessment in _DIRECTIONAL:
            ce.assessment = CriterionAssessment.UNCERTAIN
            ce.knowledge_coverage = KnowledgeCoverage.PARTIAL


def _derive_lineage(
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> tuple[list[int], list[str]]:
    verification_ids: list[int] = []
    categories: list[str] = []
    seen_v: set[int] = set()
    seen_c: set[str] = set()
    for cid in claim_ids:
        entry = catalog.get(cid)
        if not entry:
            continue
        if entry.verification_id is not None and entry.verification_id not in seen_v:
            verification_ids.append(entry.verification_id)
            seen_v.add(entry.verification_id)
        cat = entry.knowledge_category or entry.bucket
        if cat not in seen_c:
            categories.append(cat)
            seen_c.add(cat)
    return verification_ids, categories


def _frame_option_map(frame: DecisionFrame) -> dict[str, tuple[str, str]]:
    return {_normalize_label(o.label): (o.label, o.origin) for o in frame.options}


def _frame_criterion_map(frame: DecisionFrame) -> dict[str, tuple[str, str, str]]:
    return {
        _normalize_label(c.label): (c.label, c.origin, c.priority) for c in frame.criteria
    }


def validate_and_build_evaluation(
    llm_output: OptionEvaluationLLMOutput,
    frame: DecisionFrame,
    catalog: dict[int, ClaimCatalogEntry],
    *,
    reject_on_leakage: bool = True,
) -> tuple[OptionEvaluation | None, OptionEvaluationMetrics]:
    """Validate LLM rows and assemble OptionEvaluation."""
    metrics = OptionEvaluationMetrics(
        option_count=len(frame.options),
        criterion_count=len(frame.criteria),
        catalog_claim_count=len(catalog),
    )
    option_map = _frame_option_map(frame)
    criterion_map = _frame_criterion_map(frame)
    expected_pairs = len(frame.options) * len(frame.criteria)

    by_option: dict[str, OptionEvaluationEntry] = {}
    valid_rows = 0
    contamination_rows = 0

    for row in llm_output.evaluations:
        opt_key = _normalize_label(row.option_label)
        crit_key = _normalize_label(row.criterion_label)

        if opt_key not in option_map or crit_key not in criterion_map:
            metrics.rejected_row_count += 1
            continue

        if _has_recommendation_leakage(row.reason):
            metrics.rejected_row_count += 1
            contamination_rows += 1
            continue

        opt_label, opt_origin = option_map[opt_key]
        crit_label, crit_origin, crit_priority = criterion_map[crit_key]

        valid_claim_ids: list[int] = []
        for cid in row.claim_ids:
            if cid in catalog:
                valid_claim_ids.append(cid)
            else:
                metrics.invalid_reference_count += 1

        assessment = _downgrade_assessment(row.assessment, valid_claim_ids, catalog)
        coverage = _compute_knowledge_coverage(assessment, valid_claim_ids, catalog)
        verification_ids, categories = _derive_lineage(valid_claim_ids, catalog)

        if opt_label not in by_option:
            by_option[opt_label] = OptionEvaluationEntry(
                option_label=opt_label,
                option_origin=opt_origin,
            )

        by_option[opt_label].criteria_evaluations.append(
            CriterionEvaluation(
                criterion_label=crit_label,
                criterion_origin=crit_origin,
                criterion_priority=crit_priority,
                assessment=assessment,
                knowledge_coverage=coverage,
                claim_ids=valid_claim_ids,
                verification_ids=verification_ids,
                knowledge_categories=categories,
                reason=row.reason.strip(),
            )
        )
        metrics.referenced_claim_count += len(valid_claim_ids)
        valid_rows += 1

        if coverage == KnowledgeCoverage.GROUNDED:
            metrics.grounded_evaluation_count += 1
        elif coverage == KnowledgeCoverage.PARTIAL:
            metrics.partial_evaluation_count += 1
        else:
            metrics.insufficient_evaluation_count += 1

    metrics.evaluations_generated = valid_rows

    if contamination_rows > 0 and valid_rows == 0:
        metrics.evaluation_failed = True
        metrics.failure_reason = "recommendation_leakage"
        return None, metrics

    if expected_pairs > 0 and valid_rows == 0:
        metrics.evaluation_failed = True
        metrics.failure_reason = "no_valid_evaluations"
        return None, metrics

    if reject_on_leakage and contamination_rows > expected_pairs // 2:
        metrics.evaluation_failed = True
        metrics.failure_reason = "material_recommendation_contamination"
        return None, metrics

    for crit_label, _, _ in criterion_map.values():
        _apply_cross_option_scope_guard(by_option, crit_label, catalog)

    evaluation = OptionEvaluation(
        decision=frame.decision,
        option_evaluations=list(by_option.values()),
        decision_limitations=list(frame.missing_decision_context),
        constraints=list(frame.constraints),
    )
    return evaluation, metrics


def skip_metrics(reason: str) -> OptionEvaluationMetrics:
    return OptionEvaluationMetrics(
        evaluation_skipped=True,
        evaluation_skipped_reason=reason,
    )


async def evaluate_options(
    decision_frame: DecisionFrame,
    knowledge_state: KnowledgeState,
    material_claims: list[Claim],
    *,
    llm: Any | None = None,
) -> tuple[OptionEvaluation | None, OptionEvaluationMetrics]:
    """
    Map verified knowledge to option×criterion evaluations.

    Fails open on error. Skips cleanly when no concrete options.
    """
    start = time.monotonic()

    if not decision_frame.options:
        metrics = skip_metrics("no_concrete_options")
        metrics.evaluation_time_ms = (time.monotonic() - start) * 1000
        return None, metrics

    catalog = build_claim_catalog(knowledge_state, material_claims)
    metrics = OptionEvaluationMetrics(
        option_count=len(decision_frame.options),
        criterion_count=len(decision_frame.criteria),
        catalog_claim_count=len(catalog),
    )

    if llm is None:
        from langchain_anthropic import ChatAnthropic

        from config import settings

        llm = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        )

    user_prompt = (
        f"DECISION FRAME:\n{format_decision_frame_for_prompt(decision_frame)}\n\n"
        f"CLAIM CATALOG:\n{format_claim_catalog(catalog)}\n\n"
        "Produce one evaluation row for every option × criterion pair."
    )

    try:
        with trace_llm_call("option_evaluator", "evaluate_options") as span:
            span.set_input({
                "decision": decision_frame.decision[:200],
                "option_count": len(decision_frame.options),
                "criterion_count": len(decision_frame.criteria),
                "catalog_claim_count": len(catalog),
            })
            structured = llm.with_structured_output(OptionEvaluationLLMOutput)
            llm_output: OptionEvaluationLLMOutput = await structured.ainvoke(
                [
                    {"role": "system", "content": _EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            span.set_output({"row_count": len(llm_output.evaluations)})

        metrics.evaluation_llm_calls = 1
        evaluation, val_metrics = validate_and_build_evaluation(
            llm_output, decision_frame, catalog
        )
        metrics.evaluations_generated = val_metrics.evaluations_generated
        metrics.grounded_evaluation_count = val_metrics.grounded_evaluation_count
        metrics.partial_evaluation_count = val_metrics.partial_evaluation_count
        metrics.insufficient_evaluation_count = val_metrics.insufficient_evaluation_count
        metrics.referenced_claim_count = val_metrics.referenced_claim_count
        metrics.invalid_reference_count = val_metrics.invalid_reference_count
        metrics.rejected_row_count = val_metrics.rejected_row_count
        metrics.evaluation_failed = val_metrics.evaluation_failed
        metrics.failure_reason = val_metrics.failure_reason
        metrics.evaluation_time_ms = (time.monotonic() - start) * 1000

        if val_metrics.evaluation_failed:
            return None, metrics
        return evaluation, metrics

    except Exception as exc:
        logger.warning("Option evaluation failed open: %s", exc)
        metrics.evaluation_failed = True
        metrics.failure_reason = str(exc)
        metrics.evaluation_time_ms = (time.monotonic() - start) * 1000
        metrics.evaluation_llm_calls = 1
        return None, metrics
