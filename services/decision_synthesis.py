"""Evidence-grounded decision synthesis (Phase 3C)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from domain.models import Claim
from services.decision_framing_schemas import DecisionFrame
from services.decision_synthesis_schemas import (
    ChangeCondition,
    ConstraintAssessment,
    ConstraintCompliance,
    CriterionReference,
    DecisionSynthesis,
    DecisionSynthesisLLMOutput,
    DecisionSynthesisMetrics,
    RecommendationStatus,
    SynthesisPreCheck,
)
from services.knowledge_state_schemas import KnowledgeState
from services.option_evaluation import build_claim_catalog, format_claim_catalog
from services.option_evaluation_schemas import (
    ClaimCatalogEntry,
    CriterionAssessment,
    KnowledgeCoverage,
    OptionEvaluation,
)
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

_STATUS_RANK = {
    RecommendationStatus.INSUFFICIENT_BASIS: 0,
    RecommendationStatus.TENTATIVE_RECOMMENDATION: 1,
    RecommendationStatus.RECOMMEND: 2,
}

_SUPPORT_BUCKETS = frozenset({"known", "likely"})
_WEAK_BUCKETS = frozenset({"disputed", "unknown", "contradicted", "unverifiable"})
_DIRECTIONAL_COMPLIANCE = frozenset({ConstraintCompliance.SATISFIED, ConstraintCompliance.VIOLATED})

_NUMERIC_TOKEN = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s*(?:k|m|b|million|billion))?"
    r"|\b\d+(?:\.\d+)?%"
    r"|\b\d+(?:\.\d+)?\s*(?:months?|years?|days?|weeks?)\b",
    re.IGNORECASE,
)

_SYNTHESIS_SYSTEM_PROMPT = """You are an evidence-grounded decision synthesizer.

You receive:
1. A DecisionFrame (decision, options, criteria with origin/priority, constraints, assumptions, missing context)
2. A complete OptionEvaluation matrix (option × criterion assessments with claim lineage)
3. A trusted claim catalog for HARD CONSTRAINT assessment only

Your job:
- Determine which option, if any, is best supported by the structured evaluations
- Identify what could materially change that recommendation
- Assess EVERY option × constraint pair for hard constraint compliance

DECISION HIERARCHY (qualitative, not numeric):
1. Hard constraints — must be satisfied for full recommendation
2. Explicit primary criteria
3. Explicit standard criteria
4. Inferred criteria (secondary; cannot override unresolved primary explicit criteria)

RECOMMENDATION STATUS:
- recommend: clear support, constraints satisfied/established, matrix complete, no major blockers
- tentative_recommendation: defensible lean with material uncertainty (unestablished constraints, uncertain primary criteria, competing strengths, critical missing context)
- insufficient_basis: cannot responsibly select; set recommended_option=null

RULES:
- Use ONLY options, criteria, constraints, assumptions, and missing-context items from the DecisionFrame
- Criterion references: option_label + criterion_label only (metadata copied by validator)
- Constraint assessments: cite claim_ids ONLY from the supplied claim catalog
- Produce one constraint_assessment row per option × constraint
- critical_missing_context: ONLY exact items from missing_decision_context list
- assumptions_relied_on: ONLY exact items from explicit_assumptions list
- change_conditions: traceable to criteria, constraints, assumptions, missing context, or claim_ids
- Do NOT invent options, criteria, claims, thresholds, weights, scores, or new facts
- Do NOT use outside knowledge
- Inferred criteria must not override unfavorable/unresolved primary explicit criteria
- Absence of proof is NOT constraint violation — use not_established
- Do NOT treat one option's favorable evidence as comparative dominance when competing options lack comparable evidence on the same primary criterion (comparative coverage gap)"""


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _normalize_constraint(text: str) -> str:
    return " ".join(text.strip().split())


def _cap_status(
    status: RecommendationStatus,
    ceiling: RecommendationStatus,
) -> RecommendationStatus:
    if _STATUS_RANK[status] > _STATUS_RANK[ceiling]:
        return ceiling
    return status


def build_oe_index(
    option_evaluation: OptionEvaluation,
) -> dict[tuple[str, str], CriterionReference]:
    index: dict[tuple[str, str], CriterionReference] = {}
    for opt in option_evaluation.option_evaluations:
        for ce in opt.criteria_evaluations:
            key = (_normalize_label(opt.option_label), _normalize_label(ce.criterion_label))
            index[key] = CriterionReference(
                option_label=opt.option_label,
                criterion_label=ce.criterion_label,
                criterion_origin=ce.criterion_origin,
                criterion_priority=ce.criterion_priority,
                assessment=ce.assessment,
                knowledge_coverage=ce.knowledge_coverage,
                claim_ids=list(ce.claim_ids),
            )
    return index


def count_evaluation_pairs(option_evaluation: OptionEvaluation) -> int:
    return sum(len(o.criteria_evaluations) for o in option_evaluation.option_evaluations)


def detect_comparative_coverage_gaps(
    frame: DecisionFrame,
    oe_index: dict[tuple[str, str], CriterionReference],
) -> list[str]:
    """
    Criteria where grounded directional evidence exists for some options but not others.

    Absolute evidence for one option must not imply comparative dominance.
    """
    gaps: list[str] = []
    for crit in frame.criteria:
        if crit.origin != "explicit":
            continue
        crit_norm = _normalize_label(crit.label)
        rows = [row for (_, c_key), row in oe_index.items() if c_key == crit_norm]
        if len(rows) < 2:
            continue

        grounded_directional = [
            row
            for row in rows
            if row.knowledge_coverage == KnowledgeCoverage.GROUNDED
            and row.assessment
            in (
                CriterionAssessment.FAVORABLE,
                CriterionAssessment.UNFAVORABLE,
                CriterionAssessment.MIXED,
            )
        ]
        weak_or_missing = [
            row
            for row in rows
            if row.assessment == CriterionAssessment.INSUFFICIENT_INFORMATION
            or row.knowledge_coverage == KnowledgeCoverage.INSUFFICIENT
        ]
        if grounded_directional and weak_or_missing:
            gaps.append(crit.label)
    return gaps


def _apply_comparative_coverage_ceiling(
    frame: DecisionFrame,
    comparative_coverage_gaps: list[str],
    ceiling: RecommendationStatus,
) -> RecommendationStatus:
    if not comparative_coverage_gaps:
        return ceiling

    explicit_primary = [
        c.label for c in frame.criteria if c.origin == "explicit" and c.priority == "primary"
    ]
    if not explicit_primary:
        return ceiling

    primary_gaps = [label for label in comparative_coverage_gaps if label in explicit_primary]
    if not primary_gaps:
        return ceiling

    ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)
    if set(explicit_primary) <= set(primary_gaps):
        return RecommendationStatus.INSUFFICIENT_BASIS
    return ceiling


def run_pre_check(frame: DecisionFrame, option_evaluation: OptionEvaluation) -> SynthesisPreCheck:
    expected = len(frame.options) * len(frame.criteria)
    actual = count_evaluation_pairs(option_evaluation)
    matrix_complete = expected > 0 and actual >= expected

    pre = SynthesisPreCheck(
        matrix_complete=matrix_complete,
        expected_pairs=expected,
        actual_pairs=actual,
        option_count=len(frame.options),
        primary_criterion_count=sum(1 for c in frame.criteria if c.priority == "primary"),
        explicit_criterion_count=sum(1 for c in frame.criteria if c.origin == "explicit"),
        inferred_criterion_count=sum(1 for c in frame.criteria if c.origin == "inferred"),
        constraint_count=len(frame.constraints),
        missing_context_count=len(frame.missing_decision_context),
    )

    ceiling = RecommendationStatus.RECOMMEND
    if not matrix_complete:
        pre.blockers.append("incomplete_evaluation_matrix")
        ceiling = RecommendationStatus.INSUFFICIENT_BASIS

    pre.status_ceiling = ceiling
    return pre


def _insufficient_basis_artifact(
    frame: DecisionFrame,
    option_evaluation: OptionEvaluation,
    *,
    rationale: str,
) -> DecisionSynthesis:
    return DecisionSynthesis(
        decision=frame.decision,
        recommendation_status=RecommendationStatus.INSUFFICIENT_BASIS,
        recommended_option=None,
        rationale=rationale,
        decision_limitations=list(option_evaluation.decision_limitations or frame.missing_decision_context),
    )


def _buckets_for_claims(
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> set[str]:
    return {catalog[cid].bucket for cid in claim_ids if cid in catalog}


def _downgrade_constraint_compliance(
    compliance: ConstraintCompliance,
    claim_ids: list[int],
    catalog: dict[int, ClaimCatalogEntry],
) -> ConstraintCompliance:
    if compliance == ConstraintCompliance.NOT_ESTABLISHED or not claim_ids:
        return ConstraintCompliance.NOT_ESTABLISHED

    buckets = _buckets_for_claims(claim_ids, catalog)
    if not buckets:
        return ConstraintCompliance.NOT_ESTABLISHED

    if buckets == {"contradicted"} or buckets == {"unknown"}:
        return ConstraintCompliance.NOT_ESTABLISHED

    if compliance in _DIRECTIONAL_COMPLIANCE:
        if buckets <= _WEAK_BUCKETS:
            return ConstraintCompliance.NOT_ESTABLISHED
        if not (buckets & _SUPPORT_BUCKETS):
            return ConstraintCompliance.NOT_ESTABLISHED

    return compliance


def _collect_trusted_numeric_tokens(
    frame: DecisionFrame,
    catalog: dict[int, ClaimCatalogEntry],
) -> set[str]:
    tokens: set[str] = set()
    texts = list(frame.constraints) + list(frame.explicit_assumptions) + list(frame.missing_decision_context)
    for entry in catalog.values():
        if entry.claim_text:
            texts.append(entry.claim_text)
    for text in texts:
        for m in _NUMERIC_TOKEN.finditer(text):
            tokens.add(m.group(0).lower())
    return tokens


def _has_fabricated_threshold(text: str, trusted: set[str]) -> bool:
    for m in _NUMERIC_TOKEN.finditer(text):
        if m.group(0).lower() not in trusted:
            return True
    return False


def _resolve_criterion_refs(
    refs: list,
    oe_index: dict[tuple[str, str], CriterionReference],
) -> list[CriterionReference]:
    resolved: list[CriterionReference] = []
    for ref in refs:
        key = (_normalize_label(ref.option_label), _normalize_label(ref.criterion_label))
        if key in oe_index:
            resolved.append(oe_index[key])
    return resolved


def _match_frame_string(value: str, candidates: list[str]) -> str | None:
    norm = _normalize_constraint(value)
    for c in candidates:
        if _normalize_constraint(c) == norm:
            return c
    return None


def _missing_constraint_pairs(
    frame: DecisionFrame,
    constraint_assessments: list[ConstraintAssessment],
) -> list[str]:
    present = {
        (_normalize_label(a.option_label), _normalize_constraint(a.constraint))
        for a in constraint_assessments
    }
    missing: list[str] = []
    for option in frame.options:
        for constraint in frame.constraints:
            key = (_normalize_label(option.label), _normalize_constraint(constraint))
            if key not in present:
                missing.append(f"{option.label} × {constraint}")
    return missing


def compute_status_ceiling(
    frame: DecisionFrame,
    pre: SynthesisPreCheck,
    oe_index: dict[tuple[str, str], CriterionReference],
    *,
    recommended_option: str | None,
    constraint_assessments: list[ConstraintAssessment],
    supporting: list[CriterionReference],
    limiting: list[CriterionReference],
    critical_missing_context: list[str],
    constraint_matrix_complete: bool = True,
    comparative_coverage_gaps: list[str] | None = None,
) -> RecommendationStatus:
    ceiling = pre.status_ceiling

    if not constraint_matrix_complete:
        return RecommendationStatus.INSUFFICIENT_BASIS

    ceiling = _apply_comparative_coverage_ceiling(
        frame,
        comparative_coverage_gaps or [],
        ceiling,
    )

    if critical_missing_context:
        ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)

    if recommended_option:
        opt_key = _normalize_label(recommended_option)
        for ca in constraint_assessments:
            if _normalize_label(ca.option_label) != opt_key:
                continue
            if ca.compliance == ConstraintCompliance.VIOLATED:
                return RecommendationStatus.INSUFFICIENT_BASIS
            if ca.compliance == ConstraintCompliance.NOT_ESTABLISHED:
                ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)

        primary_rows = [
            row for (opt_key_row, _), row in oe_index.items()
            if row.criterion_priority == "primary" and opt_key_row == opt_key
        ]
        if not primary_rows:
            primary_rows = [
                r for r in supporting + limiting
                if r.criterion_priority == "primary" and _normalize_label(r.option_label) == opt_key
            ]
        for row in primary_rows:
            if row.assessment in (
                CriterionAssessment.UNCERTAIN,
                CriterionAssessment.INSUFFICIENT_INFORMATION,
                CriterionAssessment.MIXED,
            ):
                if row.assessment == CriterionAssessment.INSUFFICIENT_INFORMATION:
                    ceiling = _cap_status(ceiling, RecommendationStatus.INSUFFICIENT_BASIS)
                else:
                    ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)

        unfavorable_primary = [
            r for r in limiting
            if r.criterion_priority == "primary"
            and _normalize_label(r.option_label) == opt_key
            and r.assessment == CriterionAssessment.UNFAVORABLE
            and r.knowledge_coverage == KnowledgeCoverage.GROUNDED
        ]
        if unfavorable_primary:
            ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)

        explicit_standard_labels = {
            c.label for c in frame.criteria if c.origin == "explicit" and c.priority == "standard"
        }
        if len(explicit_standard_labels) >= 2:
            opt_favorable = {
                r.criterion_label for r in supporting
                if _normalize_label(r.option_label) == opt_key
                and r.criterion_label in explicit_standard_labels
                and r.assessment == CriterionAssessment.FAVORABLE
                and r.knowledge_coverage == KnowledgeCoverage.GROUNDED
            }
            opt_unfavorable = {
                r.criterion_label for r in limiting
                if _normalize_label(r.option_label) == opt_key
                and r.criterion_label in explicit_standard_labels
                and r.assessment == CriterionAssessment.UNFAVORABLE
                and r.knowledge_coverage == KnowledgeCoverage.GROUNDED
            }
            if opt_favorable and opt_unfavorable:
                ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)
            for other in frame.options:
                if _normalize_label(other.label) == opt_key:
                    continue
                other_fav = {
                    r.criterion_label for r in supporting
                    if _normalize_label(r.option_label) == _normalize_label(other.label)
                    and r.criterion_label in explicit_standard_labels
                    and r.assessment == CriterionAssessment.FAVORABLE
                    and r.knowledge_coverage == KnowledgeCoverage.GROUNDED
                }
                if opt_favorable & other_fav and opt_unfavorable:
                    ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)
                split_win = bool(opt_favorable) and bool(other_fav) and opt_favorable != other_fav
                if split_win and len(explicit_standard_labels) >= 2:
                    ceiling = _cap_status(ceiling, RecommendationStatus.TENTATIVE_RECOMMENDATION)

    if not pre.matrix_complete:
        ceiling = RecommendationStatus.INSUFFICIENT_BASIS

    return ceiling


def validate_and_build_synthesis(
    llm_output: DecisionSynthesisLLMOutput,
    frame: DecisionFrame,
    option_evaluation: OptionEvaluation,
    catalog: dict[int, ClaimCatalogEntry],
    pre: SynthesisPreCheck,
) -> tuple[DecisionSynthesis | None, DecisionSynthesisMetrics, list[str]]:
    """Validate LLM synthesis and assemble DecisionSynthesis."""
    metrics = DecisionSynthesisMetrics(
        matrix_complete=pre.matrix_complete,
        expected_pairs=pre.expected_pairs,
        actual_pairs=pre.actual_pairs,
        primary_criterion_count=pre.primary_criterion_count,
        constraint_count=pre.constraint_count,
    )
    errors: list[str] = []
    oe_index = build_oe_index(option_evaluation)
    option_map = {_normalize_label(o.label): o.label for o in frame.options}
    criterion_map = {_normalize_label(c.label): c.label for c in frame.criteria}
    constraint_set = {_normalize_constraint(c): c for c in frame.constraints}
    trusted_numeric = _collect_trusted_numeric_tokens(frame, catalog)

    supporting = _resolve_criterion_refs(llm_output.supporting_criteria, oe_index)
    limiting = _resolve_criterion_refs(llm_output.limiting_criteria, oe_index)

    constraint_assessments: list[ConstraintAssessment] = []
    expected_constraint_rows = len(frame.options) * len(frame.constraints)
    for row in llm_output.constraint_assessments:
        opt_key = _normalize_label(row.option_label)
        con_key = _normalize_constraint(row.constraint)
        if opt_key not in option_map or con_key not in constraint_set:
            errors.append(f"invalid_constraint_row:{row.option_label}/{row.constraint}")
            continue

        valid_claim_ids = [cid for cid in row.claim_ids if cid in catalog]
        invalid = len(row.claim_ids) - len(valid_claim_ids)
        if invalid:
            errors.append(f"invalid_constraint_claim_ids:{invalid}")

        compliance = _downgrade_constraint_compliance(
            row.compliance, valid_claim_ids, catalog
        )
        constraint_assessments.append(
            ConstraintAssessment(
                option_label=option_map[opt_key],
                constraint=constraint_set[con_key],
                compliance=compliance,
                claim_ids=valid_claim_ids,
                reason=row.reason.strip(),
            )
        )
        if compliance == ConstraintCompliance.VIOLATED:
            metrics.constraint_violation_count += 1
        elif compliance == ConstraintCompliance.NOT_ESTABLISHED:
            metrics.constraint_not_established_count += 1

    if frame.constraints and len(constraint_assessments) < expected_constraint_rows:
        errors.append(
            f"incomplete_constraint_matrix:{len(constraint_assessments)}/{expected_constraint_rows}"
        )

    missing_constraint_pairs = _missing_constraint_pairs(frame, constraint_assessments)
    constraint_matrix_complete = not missing_constraint_pairs
    if frame.constraints and not constraint_matrix_complete:
        errors.append(
            f"missing_constraint_pairs:{missing_constraint_pairs}"
        )

    critical_missing: list[str] = []
    for item in llm_output.critical_missing_context:
        matched = _match_frame_string(item, frame.missing_decision_context)
        if matched:
            critical_missing.append(matched)

    assumptions: list[str] = []
    for item in llm_output.assumptions_relied_on:
        matched = _match_frame_string(item, frame.explicit_assumptions)
        if matched:
            assumptions.append(matched)

    change_conditions: list[ChangeCondition] = []
    for cc in llm_output.change_conditions:
        if _has_fabricated_threshold(cc.description, trusted_numeric):
            errors.append("fabricated_threshold_in_change_condition")
            continue
        if cc.related_option_label:
            if _normalize_label(cc.related_option_label) not in option_map:
                errors.append("invalid_change_option")
                continue
        if cc.related_criterion_label:
            if _normalize_label(cc.related_criterion_label) not in criterion_map:
                errors.append("invalid_change_criterion")
                continue
        if cc.related_constraint:
            if _normalize_constraint(cc.related_constraint) not in constraint_set:
                errors.append("invalid_change_constraint")
                continue
        if cc.related_assumption:
            if not _match_frame_string(cc.related_assumption, frame.explicit_assumptions):
                errors.append("invalid_change_assumption")
                continue
        if cc.related_missing_context:
            if not _match_frame_string(cc.related_missing_context, frame.missing_decision_context):
                errors.append("invalid_change_missing_context")
                continue
        valid_claim_ids = [cid for cid in cc.related_claim_ids if cid in catalog]
        change_conditions.append(
            ChangeCondition(
                description=cc.description.strip(),
                change_type=cc.change_type,
                related_option_label=(
                    option_map.get(_normalize_label(cc.related_option_label))
                    if cc.related_option_label else None
                ),
                related_criterion_label=(
                    criterion_map.get(_normalize_label(cc.related_criterion_label))
                    if cc.related_criterion_label else None
                ),
                related_constraint=(
                    constraint_set.get(_normalize_constraint(cc.related_constraint))
                    if cc.related_constraint else None
                ),
                related_assumption=(
                    _match_frame_string(cc.related_assumption, frame.explicit_assumptions)
                    if cc.related_assumption else None
                ),
                related_missing_context=(
                    _match_frame_string(cc.related_missing_context, frame.missing_decision_context)
                    if cc.related_missing_context else None
                ),
                related_claim_ids=valid_claim_ids,
            )
        )

    recommended_option = llm_output.recommended_option
    if recommended_option:
        opt_key = _normalize_label(recommended_option)
        if opt_key not in option_map:
            errors.append("invented_recommended_option")
            recommended_option = None
        else:
            recommended_option = option_map[opt_key]

    status = llm_output.recommendation_status
    comparative_gaps = detect_comparative_coverage_gaps(frame, oe_index)
    ceiling = compute_status_ceiling(
        frame,
        pre,
        oe_index,
        recommended_option=recommended_option,
        constraint_assessments=constraint_assessments,
        supporting=supporting,
        limiting=limiting,
        critical_missing_context=critical_missing,
        constraint_matrix_complete=constraint_matrix_complete,
        comparative_coverage_gaps=comparative_gaps,
    )
    status = _cap_status(status, ceiling)

    if status == RecommendationStatus.INSUFFICIENT_BASIS:
        recommended_option = None

    if comparative_gaps and status == RecommendationStatus.RECOMMEND:
        status = RecommendationStatus.TENTATIVE_RECOMMENDATION

    metrics.supporting_criterion_count = len(supporting)
    metrics.limiting_criterion_count = len(limiting)
    metrics.critical_missing_context_count = len(critical_missing)
    metrics.assumptions_relied_on_count = len(assumptions)
    metrics.change_condition_count = len(change_conditions)
    metrics.recommendation_status = status.value
    metrics.recommendation_present = recommended_option is not None

    synthesis = DecisionSynthesis(
        decision=frame.decision,
        recommendation_status=status,
        recommended_option=recommended_option,
        rationale=llm_output.rationale.strip(),
        supporting_criteria=supporting,
        limiting_criteria=limiting,
        constraint_assessments=constraint_assessments,
        key_uncertainties=[u.strip() for u in llm_output.key_uncertainties if u.strip()],
        decision_limitations=list(option_evaluation.decision_limitations or frame.missing_decision_context),
        critical_missing_context=critical_missing,
        assumptions_relied_on=assumptions,
        change_conditions=change_conditions,
    )
    return synthesis, metrics, errors


def format_option_evaluation_for_prompt(option_evaluation: OptionEvaluation) -> str:
    lines = [f"Decision: {option_evaluation.decision}"]
    for opt in option_evaluation.option_evaluations:
        lines.append(f"\nOption: {opt.option_label} (origin={opt.option_origin})")
        for ce in opt.criteria_evaluations:
            lines.append(
                f"  - {ce.criterion_label} (origin={ce.criterion_origin}, "
                f"priority={ce.criterion_priority}): assessment={ce.assessment.value}, "
                f"coverage={ce.knowledge_coverage.value}, claim_ids={ce.claim_ids}\n"
                f"    reason: {ce.reason[:300]}"
            )
    return "\n".join(lines)


def format_frame_for_synthesis(frame: DecisionFrame) -> str:
    options = "\n".join(f"- {o.label} (origin={o.origin})" for o in frame.options)
    criteria = "\n".join(
        f"- {c.label} (origin={c.origin}, priority={c.priority})" for c in frame.criteria
    )
    return (
        f"Decision: {frame.decision}\n"
        f"Type: {frame.decision_type.value}\n"
        f"Time horizon: {frame.time_horizon}\n"
        f"Options:\n{options}\n"
        f"Criteria:\n{criteria}\n"
        f"Hard constraints:\n{frame.constraints or []}\n"
        f"Explicit assumptions:\n{frame.explicit_assumptions or []}\n"
        f"Missing decision context:\n{frame.missing_decision_context or []}"
    )


async def synthesize_decision(
    frame: DecisionFrame,
    option_evaluation: OptionEvaluation,
    knowledge_state: KnowledgeState,
    material_claims: list[Claim],
    *,
    llm: Any | None = None,
) -> tuple[DecisionSynthesis | None, DecisionSynthesisMetrics]:
    """Produce evidence-grounded decision synthesis. Fails open on error."""
    start = time.monotonic()
    pre = run_pre_check(frame, option_evaluation)
    metrics = DecisionSynthesisMetrics(
        matrix_complete=pre.matrix_complete,
        expected_pairs=pre.expected_pairs,
        actual_pairs=pre.actual_pairs,
        primary_criterion_count=pre.primary_criterion_count,
        constraint_count=pre.constraint_count,
    )

    if not pre.matrix_complete:
        artifact = _insufficient_basis_artifact(
            frame,
            option_evaluation,
            rationale=(
                f"Option evaluation matrix is incomplete "
                f"({pre.actual_pairs}/{pre.expected_pairs} pairs). "
                "Cannot responsibly recommend an option."
            ),
        )
        metrics.recommendation_status = artifact.recommendation_status.value
        metrics.recommendation_present = False
        metrics.synthesis_time_ms = (time.monotonic() - start) * 1000
        return artifact, metrics

    catalog = build_claim_catalog(knowledge_state, material_claims)

    if llm is None:
        from langchain_anthropic import ChatAnthropic

        from config import settings

        llm = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        )

    user_prompt = (
        f"DECISION FRAME:\n{format_frame_for_synthesis(frame)}\n\n"
        f"OPTION EVALUATION:\n{format_option_evaluation_for_prompt(option_evaluation)}\n\n"
        f"CLAIM CATALOG (for constraint assessment only):\n{format_claim_catalog(catalog)}\n\n"
        f"Pre-check: matrix_complete={pre.matrix_complete}, "
        f"expected_constraint_rows={len(frame.options) * len(frame.constraints)}\n"
        "Produce constraint_assessments for every option × constraint pair."
    )

    try:
        with trace_llm_call("decision_synthesizer", "synthesize_decision") as span:
            span.set_input({
                "decision": frame.decision[:200],
                "option_count": len(frame.options),
                "constraint_count": len(frame.constraints),
            })
            structured = llm.with_structured_output(DecisionSynthesisLLMOutput)
            llm_output: DecisionSynthesisLLMOutput = await structured.ainvoke(
                [
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            span.set_output(llm_output.model_dump(mode="json"))

        metrics.synthesis_llm_calls = 1
        synthesis, val_metrics, _errors = validate_and_build_synthesis(
            llm_output, frame, option_evaluation, catalog, pre
        )
        metrics.supporting_criterion_count = val_metrics.supporting_criterion_count
        metrics.limiting_criterion_count = val_metrics.limiting_criterion_count
        metrics.constraint_violation_count = val_metrics.constraint_violation_count
        metrics.constraint_not_established_count = val_metrics.constraint_not_established_count
        metrics.critical_missing_context_count = val_metrics.critical_missing_context_count
        metrics.assumptions_relied_on_count = val_metrics.assumptions_relied_on_count
        metrics.change_condition_count = val_metrics.change_condition_count
        metrics.recommendation_status = val_metrics.recommendation_status
        metrics.recommendation_present = val_metrics.recommendation_present
        metrics.synthesis_time_ms = (time.monotonic() - start) * 1000
        return synthesis, metrics

    except Exception as exc:
        logger.warning("Decision synthesis failed open: %s", exc)
        metrics.synthesis_failed = True
        metrics.failure_reason = str(exc)
        metrics.synthesis_time_ms = (time.monotonic() - start) * 1000
        metrics.synthesis_llm_calls = 1
        return None, metrics


def skip_metrics(reason: str) -> DecisionSynthesisMetrics:
    return DecisionSynthesisMetrics(
        synthesis_skipped=True,
        synthesis_skipped_reason=reason,
    )
