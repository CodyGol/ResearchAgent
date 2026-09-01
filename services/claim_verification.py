"""Cross-source verification of material claims against validated evidence."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    Evidence,
    EvidenceConfidence,
    Source,
    VerificationResult,
    VerificationStatus,
)
from services.claim_validator import validate_claim_support_deterministic
from services.claim_verification_schemas import (
    ClaimEvidenceAssessment,
    ClaimRelationshipBatchOutput,
)
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 5
_UNVERIFIABLE_TYPES = frozenset({"opinion", "predictive", "analytical"})


@dataclass
class ClaimVerificationMetrics:
    """Observability for cross-source claim verification."""

    material_claims_processed: int = 0
    cross_source_relations_added: int = 0
    deterministic_supports: int = 0
    deterministic_contradicts: int = 0
    deterministic_qualifies: int = 0
    llm_classifications: int = 0
    llm_batches: int = 0
    insufficient_evidence: int = 0
    supported: int = 0
    partially_supported: int = 0
    contradicted: int = 0
    uncertain: int = 0
    unverifiable: int = 0
    processing_time_ms: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_claims_processed": self.material_claims_processed,
            "cross_source_relations_added": self.cross_source_relations_added,
            "deterministic_supports": self.deterministic_supports,
            "deterministic_contradicts": self.deterministic_contradicts,
            "deterministic_qualifies": self.deterministic_qualifies,
            "llm_classifications": self.llm_classifications,
            "llm_batches": self.llm_batches,
            "insufficient_evidence": self.insufficient_evidence,
            "supported": self.supported,
            "partially_supported": self.partially_supported,
            "contradicted": self.contradicted,
            "uncertain": self.uncertain,
            "unverifiable": self.unverifiable,
            "processing_time_ms": self.processing_time_ms,
            "failure_count": len(self.failures),
        }


def source_publisher_domain(source: Source | None) -> str:
    """Publisher/domain key for independence checks."""
    if source is None:
        return ""
    domain = source.metadata.get("domain", "")
    if domain:
        return domain.lower().removeprefix("www.")
    try:
        return urlparse(source.url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _origin_domains(
    claim_id: int,
    relations: list[ClaimEvidenceRelation],
    evidence_by_id: dict[int, Evidence],
    sources_by_id: dict[int, Source],
) -> set[str]:
    domains: set[str] = set()
    for rel in relations:
        if rel.claim_id != claim_id or rel.relationship != ClaimEvidenceRelationship.SUPPORTS:
            continue
        ev = evidence_by_id.get(rel.evidence_id)
        if ev is None:
            continue
        domains.add(source_publisher_domain(sources_by_id.get(ev.source_id)))
    return {d for d in domains if d}


def _origin_evidence_ids(
    claim_id: int, relations: list[ClaimEvidenceRelation]
) -> set[int]:
    return {
        rel.evidence_id
        for rel in relations
        if rel.claim_id == claim_id
        and rel.relationship == ClaimEvidenceRelationship.SUPPORTS
    }


def _claim_terms(claim: Claim) -> set[str]:
    stop = frozenset({
        "the", "a", "an", "is", "was", "were", "are", "in", "of", "for", "to", "and", "or",
    })
    words = re.findall(r"[a-z0-9]+", claim.text.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def _relevance_score(claim: Claim, evidence: Evidence) -> float:
    claim_terms = _claim_terms(claim)
    if not claim_terms:
        return 0.0
    text = evidence.exact_text.lower()
    overlap = sum(1 for t in claim_terms if t in text)
    return overlap / len(claim_terms)


def select_cross_source_candidates(
    claim: Claim,
    evidence_list: list[Evidence],
    sources_by_id: dict[int, Source],
    origin_evidence_ids: set[int],
    origin_domains: set[str],
    *,
    max_candidates: int = _MAX_CANDIDATES,
) -> list[Evidence]:
    """Select up to N cross-source evidence items from independent publisher domains."""
    scored: list[tuple[float, Evidence]] = []

    for ev in evidence_list:
        if ev.id in origin_evidence_ids:
            continue
        source = sources_by_id.get(ev.source_id)
        domain = source_publisher_domain(source)
        if not domain:
            continue
        if domain in origin_domains:
            continue
        score = _relevance_score(claim, ev)
        if score < 0.15:
            continue
        scored.append((score, ev))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer one evidence item per independent domain
    seen_domains: set[str] = set()
    selected: list[Evidence] = []
    for score, ev in scored:
        domain = source_publisher_domain(sources_by_id.get(ev.source_id))
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        selected.append(ev)
        if len(selected) >= max_candidates:
            break

    return selected


def _extract_significant_numbers(text: str) -> list[str]:
    """Extract numeric values excluding standalone year tokens."""
    nums = re.findall(r"\d[\d,.]*", text)
    significant: list[str] = []
    for n in nums:
        clean = n.rstrip(".").replace(",", "")
        if re.fullmatch(r"(?:19|20)\d{2}", clean):
            continue
        if clean:
            significant.append(clean)
    return significant


def _number_in_text(num: str, text: str) -> bool:
    """Match numeric token without substring false positives (e.g. 20 in 2025)."""
    normalized = text.replace(",", "")
    return bool(re.search(rf"(?<!\d){re.escape(num)}(?!\d)", normalized))


def _numbers_aligned(claim: Claim, evidence_text: str) -> bool:
    claim_nums = _extract_significant_numbers(claim.text)
    if not claim_nums:
        return True
    for num in claim_nums:
        if not _number_in_text(num, evidence_text):
            return False
    if claim.temporal_scope:
        if claim.temporal_scope.lower() not in evidence_text.lower():
            years = re.findall(r"\b(19|20)\d{2}\b", claim.temporal_scope)
            if not any(y in evidence_text for y in years):
                return False
    return True


def _deterministic_support(claim: Claim, evidence: Evidence) -> tuple[bool, str]:
    """Conservative direct-equivalence support only."""
    claim_lower = claim.text.lower().strip().rstrip(".")
    ev_lower = evidence.exact_text.lower()

    if claim_lower in ev_lower or ev_lower in claim_lower:
        return True, "Direct proposition overlap in evidence text"

    # Quantitative alignment: same significant numbers + temporal scope
    claim_nums = _extract_significant_numbers(claim.text)
    if claim_nums and _numbers_aligned(claim, evidence.exact_text):
        det = validate_claim_support_deterministic(claim.text, evidence.exact_text)
        if det.is_supported:
            return True, "Aligned quantitative fact with matching scope"

    return False, ""


def _deterministic_contradict(claim: Claim, evidence: Evidence) -> tuple[bool, str]:
    """Conservative explicit contradiction signals."""
    claim_lower = claim.text.lower()
    ev_lower = evidence.exact_text.lower()

    # Negation flip on shared predicate
    if re.search(r"\b(not|n't|never|did not|was not)\b", ev_lower):
        if not re.search(r"\b(not|n't|never|did not|was not)\b", claim_lower):
            shared = _claim_terms(claim) & set(re.findall(r"[a-z0-9]+", ev_lower))
            if len(shared) >= 2:
                return True, "Evidence negates proposition while sharing subject context"

    # Conflicting numbers for aligned scope (skip when evidence adds methodology/scope qualifiers)
    qualify_markers = (
        "non-gaap", "excluding", "approximately", "estimated", "subject to",
        "continuing operations", "discontinued", "preliminary", "unaudited",
        "constant-currency", "constant currency", "on a reported basis",
    )
    if any(q in ev_lower for q in qualify_markers):
        return False, ""

    claim_nums = _extract_significant_numbers(claim.text)
    ev_nums = _extract_significant_numbers(evidence.exact_text)
    if claim_nums and ev_nums and _relevance_score(claim, evidence) >= 0.25:
        claim_set = set(claim_nums)
        ev_set = set(ev_nums)
        if claim_set and ev_set and not claim_set & ev_set:
            scope_ok = (
                (claim.temporal_scope and claim.temporal_scope.lower() in ev_lower)
                or ("2025" in claim.text and "2025" in ev_lower)
                or ("billion" in claim_lower and "billion" in ev_lower)
            )
            if scope_ok:
                return True, "Conflicting numeric values for same scoped proposition"

    return False, ""


def _deterministic_qualify(claim: Claim, evidence: Evidence) -> tuple[bool, str]:
    """Evidence adds scope/condition without full support."""
    ev_lower = evidence.exact_text.lower()
    qualifiers = (
        "non-gaap", "excluding", "approximately", "estimated", "subject to",
        "continuing operations", "discontinued", "preliminary", "unaudited",
        "constant-currency", "constant currency", "on a reported basis",
    )
    if not any(q in ev_lower for q in qualifiers):
        return False, ""
    overlap = _relevance_score(claim, evidence)
    if overlap < 0.2:
        return False, ""
    if _deterministic_support(claim, evidence)[0]:
        return False, ""
    if _deterministic_contradict(claim, evidence)[0]:
        return False, ""
    return True, "Evidence qualifies proposition with additional scope/conditions"


def _classify_deterministic(
    claim: Claim, evidence: Evidence
) -> ClaimEvidenceAssessment | None:
    ok, reason = _deterministic_support(claim, evidence)
    if ok:
        return ClaimEvidenceAssessment(
            evidence_id=evidence.id or 0,
            relationship=ClaimEvidenceRelationship.SUPPORTS.value,
            reasoning=reason,
            classification_mode="deterministic",
        )

    ok, reason = _deterministic_contradict(claim, evidence)
    if ok:
        return ClaimEvidenceAssessment(
            evidence_id=evidence.id or 0,
            relationship=ClaimEvidenceRelationship.CONTRADICTS.value,
            reasoning=reason,
            classification_mode="deterministic",
        )

    ok, reason = _deterministic_qualify(claim, evidence)
    if ok:
        return ClaimEvidenceAssessment(
            evidence_id=evidence.id or 0,
            relationship=ClaimEvidenceRelationship.QUALIFIES.value,
            reasoning=reason,
            classification_mode="deterministic",
        )

    return None


async def _classify_batch_llm(
    claim: Claim,
    candidates: list[Evidence],
    *,
    llm: Any,
) -> list[ClaimEvidenceAssessment]:
    if not candidates or llm is None:
        return []

    evidence_block = "\n\n".join(
        f"[EVIDENCE_ID={ev.id}]\n{ev.exact_text}" for ev in candidates
    )

    system_prompt = """You classify how each evidence passage relates to ONE material claim.

For each evidence item return exactly one of:
- supports: evidence directly supports the exact claim proposition
- contradicts: evidence clearly contradicts the claim (not mere absence of support)
- qualifies: evidence partially relates but adds conditions, scope limits, or hedges
- none: evidence is irrelevant or too weak to classify

Be conservative:
- supports requires direct entailment, not topical similarity
- contradicts requires explicit conflict, not missing information
- qualifies is for partial/conditional alignment only
- when uncertain, return none"""

    user_prompt = f"""Material claim:
{claim.text}

Temporal scope: {claim.temporal_scope or 'unspecified'}

Evidence items:
{evidence_block}

Classify each evidence_id."""

    results: list[ClaimEvidenceAssessment] = []
    with trace_llm_call("claim_verification", "classify_relationships_batch") as span:
        span.set_input({"claim": claim.text[:120], "candidate_count": len(candidates)})
        try:
            structured = llm.with_structured_output(ClaimRelationshipBatchOutput)
            batch: ClaimRelationshipBatchOutput = await structured.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            valid = {
                ClaimEvidenceRelationship.SUPPORTS.value,
                ClaimEvidenceRelationship.CONTRADICTS.value,
                ClaimEvidenceRelationship.QUALIFIES.value,
            }
            for item in batch.assessments:
                rel = item.relationship.lower().strip()
                if rel not in valid:
                    continue
                results.append(
                    ClaimEvidenceAssessment(
                        evidence_id=item.evidence_id,
                        relationship=rel,
                        reasoning=item.reasoning or "LLM classification",
                        classification_mode="llm",
                    )
                )
            span.set_output({"classified": len(results)})
        except Exception as e:
            logger.warning("LLM relationship classification failed: %s", e)
            span.set_error(e)

    return results


def _count_independent_supports(
    claim_id: int,
    relations: list[ClaimEvidenceRelation],
    evidence_by_id: dict[int, Evidence],
    sources_by_id: dict[int, Source],
) -> int:
    domains: set[str] = set()
    for rel in relations:
        if rel.claim_id != claim_id or rel.relationship != ClaimEvidenceRelationship.SUPPORTS:
            continue
        ev = evidence_by_id.get(rel.evidence_id)
        if ev is None:
            continue
        domain = source_publisher_domain(sources_by_id.get(ev.source_id))
        if domain:
            domains.add(domain)
    return len(domains)


def _has_credible(
    claim_id: int,
    relationship: ClaimEvidenceRelationship,
    relations: list[ClaimEvidenceRelation],
) -> bool:
    return any(
        rel.claim_id == claim_id and rel.relationship == relationship
        for rel in relations
    )


def aggregate_verification_status(
    claim: Claim,
    all_relations: list[ClaimEvidenceRelation],
    evidence_by_id: dict[int, Evidence],
    sources_by_id: dict[int, Source],
) -> tuple[VerificationStatus, EvidenceConfidence, str]:
    """Aggregate relationships into verification status and evidence confidence."""
    claim_id = claim.id or 0

    if claim.claim_type.value in _UNVERIFIABLE_TYPES:
        return (
            VerificationStatus.UNVERIFIABLE,
            EvidenceConfidence.LOW,
            "Claim type is not directly verifiable from documentary evidence",
        )

    supports = _has_credible(claim_id, ClaimEvidenceRelationship.SUPPORTS, all_relations)
    contradicts = _has_credible(
        claim_id, ClaimEvidenceRelationship.CONTRADICTS, all_relations
    )
    qualifies = _has_credible(
        claim_id, ClaimEvidenceRelationship.QUALIFIES, all_relations
    )
    independent_supports = _count_independent_supports(
        claim_id, all_relations, evidence_by_id, sources_by_id
    )

    parts: list[str] = []
    if independent_supports:
        parts.append(f"{independent_supports} independent publisher domain(s) support")
    if contradicts:
        parts.append("contradicting evidence present")
    if qualifies:
        parts.append("qualifying evidence present")

    if supports and contradicts:
        return (
            VerificationStatus.UNCERTAIN,
            EvidenceConfidence.LOW,
            "; ".join(parts) or "Credible support and contradict coexist",
        )

    if not supports and contradicts:
        return (
            VerificationStatus.CONTRADICTED,
            EvidenceConfidence.LOW,
            "; ".join(parts) or "Strong relevant contradict without support",
        )

    if independent_supports >= 2 and not contradicts:
        return (
            VerificationStatus.SUPPORTED,
            EvidenceConfidence.HIGH,
            "; ".join(parts) or "Multiple independent sources support",
        )

    if supports and qualifies and not contradicts:
        return (
            VerificationStatus.PARTIALLY_SUPPORTED,
            EvidenceConfidence.MEDIUM,
            "; ".join(parts) or "Support with qualifying conditions",
        )

    if supports and not contradicts:
        return (
            VerificationStatus.PARTIALLY_SUPPORTED,
            EvidenceConfidence.MEDIUM,
            "; ".join(parts) or "Single-source or limited support",
        )

    if not supports and not contradicts and not qualifies:
        return (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            EvidenceConfidence.LOW,
            "No meaningful relevant cross-source evidence",
        )

    return (
        VerificationStatus.UNCERTAIN,
        EvidenceConfidence.LOW,
        "; ".join(parts) or "Ambiguous evidence state",
    )


async def verify_material_claims(
    material_claims: list[Claim],
    evidence_list: list[Evidence],
    sources: list[Source],
    origin_relations: list[ClaimEvidenceRelation],
    research_run_id: int,
    *,
    llm: Any | None = None,
    use_llm: bool = True,
) -> tuple[
    list[VerificationResult],
    list[ClaimEvidenceRelation],
    ClaimVerificationMetrics,
]:
    """
    Cross-source verification for material claims.

    Preserves origin SUPPORTS relations; adds cross-source links only.
    """
    start = time.monotonic()
    metrics = ClaimVerificationMetrics()
    evidence_by_id = {ev.id: ev for ev in evidence_list if ev.id is not None}
    sources_by_id = {s.id: s for s in sources if s.id is not None}

    all_relations = list(origin_relations)
    new_relations: list[ClaimEvidenceRelation] = []
    verifications: list[VerificationResult] = []

    existing_pairs = {
        (rel.claim_id, rel.evidence_id, rel.relationship)
        for rel in origin_relations
    }

    for claim in material_claims:
        if claim.id is None:
            continue
        metrics.material_claims_processed += 1

        origin_ids = _origin_evidence_ids(claim.id, all_relations)
        origin_domains = _origin_domains(
            claim.id, all_relations, evidence_by_id, sources_by_id
        )

        candidates = select_cross_source_candidates(
            claim,
            evidence_list,
            sources_by_id,
            origin_ids,
            origin_domains,
        )

        ambiguous: list[Evidence] = []
        for ev in candidates:
            det = _classify_deterministic(claim, ev)
            if det:
                rel_enum = ClaimEvidenceRelationship(det.relationship)
                key = (claim.id, ev.id, rel_enum)
                if key not in existing_pairs:
                    relation = ClaimEvidenceRelation(
                        claim_id=claim.id,
                        evidence_id=ev.id or 0,
                        relationship=rel_enum,
                        reasoning=det.reasoning,
                    )
                    new_relations.append(relation)
                    all_relations.append(relation)
                    existing_pairs.add(key)
                    metrics.cross_source_relations_added += 1
                    if rel_enum == ClaimEvidenceRelationship.SUPPORTS:
                        metrics.deterministic_supports += 1
                    elif rel_enum == ClaimEvidenceRelationship.CONTRADICTS:
                        metrics.deterministic_contradicts += 1
                    else:
                        metrics.deterministic_qualifies += 1
            else:
                ambiguous.append(ev)

        if ambiguous and use_llm and llm is not None:
            metrics.llm_batches += 1
            llm_results = await _classify_batch_llm(claim, ambiguous, llm=llm)
            metrics.llm_classifications += len(llm_results)
            for assessment in llm_results:
                rel_enum = ClaimEvidenceRelationship(assessment.relationship)
                key = (claim.id, assessment.evidence_id, rel_enum)
                if key in existing_pairs:
                    continue
                relation = ClaimEvidenceRelation(
                    claim_id=claim.id,
                    evidence_id=assessment.evidence_id,
                    relationship=rel_enum,
                    reasoning=assessment.reasoning,
                )
                new_relations.append(relation)
                all_relations.append(relation)
                existing_pairs.add(key)
                metrics.cross_source_relations_added += 1

        status, confidence, reasoning = aggregate_verification_status(
            claim, all_relations, evidence_by_id, sources_by_id
        )

        if status == VerificationStatus.SUPPORTED:
            metrics.supported += 1
        elif status == VerificationStatus.PARTIALLY_SUPPORTED:
            metrics.partially_supported += 1
        elif status == VerificationStatus.CONTRADICTED:
            metrics.contradicted += 1
        elif status == VerificationStatus.UNCERTAIN:
            metrics.uncertain += 1
        elif status == VerificationStatus.UNVERIFIABLE:
            metrics.unverifiable += 1
        else:
            metrics.insufficient_evidence += 1

        verifications.append(
            VerificationResult(
                claim_id=claim.id,
                research_run_id=research_run_id,
                status=status,
                confidence=confidence,
                reasoning=reasoning,
                knowledge_category=None,
            )
        )

    metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
    return verifications, new_relations, metrics
