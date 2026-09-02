"""Decision-aware research coverage and authoritative retrieval helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from domain.models import Source, SourceQuality, SourceType
from services.decision_framing_schemas import DecisionFrame, DecisionOption
from services.source_normalizer import is_likely_first_party_vendor_site
from state import ResearchPlan, SearchResult

_VENDOR_CONTROLLED_KEYWORDS = frozenset({
    "cost",
    "pricing",
    "price",
    "api",
    "capability",
    "capabilities",
    "specification",
    "specifications",
    "tier",
    "tiers",
    "integration",
    "integrations",
    "limit",
    "limits",
    "quota",
    "quotas",
    "service level",
    "sla",
    "documentation",
    "docs",
    "release",
    "releases",
    "product",
    "products",
    "enterprise",
    "subscription",
    "billing",
})

_LOW_AUTHORITY_DOMAINS = frozenset({
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "old.reddit.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "medium.com",
})

_VENDOR_PATH_HINTS = (
    "/pricing",
    "/api",
    "/docs",
    "/platform",
    "/products",
    "/enterprise",
    "/billing",
    "/rate",
    "/rates",
    "/plans",
)

_BRAND_TLDS = frozenset({"com", "io", "ai", "dev", "co", "net", "org"})


@dataclass
class CoveragePairSpec:
    """Authoritative search plan for one explicit option × primary criterion pair."""

    option_label: str
    criterion_label: str
    vendor_controlled: bool
    primary_query: str
    retry_query: str
    official_domain_candidates: list[str] = field(default_factory=list)


@dataclass
class DecisionCoverageMetrics:
    """Observability for decision-critical retrieval coverage."""

    decision_coverage_pairs: int = 0
    authoritative_search_attempts: int = 0
    authoritative_retries: int = 0
    authoritative_results_found: int = 0
    authoritative_evidence_accepted: int = 0
    decision_coverage_pairs_without_evidence: int = 0
    pair_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_coverage_pairs": self.decision_coverage_pairs,
            "authoritative_search_attempts": self.authoritative_search_attempts,
            "authoritative_retries": self.authoritative_retries,
            "authoritative_results_found": self.authoritative_results_found,
            "authoritative_evidence_accepted": self.authoritative_evidence_accepted,
            "decision_coverage_pairs_without_evidence": self.decision_coverage_pairs_without_evidence,
            "pair_details": self.pair_details,
        }


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _normalize_label(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_vendor_controlled_criterion(criterion_label: str) -> bool:
    """Whether a criterion concerns vendor-controlled factual information."""
    label = _normalize_label(criterion_label)
    return any(keyword in label for keyword in _VENDOR_CONTROLLED_KEYWORDS)


def is_pricing_criterion(criterion_label: str) -> bool:
    """Whether a criterion concerns vendor pricing."""
    label = _normalize_label(criterion_label)
    return label in {"cost", "pricing", "price"}


def is_pricing_relevant_url(url: str) -> bool:
    """Whether a URL likely contains vendor pricing information."""
    url_lower = url.lower()
    return any(hint in url_lower for hint in _VENDOR_PATH_HINTS if hint in {
        "/pricing", "/price", "/billing", "/rate", "/rates", "/plans",
    })


def discover_pricing_domains_from_results(
    results: list[SearchResult],
    option_label: str,
) -> list[str]:
    """
    Discover likely first-party pricing domains from search results (general, not vendor-specific).

    Handles brand-split product domains (e.g. claude.com for Anthropic) when pricing pages
    mention the option label in title or content.
    """
    option_lower = option_label.lower()
    slug = re.sub(r"[^a-z0-9]", "", option_lower)
    discovered: list[str] = []
    for result in results:
        if not is_pricing_relevant_url(result.url):
            continue
        domain = extract_domain(result.url)
        if not domain or is_low_authority_domain(domain):
            continue
        text = f"{result.title} {result.content}".lower()
        if slug and slug in domain.replace("-", "").replace(".", ""):
            discovered.append(domain)
            continue
        if any(domain.startswith(prefix) for prefix in ("docs.", "api.", "platform.", "developer.")):
            if option_lower in text:
                discovered.append(domain)
            continue
        if option_lower in text and len(domain.split(".")) <= 3:
            discovered.append(domain)
    return list(dict.fromkeys(discovered))


def infer_official_domain_candidates(option_label: str) -> list[str]:
    """
    Infer likely first-party domains from an option label (general, not vendor-specific).

    Examples: "OpenAI" -> openai.com, docs.openai.com; "Acme Corp" -> acmecorp.com
    """
    slug = re.sub(r"[^a-z0-9]", "", option_label.lower())
    if len(slug) < 3:
        return []

    candidates = [f"{slug}.com", f"docs.{slug}.com", f"api.{slug}.com"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for domain in candidates:
        if domain not in seen:
            seen.add(domain)
            ordered.append(domain)
    return ordered


def build_authority_seeking_query(option: DecisionOption, criterion_label: str) -> str:
    """Build an authority-seeking search query for a coverage pair."""
    crit = _normalize_label(criterion_label)
    if is_vendor_controlled_criterion(criterion_label):
        if crit in {"cost", "pricing", "price"}:
            return f"{option.label} API pricing official"
        return f"{option.label} {criterion_label} official documentation"
    return f"{option.label} {criterion_label}"


def build_authority_retry_query(option: DecisionOption, criterion_label: str) -> str:
    """Targeted retry query when the initial authority search misses first-party results."""
    crit = _normalize_label(criterion_label)
    if crit in {"cost", "pricing", "price"}:
        return f"{option.label} API pricing official documentation"
    return f"{option.label} {criterion_label} official documentation site"


def build_coverage_pair_specs(frame: DecisionFrame) -> list[CoveragePairSpec]:
    """Build authoritative search specs for explicit option × primary explicit criteria."""
    explicit_options = [o for o in frame.options if o.origin == "explicit"]
    primary_criteria = [
        c for c in frame.criteria if c.origin == "explicit" and c.priority == "primary"
    ]
    specs: list[CoveragePairSpec] = []
    for option in explicit_options:
        for criterion in primary_criteria:
            vendor_controlled = is_vendor_controlled_criterion(criterion.label)
            specs.append(
                CoveragePairSpec(
                    option_label=option.label,
                    criterion_label=criterion.label,
                    vendor_controlled=vendor_controlled,
                    primary_query=build_authority_seeking_query(option, criterion.label),
                    retry_query=build_authority_retry_query(option, criterion.label),
                    official_domain_candidates=infer_official_domain_candidates(option.label)
                    if vendor_controlled
                    else [],
                )
            )
    return specs


def build_coverage_subqueries(frame: DecisionFrame) -> list[str]:
    """Deterministic authority-seeking queries for each explicit option × primary criterion."""
    return [spec.primary_query for spec in build_coverage_pair_specs(frame)]


def merge_decision_coverage_into_plan(
    plan: ResearchPlan,
    frame: DecisionFrame,
    *,
    max_queries: int,
) -> ResearchPlan:
    """Prepend decision coverage queries so they survive budget trimming."""
    coverage = build_coverage_subqueries(frame)
    if not coverage:
        return plan

    seen = {_normalize_query(q) for q in plan.sub_queries}
    prepended: list[str] = []
    for query in coverage:
        norm = _normalize_query(query)
        if norm not in seen:
            prepended.append(query)
            seen.add(norm)

    merged_sub_queries = prepended + list(plan.sub_queries)
    merged_sub_queries = list(dict.fromkeys(merged_sub_queries))[:max_queries]

    merged_search_terms = list(dict.fromkeys(prepended + list(plan.search_terms)))

    return plan.model_copy(
        update={
            "sub_queries": merged_sub_queries,
            "search_terms": merged_search_terms,
        }
    )


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def is_low_authority_domain(domain: str) -> bool:
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in _LOW_AUTHORITY_DOMAINS)


def domain_matches_candidate(domain: str, candidates: list[str]) -> bool:
    if not domain or not candidates:
        return False
    for candidate in candidates:
        cand = candidate.lower().removeprefix("www.")
        if domain == cand or domain.endswith(f".{cand}") or cand.endswith(domain):
            return True
    return False


def _official_domain_candidates(
    option_label: str,
    extra_domains: list[str] | None = None,
) -> list[str]:
    candidates = infer_official_domain_candidates(option_label)
    if extra_domains:
        return list(dict.fromkeys(candidates + extra_domains))
    return candidates


def is_first_party_result(
    result: SearchResult,
    option_label: str,
    *,
    extra_domains: list[str] | None = None,
) -> bool:
    """Whether a search result likely comes from the option's own first-party domain."""
    domain = extract_domain(result.url)
    if not domain or is_low_authority_domain(domain):
        return False
    candidates = _official_domain_candidates(option_label, extra_domains)
    if domain_matches_candidate(domain, candidates):
        return True
    return is_likely_first_party_vendor_site(domain, result.url)


def is_first_party_source(
    source: Source,
    option_label: str,
    *,
    extra_domains: list[str] | None = None,
) -> bool:
    domain = source.metadata.get("domain") or extract_domain(source.url)
    if not domain or is_low_authority_domain(domain):
        return False
    if domain_matches_candidate(domain, _official_domain_candidates(option_label, extra_domains)):
        return True
    return is_likely_first_party_vendor_site(domain, source.url)


def is_usable_authoritative_result(
    result: SearchResult,
    option_label: str,
    *,
    extra_domains: list[str] | None = None,
) -> bool:
    if is_first_party_result(result, option_label, extra_domains=extra_domains):
        return True
    domain = extract_domain(result.url)
    if is_low_authority_domain(domain):
        return False
    if any(domain.startswith(prefix) for prefix in ("docs.", "api.", "developer.")):
        return True
    if domain.endswith(".gov") or ".edu" in domain:
        return True
    return False


def is_usable_authoritative_source(source: Source, option_label: str) -> bool:
    if is_first_party_source(source, option_label):
        return True
    if source.source_quality in (SourceQuality.OFFICIAL, SourceQuality.PRIMARY, SourceQuality.ACADEMIC):
        return True
    if source.source_type == SourceType.OFFICIAL:
        return True
    return False


def result_has_authoritative_hit(
    results: list[SearchResult],
    spec: CoveragePairSpec,
) -> bool:
    """Whether results contain usable first-party / authoritative evidence for the pair."""
    for result in results:
        if not is_usable_authoritative_result(
            result,
            spec.option_label,
            extra_domains=spec.official_domain_candidates,
        ):
            continue
        if is_pricing_criterion(spec.criterion_label) and not is_pricing_relevant_url(result.url):
            continue
        return True
    return False


def pin_coverage_sources(
    sources: list[Source],
    specs: list[CoveragePairSpec],
) -> list[Source]:
    """
    Ensure first-party / authoritative coverage sources are not dropped by source caps.

    Preserves relative order within pinned and unpinned groups.
    """
    if not specs:
        return sources

    pinned: list[Source] = []
    pinned_ids: set[int] = set()
    for spec in specs:
        for source in sources:
            sid = id(source)
            if sid in pinned_ids:
                continue
            if is_first_party_source(
                source,
                spec.option_label,
                extra_domains=spec.official_domain_candidates,
            ) or (
                spec.vendor_controlled and is_usable_authoritative_source(source, spec.option_label)
            ):
                pinned.append(source)
                pinned_ids.add(sid)

    remainder = [s for s in sources if id(s) not in pinned_ids]
    return pinned + remainder
