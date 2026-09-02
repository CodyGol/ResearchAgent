"""Source normalization from Tavily search hits to domain Source entities."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from domain.models import Source, SourceQuality, SourceType
from services.url_canonicalizer import canonicalize_url
from state import SearchResult

_ACADEMIC_DOMAINS = frozenset({
    "arxiv.org",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "ieee.org",
    "acm.org",
    "nature.com",
    "sciencedirect.com",
})

_SPORTS_OFFICIAL_DOMAINS = (
    "formula1.com",
    "fia.com",
    "fifa.com",
    "olympics.com",
    "nba.com",
    "nfl.com",
)

_OFFICIAL_PATTERNS = (
    ".gov",
    ".edu",
    "docs.",
    "developer.",
    "api.",
) + tuple(f".{d}" if not d.startswith(".") else d for d in _SPORTS_OFFICIAL_DOMAINS)

_NEWS_DOMAINS = frozenset({
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
})

_LOW_AUTHORITY_DOMAINS = frozenset({
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "old.reddit.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
})

_BRAND_TLDS = frozenset({"com", "io", "ai", "dev", "co", "net", "org"})

_VENDOR_OWNERSHIP_SUBDOMAINS = (
    "docs.",
    "api.",
    "developer.",
    "developers.",
    "platform.",
    "help.",
    "support.",
)

_PUBLISHER_DOMAIN_SUFFIXES = frozenset({
    "blog",
    "news",
    "reviews",
    "review",
    "hub",
    "magazine",
    "daily",
})

_PUBLISHER_PATH_SEGMENTS = frozenset({
    "blog",
    "news",
    "articles",
    "posts",
    "tag",
    "category",
    "categories",
    "author",
})

_VENDOR_INFRA_PATH_SEGMENTS = frozenset({
    "api",
    "docs",
    "developer",
    "developers",
    "platform",
    "pricing",
    "products",
    "enterprise",
    "billing",
    "plans",
    "rates",
    "rate",
})


def _path_segments(url: str) -> list[str]:
    path = urlparse(url).path.lower().strip("/")
    return [segment for segment in path.split("/") if segment]


def _domain_label(domain: str) -> str:
    parts = domain.split(".")
    return parts[0] if parts else ""


def _looks_like_publisher_domain(domain: str) -> bool:
    label = _domain_label(domain)
    return any(label.endswith(suffix) for suffix in _PUBLISHER_DOMAIN_SUFFIXES)


def is_likely_first_party_vendor_site(domain: str, url: str) -> bool:
    """
    Whether a URL likely belongs to a vendor's own site (ownership-first heuristic).

    Domain/subdomain identity establishes ownership. Path segments may confirm vendor
    infrastructure pages on an owned domain, but path tokens alone never establish ownership.
    """
    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in _LOW_AUTHORITY_DOMAINS):
        return False

    if any(domain.startswith(prefix) for prefix in _VENDOR_OWNERSHIP_SUBDOMAINS):
        return True

    parts = domain.split(".")
    if len(parts) >= 3 and parts[0] in {
        "docs", "api", "developer", "developers", "platform", "help", "support",
    }:
        return True

    if len(parts) == 2 and parts[1] in _BRAND_TLDS:
        if _looks_like_publisher_domain(domain):
            return False
        segments = _path_segments(url)
        if not segments or segments[0] in _PUBLISHER_PATH_SEGMENTS:
            return False
        return segments[0] in _VENDOR_INFRA_PATH_SEGMENTS

    return False


def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


def _classify_source_type(url: str, domain: str) -> SourceType:
    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in _LOW_AUTHORITY_DOMAINS):
        return SourceType.WEB
    if is_likely_first_party_vendor_site(domain, url):
        return SourceType.OFFICIAL
    if any(academic in domain for academic in _ACADEMIC_DOMAINS):
        return SourceType.ACADEMIC
    if domain in _SPORTS_OFFICIAL_DOMAINS or any(
        domain.endswith(f".{d}") for d in _SPORTS_OFFICIAL_DOMAINS
    ):
        return SourceType.OFFICIAL
    if any(pattern in domain for pattern in _OFFICIAL_PATTERNS):
        return SourceType.OFFICIAL
    if any(news in domain for news in _NEWS_DOMAINS):
        return SourceType.NEWS
    if domain.endswith(".gov"):
        return SourceType.OFFICIAL
    return SourceType.WEB


def _classify_source_quality(source_type: SourceType, domain: str, url: str = "") -> SourceQuality:
    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in _LOW_AUTHORITY_DOMAINS):
        return SourceQuality.USER_GENERATED
    if is_likely_first_party_vendor_site(domain, url):
        return SourceQuality.OFFICIAL
    if source_type == SourceType.ACADEMIC:
        return SourceQuality.ACADEMIC
    if source_type == SourceType.OFFICIAL or domain.endswith(".gov"):
        return SourceQuality.OFFICIAL
    if source_type == SourceType.NEWS:
        return SourceQuality.REPUTABLE_SECONDARY
    if "github.com" in domain or "stackoverflow.com" in domain:
        return SourceQuality.PRIMARY
    return SourceQuality.GENERAL_SECONDARY


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_publisher(domain: str, title: str) -> str | None:
    if not domain:
        return None
    # Use domain as publisher hint; strip TLD for readability
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2].replace("-", " ").title()
    return domain


def normalize_search_result(
    result: SearchResult,
    research_run_id: int,
) -> Source:
    """
    Convert a Tavily SearchResult into a normalized Source entity.

    Args:
        result: Raw search hit from Tavily
        research_run_id: Parent research run ID

    Returns:
        Source ready for persistence
    """
    domain = _extract_domain(result.url)
    source_type = _classify_source_type(result.url, domain)
    source_quality = _classify_source_quality(source_type, domain, result.url)
    content = result.content or ""
    now = datetime.now(timezone.utc)

    return Source(
        research_run_id=research_run_id,
        url=result.url,
        title=result.title or "",
        publisher=_extract_publisher(domain, result.title),
        accessed_at=now,
        source_type=source_type,
        source_quality=source_quality,
        content=content,
        content_hash=_content_hash(content),
        relevance_score=result.score,
        metadata={"domain": domain, "tavily_score": result.score},
    )


def _merge_snippet_content(existing: str, new: str) -> str:
    """Merge two snippets conservatively, keeping the richer unique content."""
    if not existing:
        return new
    if not new:
        return existing
    if existing == new:
        return existing
    if new in existing:
        return existing
    if existing in new:
        return new
    # Append non-overlapping content with separator
    return f"{existing}\n\n{new}"


@dataclass
class SourceDeduplicationMetrics:
    """Metrics for within-run source canonicalization."""

    raw_sources_found: int = 0
    canonical_sources_retained: int = 0
    duplicates_removed: int = 0
    merged_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_sources_found": self.raw_sources_found,
            "canonical_sources_retained": self.canonical_sources_retained,
            "duplicates_removed": self.duplicates_removed,
            "merged_url_count": len(self.merged_urls),
        }


def normalize_search_results(
    results: list[SearchResult],
    research_run_id: int,
) -> list[Source]:
    """
    Normalize and deduplicate search results within a run.

    Deduplicates by canonical URL. When the same URL appears with different
    snippets, merges content conservatively and keeps the best metadata.

    Args:
        results: List of Tavily search hits
        research_run_id: Parent research run ID

    Returns:
        Deduplicated list of Source entities
    """
    sources, _ = normalize_search_results_with_metrics(results, research_run_id)
    return sources


def normalize_search_results_with_metrics(
    results: list[SearchResult],
    research_run_id: int,
) -> tuple[list[Source], SourceDeduplicationMetrics]:
    """
    Normalize search results with deduplication metrics.

    Returns:
        Tuple of (deduplicated sources, metrics)
    """
    metrics = SourceDeduplicationMetrics(raw_sources_found=len(results))
    canonical_map: dict[str, Source] = {}

    for result in results:
        source = normalize_search_result(result, research_run_id)
        canonical = canonicalize_url(source.url)
        source.metadata["canonical_url"] = canonical
        source.metadata.setdefault("original_urls", [source.url])

        if canonical in canonical_map:
            metrics.duplicates_removed += 1
            existing = canonical_map[canonical]
            merged_content = _merge_snippet_content(existing.content, source.content)
            merged_title = existing.title if len(existing.title) >= len(source.title) else source.title
            merged_score = max(existing.relevance_score, source.relevance_score)
            original_urls = list(
                dict.fromkeys(
                    existing.metadata.get("original_urls", [])
                    + source.metadata.get("original_urls", [])
                )
            )
            canonical_map[canonical] = existing.model_copy(
                update={
                    "url": canonical,
                    "title": merged_title,
                    "content": merged_content,
                    "content_hash": _content_hash(merged_content),
                    "relevance_score": merged_score,
                    "metadata": {
                        **existing.metadata,
                        "canonical_url": canonical,
                        "original_urls": original_urls,
                        "merge_count": existing.metadata.get("merge_count", 1) + 1,
                    },
                }
            )
            if canonical not in metrics.merged_urls:
                metrics.merged_urls.append(canonical)
        else:
            canonical_map[canonical] = source.model_copy(update={"url": canonical})

    sources = list(canonical_map.values())
    metrics.canonical_sources_retained = len(sources)
    return sources, metrics


def normalize_claim_text(text: str) -> str:
    """Normalize text for conservative duplicate detection."""
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s$%.,\-]", "", normalized)
    return normalized
