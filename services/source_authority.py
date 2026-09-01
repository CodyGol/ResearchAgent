"""Domain-aware source authority for fact sufficiency."""

from domain.models import Source, SourceQuality, SourceType
from services.fact_target import FactDomain

_QUALITY_RANK: dict[SourceQuality, int] = {
    SourceQuality.PRIMARY: 5,
    SourceQuality.OFFICIAL: 5,
    SourceQuality.ACADEMIC: 4,
    SourceQuality.REPUTABLE_SECONDARY: 3,
    SourceQuality.GENERAL_SECONDARY: 2,
    SourceQuality.USER_GENERATED: 1,
    SourceQuality.UNKNOWN: 0,
}

_DOMAIN_MIN_RANK: dict[FactDomain, int] = {
    FactDomain.GEOGRAPHIC: 3,
    FactDomain.FINANCIAL: 4,
    FactDomain.SPORTS: 3,  # official sports domains boosted below
    FactDomain.CORPORATE: 3,
    FactDomain.TECHNICAL: 3,
    FactDomain.GENERAL: 3,
}

_DOMAIN_PREFERRED: dict[FactDomain, tuple[str, ...]] = {
    FactDomain.GEOGRAPHIC: (".gov", "cia.gov", "britannica.com", "worldbank.org"),
    FactDomain.FINANCIAL: (
        "investor.apple.com", "sec.gov", "ir.", "investor.", "apple.com/newsroom",
    ),
    FactDomain.SPORTS: (
        "formula1.com", "fia.com", "fifa.com", "olympics.com", "nba.com", "nfl.com",
    ),
    FactDomain.CORPORATE: ("sec.gov", "investor.", "ir."),
    FactDomain.TECHNICAL: ("docs.", "github.com", "python.org", "pypi.org"),
}

# Domains that are always authoritative for their fact domain
_SPORTS_OFFICIAL_DOMAINS = ("formula1.com", "fia.com", "fifa.com", "olympics.com")
_FINANCIAL_OFFICIAL_DOMAINS = ("sec.gov", "investor.", "ir.", "investor.apple.com")


def _url_matches(url_lower: str, patterns: tuple[str, ...]) -> bool:
    return any(p in url_lower for p in patterns)


def source_quality_rank(source: Source, domain: FactDomain | None = None) -> int:
    rank = _QUALITY_RANK.get(source.source_quality, 0)
    url_lower = source.url.lower()

    if domain:
        for pattern in _DOMAIN_PREFERRED.get(domain, ()):
            if pattern in url_lower:
                rank = max(rank, _DOMAIN_MIN_RANK.get(domain, 3) + 1)

    return rank


def is_source_adequate_for_domain(source: Source, domain: FactDomain) -> bool:
    """Whether source quality meets domain-specific sufficiency threshold."""
    url_lower = source.url.lower()

    # Sports: official competition sites are always adequate
    if domain == FactDomain.SPORTS and _url_matches(url_lower, _SPORTS_OFFICIAL_DOMAINS):
        return True

    # Financial: primary filings / IR always adequate
    if domain == FactDomain.FINANCIAL and _url_matches(url_lower, _FINANCIAL_OFFICIAL_DOMAINS):
        return True

    # Geographic: gov or reputable secondary
    if domain == FactDomain.GEOGRAPHIC and (
        source.source_type == SourceType.OFFICIAL
        or ".gov" in url_lower
        or source.source_quality == SourceQuality.REPUTABLE_SECONDARY
        or "britannica.com" in url_lower
    ):
        return True

    min_rank = _DOMAIN_MIN_RANK.get(domain, 3)
    rank = source_quality_rank(source, domain)
    return rank >= min_rank


def prioritize_sources_for_domain(
    sources: list[Source], domain: FactDomain
) -> list[Source]:
    def score(s: Source) -> int:
        return source_quality_rank(s, domain)

    return sorted(sources, key=score, reverse=True)
