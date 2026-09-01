"""Canonical URL normalization for within-run source deduplication."""

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Common tracking parameters to strip
_TRACKING_PARAMS = frozenset({
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "gbraid",
    "wbraid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
})


def canonicalize_url(url: str) -> str:
    """
    Canonicalize a URL for deduplication within a research run.

    Normalizes:
    - scheme (default https)
    - www. prefix on host
    - trailing slashes on path
    - fragments (removed)
    - common tracking query parameters

    Does NOT merge distinct paths on the same domain.
    """
    if not url or not url.strip():
        return url

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Filter tracking params, preserve order of remaining params
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {
            k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS
        }
        # Sort for stable canonical form
        query = urlencode(sorted(filtered.items()), doseq=True)
    else:
        query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))
