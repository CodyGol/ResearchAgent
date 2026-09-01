"""Tests for URL canonicalization."""

from services.url_canonicalizer import canonicalize_url


class TestCanonicalizeUrl:
    def test_strips_www(self):
        assert canonicalize_url("https://www.example.com/page") == canonicalize_url(
            "https://example.com/page"
        )

    def test_strips_trailing_slash(self):
        assert canonicalize_url("https://example.com/page/") == canonicalize_url(
            "https://example.com/page"
        )

    def test_strips_tracking_params(self):
        url = "https://example.com/page?utm_source=twitter&id=123"
        canonical = canonicalize_url(url)
        assert "utm_source" not in canonical
        assert "id=123" in canonical

    def test_removes_fragment(self):
        assert canonicalize_url("https://example.com/page#section") == canonicalize_url(
            "https://example.com/page"
        )

    def test_preserves_distinct_paths(self):
        a = canonicalize_url("https://example.com/page-a")
        b = canonicalize_url("https://example.com/page-b")
        assert a != b
