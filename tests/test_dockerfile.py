"""Regression tests for Docker deployment packaging."""

from pathlib import Path


def test_dockerfile_includes_runtime_packages():
    """Phase 2D modules must be present in the production image."""
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    content = dockerfile.read_text()
    assert "COPY services/" in content
    assert "COPY domain/" in content
