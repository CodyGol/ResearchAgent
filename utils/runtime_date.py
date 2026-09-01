"""Reference date for temporal reasoning (injectable in tests)."""

from __future__ import annotations

from datetime import date

_REFERENCE_DATE: date | None = None


def get_reference_date() -> date:
    """Return the current reference date (today in production, overridable in tests)."""
    if _REFERENCE_DATE is not None:
        return _REFERENCE_DATE
    return date.today()


def set_reference_date_for_tests(value: date | None) -> None:
    """Override reference date in tests; pass None to reset."""
    global _REFERENCE_DATE
    _REFERENCE_DATE = value


def format_reference_date_for_prompt() -> str:
    return get_reference_date().isoformat()


def classify_date_relative(evidence_date: date, reference: date | None = None) -> str:
    """Classify evidence_date relative to reference: past, same, or future."""
    ref = reference or get_reference_date()
    if evidence_date < ref:
        return "past"
    if evidence_date > ref:
        return "future"
    return "same"


def temporal_context_block() -> str:
    """Prompt block for correct past/future date reasoning."""
    today = format_reference_date_for_prompt()
    return (
        f"Today's date is {today}. "
        "Treat calendar dates before today as past or historical, not future. "
        "Do not describe past dates as upcoming or future relative to today."
    )
