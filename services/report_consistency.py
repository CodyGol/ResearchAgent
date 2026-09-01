"""Lightweight report consistency validation against validated evidence."""

import re
from dataclasses import dataclass, field


@dataclass
class ConsistencyResult:
    """Outcome of consistency checks on a draft report."""

    is_consistent: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Patterns for extracting numeric claims from text
_COUNT_PATTERN = re.compile(
    r"\b(\d+)\s+(?:of\s+)?(\d+)?\s*(?:races?|grand\s+prix|events?|games?|matches?)\b",
    re.IGNORECASE,
)
_TOTAL_RACES_PATTERN = re.compile(
    r"\b(?:season|championship|calendar)\s+(?:consisted\s+of|had|featured|included)\s+(\d+)\s+races?\b",
    re.IGNORECASE,
)
_STANDALONE_RACE_COUNT = re.compile(
    r"\b(\d+)\s+races?\b",
    re.IGNORECASE,
)


def _extract_race_totals(text: str) -> set[int]:
    """
    Extract total race counts from text, excluding subset counts from 'X of Y' patterns.

    '19 of 22 races' → total is 22, not a conflict between 19 and 22.
    """
    totals: set[int] = set()
    subset_numbers: set[int] = set()

    # Collect X from "X of Y race(s)" patterns — Y is the total, X is subset
    for match in _COUNT_PATTERN.finditer(text):
        first = int(match.group(1))
        second = match.group(2)
        if second:
            totals.add(int(second))
            subset_numbers.add(first)
        else:
            totals.add(first)

    for match in _TOTAL_RACES_PATTERN.finditer(text):
        totals.add(int(match.group(1)))

    # Standalone "N races" only counts as total if not already a subset number
    for match in _STANDALONE_RACE_COUNT.finditer(text):
        num = int(match.group(1))
        # Skip if this number is only appearing as part of an already-handled pattern
        if num not in subset_numbers:
            totals.add(num)

    return totals


def check_internal_consistency(report_content: str) -> ConsistencyResult:
    """
    Detect obvious internal contradictions in report text.

    Uses deterministic pattern matching — no LLM required.
    """
    issues: list[str] = []
    warnings: list[str] = []

    race_totals = _extract_race_totals(report_content)
    if len(race_totals) > 1:
        sorted_nums = sorted(race_totals)
        issues.append(
            f"Conflicting race counts mentioned: {sorted_nums}. "
            "Report references different total race numbers."
        )

    # Check "X of Y" where X > Y
    for match in _COUNT_PATTERN.finditer(report_content):
        first = int(match.group(1))
        second = match.group(2)
        if second and first > int(second):
            issues.append(
                f"Impossible ratio: {first} of {second} "
                f"(context: '{match.group(0)}')"
            )

    return ConsistencyResult(
        is_consistent=len(issues) == 0,
        issues=issues,
        warnings=warnings,
    )


def check_evidence_alignment(
    report_content: str,
    evidence_texts: list[str],
) -> ConsistencyResult:
    """
    Flag report numbers that don't appear in any evidence text.

    Conservative: only flags standalone numbers (4+ digits or percentages).
    """
    warnings: list[str] = []
    issues: list[str] = []

    # Extract notable numbers from report (avoid years 1900-2099 for now)
    report_numbers = set(re.findall(r"\b(\d{1,3}(?:\.\d+)?%?)\b", report_content))
    evidence_corpus = " ".join(evidence_texts).lower()

    for num in report_numbers:
        clean = num.rstrip("%")
        if clean in ("0", "1", "2", "3"):
            continue
        if clean.isdigit() and 1900 <= int(clean) <= 2099:
            continue
        if clean.lower() not in evidence_corpus and num.lower() not in evidence_corpus:
            # Only warn for significant-looking stats
            if "%" in num or (clean.replace(".", "").isdigit() and len(clean) >= 2):
                warnings.append(
                    f"Report mentions '{num}' which does not appear in validated evidence"
                )

    return ConsistencyResult(
        is_consistent=len(issues) == 0,
        issues=issues,
        warnings=warnings,
    )


def run_consistency_checks(
    report_content: str,
    evidence_texts: list[str],
) -> ConsistencyResult:
    """Run all consistency checks and merge results."""
    internal = check_internal_consistency(report_content)
    alignment = check_evidence_alignment(report_content, evidence_texts)

    return ConsistencyResult(
        is_consistent=internal.is_consistent and alignment.is_consistent,
        issues=internal.issues + alignment.issues,
        warnings=internal.warnings + alignment.warnings,
    )
