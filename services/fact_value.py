"""Structured fact value representation for SIMPLE_FACT fast path."""

import re
from enum import Enum

from pydantic import BaseModel, Field

from services.fact_target import AnswerTarget, AnswerType, FactFreshness


class FactValueType(str, Enum):
    PLACE = "place"
    PERSON = "person"
    NUMBER = "number"
    DATE = "date"
    TEXT = "text"


class StructuredFactValue(BaseModel):
    """Validated answer value extracted from decisive evidence."""

    value: str = Field(..., description="The extracted answer value")
    value_type: FactValueType
    attribute: str
    entity: str
    temporal_scope: str | None = None
    unit: str | None = None
    currency: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    category: str | None = Field(
        None, description="Sub-category e.g. drivers_championship"
    )
    freshness: FactFreshness = FactFreshness.HISTORICAL_STABLE


def _clean_place_name(name: str) -> str:
    """Strip noisy prefixes from extracted place names."""
    cleaned = name.strip().rstrip(".")
    # Drop ALL-CAPS country prefix before city, e.g. "JAPAN  Tokyo"
    parts = re.split(r"\s{2,}|\s*[-–]\s*", cleaned)
    if len(parts) > 1 and parts[0].isupper() and len(parts[0]) > 2:
        cleaned = parts[-1].strip()
    # Take last capitalized token sequence (city name)
    m = re.search(r"([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*)$", cleaned)
    return m.group(1) if m else cleaned



_CAPITAL_PATTERNS = [
    re.compile(
        r"([A-Z][a-zA-Z\s\-']+?)\s+is\s+(?:the\s+)?capital\s+of\s+([A-Za-z\s]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"capital\s+of\s+([A-Za-z\s]+?)\s+is\s+([A-Z][a-zA-Z\s\-']+)",
        re.IGNORECASE,
    ),
]

_WINNER_PATTERNS = [
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:won|secured|claimed|clinched)"
        r"(?:\s+his|\s+her|\s+their)?\s+(?:third|fourth|fifth|\w+)?\s*"
        r"(?:Formula\s*1|F1|Formula One)?\s*(?:world\s+)?(?:drivers?'?\s+)?"
        r"(?:championship|title)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+won\s+the\s+.+?(?:championship|title)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:finished|ended)\s+"
        r"(?:the\s+season\s+)?(?:in\s+)?first(?:\s+place)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b1\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
    ),
    re.compile(
        r"\|\s*1\s*\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ),
    re.compile(
        r"(?:World Drivers Champion|world champion)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+secures\s+"
        r"(?:his|her|their)?\s*(?:\w+\s+)*F1\s+world\s+title",
        re.IGNORECASE,
    ),
]

_SINGLE_NAME_PATTERNS = {_WINNER_PATTERNS[-1].pattern}

_STANDINGS_PATTERNS = {
    _WINNER_PATTERNS[3].pattern,
    _WINNER_PATTERNS[4].pattern,
}

_WINNER_STOPWORDS = frozenset({
    "see", "the", "in", "for", "and", "his", "her", "with", "after", "before",
    "qatar", "abu", "dhabi", "season", "grand", "prix",
})


def _is_valid_winner_name(name: str, *, allow_single: bool = False) -> bool:
    cleaned = name.strip()
    if not cleaned or cleaned.lower() in _WINNER_STOPWORDS:
        return False
    parts = cleaned.split()
    if len(parts) >= 2:
        return all(p[0].isupper() for p in parts if p)
    if allow_single and len(parts) == 1:
        return len(parts[0]) >= 4 and parts[0][0].isupper()
    return False

_REVENUE_PATTERNS = [
    re.compile(
        r"(?:revenue|net sales|total net sales)\s+(?:of\s+)?"
        r"(\$[\d,.]+(?:\s*(?:billion|million|B|M))?)"
        r"(?:\s+(?:in|for)\s+)?(fiscal\s+\d{4}|FY\s*\d{4}|Q\d\s+\d{4})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\$[\d,.]+(?:\s*(?:billion|million|B|M))?)\s+(?:in\s+)?(?:revenue|net sales)"
        r"(?:\s+(?:in|for)\s+)?(fiscal\s+\d{4}|FY\s*\d{4})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"reported\s+(?:total\s+)?(?:net\s+)?sales\s+of\s+(\$[\d,.]+(?:\s*billion)?)"
        r"(?:\s+(?:for|in)\s+)?(fiscal\s+\d{4}|FY\s*\d{4})?",
        re.IGNORECASE,
    ),
]

_CEO_INVALID_PREFIXES = frozenset({
    "since", "in", "on", "at", "the", "a", "an", "as", "of", "from",
})


def _is_valid_ceo_name(name: str) -> bool:
    parts = name.strip().split()
    if not parts or parts[0].lower() in _CEO_INVALID_PREFIXES:
        return False
    return len(parts) >= 2 and all(p[0].isupper() for p in parts if p)


_CEO_PATTERNS = [
    re.compile(
        r"(?:CEO|chief executive(?:\s+officer)?)\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(?:the\s+)?(?:CEO|chief executive)",
        re.IGNORECASE,
    ),
]

_DATE_PATTERNS = [
    re.compile(
        r"(?:released|launched|founded|established)\s+(?:on\s+)?"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
]


def classify_freshness(target: AnswerTarget) -> FactFreshness:
    q = target.original_question.lower()
    if target.attribute == "ceo":
        return FactFreshness.TIME_SENSITIVE
    if target.attribute == "capital":
        return FactFreshness.STRUCTURALLY_STABLE
    if target.attribute in ("revenue", "winner", "date") and target.temporal_scope:
        return FactFreshness.HISTORICAL_STABLE
    if "current" in q or "as of today" in q:
        return FactFreshness.TIME_SENSITIVE
    return FactFreshness.HISTORICAL_STABLE


def cache_key_for_target(target: AnswerTarget) -> str:
    """Minimal cache key abstraction for future reuse."""
    parts = [
        target.attribute.lower(),
        target.entity.lower().strip(),
        (target.temporal_scope or "").lower().strip(),
        target.category or "",
    ]
    return "|".join(p for p in parts if p)


def extract_fact_value(
    evidence_text: str, target: AnswerTarget
) -> StructuredFactValue | None:
    """Extract structured answer value from evidence text."""
    text = evidence_text.strip()
    freshness = classify_freshness(target)

    if target.attribute == "capital":
        for i, pat in enumerate(_CAPITAL_PATTERNS):
            m = pat.search(text)
            if not m:
                continue
            g1, g2 = m.group(1).strip().rstrip("."), m.group(2).strip().rstrip(".")
            if i == 0:
                city, country = g1, g2
            else:
                country, city = g1, g2
            if (
                target.entity.lower() in country.lower()
                or country.lower() in target.entity.lower()
            ):
                return StructuredFactValue(
                    value=_clean_place_name(city),
                    value_type=FactValueType.PLACE,
                    attribute="capital",
                    entity=target.entity.rstrip("?"),
                    freshness=freshness,
                )

    if target.attribute == "winner":
        for pat in _WINNER_PATTERNS:
            m = pat.search(text)
            if m:
                winner = m.group(1).strip()
                allow_single = pat.pattern in _SINGLE_NAME_PATTERNS
                standings_pat = pat.pattern in _STANDINGS_PATTERNS
                if not _is_valid_winner_name(winner, allow_single=allow_single):
                    continue
                if target.temporal_scope and target.temporal_scope not in text:
                    if not re.search(r"\b" + re.escape(target.temporal_scope), text):
                        championship_ctx = any(
                            k in text.lower()
                            for k in ("championship", "world title", "world champion")
                        )
                        if not standings_pat and not championship_ctx:
                            continue
                return StructuredFactValue(
                    value=winner,
                    value_type=FactValueType.PERSON,
                    attribute="winner",
                    entity=target.entity,
                    temporal_scope=target.temporal_scope,
                    category=target.category,
                    freshness=freshness,
                )

    if target.attribute == "revenue":
        for pat in _REVENUE_PATTERNS:
            m = pat.search(text)
            if m:
                amount = m.group(1).strip()
                period = m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else target.temporal_scope
                unit_match = re.search(r"(billion|million|B|M)", amount, re.I)
                unit = unit_match.group(1) if unit_match else None
                currency = "USD" if "$" in amount else None
                qualifiers: list[str] = []
                if "non-gaap" in text.lower():
                    qualifiers.append("non-GAAP")
                if "gaap" in text.lower() and "non-gaap" not in text.lower():
                    qualifiers.append("GAAP")
                if target.temporal_scope and period:
                    if target.temporal_scope.lower() not in (period or "").lower():
                        continue
                return StructuredFactValue(
                    value=re.sub(r"[^\d.]", "", amount.split()[0]) if amount else amount,
                    value_type=FactValueType.NUMBER,
                    attribute="revenue",
                    entity=target.entity,
                    temporal_scope=period or target.temporal_scope,
                    unit=unit,
                    currency=currency,
                    qualifiers=qualifiers,
                    freshness=freshness,
                )

    if target.attribute == "ceo":
        for pat in _CEO_PATTERNS:
            m = pat.search(text)
            if m:
                name = m.group(1).strip()
                if not _is_valid_ceo_name(name):
                    continue
                return StructuredFactValue(
                    value=name,
                    value_type=FactValueType.PERSON,
                    attribute="ceo",
                    entity=target.entity,
                    freshness=FactFreshness.TIME_SENSITIVE,
                )

    if target.attribute == "date":
        for pat in _DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return StructuredFactValue(
                    value=m.group(1).strip(),
                    value_type=FactValueType.DATE,
                    attribute="date",
                    entity=target.entity,
                    freshness=freshness,
                )

    return None


def validate_fact_value_in_evidence(
    fact_value: StructuredFactValue, evidence_text: str
) -> tuple[bool, str]:
    """Verify extracted value is actually present/supported in evidence."""
    text_lower = evidence_text.lower()
    value_lower = fact_value.value.lower()

    if fact_value.value_type == FactValueType.PERSON:
        if value_lower not in text_lower:
            return False, f"Person '{fact_value.value}' not found in evidence"
        return True, "Person name present in evidence"

    if fact_value.value_type == FactValueType.PLACE:
        if value_lower not in text_lower:
            return False, f"Place '{fact_value.value}' not found in evidence"
        return True, "Place name present in evidence"

    if fact_value.value_type == FactValueType.NUMBER:
        # Check numeric core appears
        num = fact_value.value.replace(",", "")
        if num and num not in evidence_text.replace(",", ""):
            # Try with decimals
            if "." in num:
                parts = num.split(".")
                if parts[0] not in evidence_text.replace(",", ""):
                    return False, f"Numeric value '{fact_value.value}' not in evidence"
            else:
                return False, f"Numeric value '{fact_value.value}' not in evidence"
        if fact_value.currency == "USD" and "$" not in evidence_text:
            return False, "USD currency not in evidence"
        if fact_value.temporal_scope:
            scope = fact_value.temporal_scope.lower()
            if scope not in text_lower and not any(
                y in text_lower for y in re.findall(r"\d{4}", scope)
            ):
                return False, f"Temporal scope '{fact_value.temporal_scope}' not in evidence"
        return True, "Numeric value validated in evidence"

    if fact_value.value_type == FactValueType.DATE:
        if value_lower not in text_lower and fact_value.value not in evidence_text:
            return False, f"Date '{fact_value.value}' not found in evidence"
        return True, "Date present in evidence"

    if value_lower in text_lower:
        return True, "Value present in evidence"
    return False, "Value not found in evidence"


def build_canonical_claim_from_value(
    fact_value: StructuredFactValue,
    evidence_text: str | None = None,
) -> str:
    """Deterministic canonical claim from structured fact value."""
    entity = fact_value.entity.rstrip("?")
    temporal = fact_value.temporal_scope or ""
    qualifiers = " ".join(fact_value.qualifiers)
    qual_suffix = f" ({qualifiers})" if qualifiers else ""
    evidence = evidence_text or ""

    if fact_value.attribute == "capital":
        return f"{fact_value.value} is the capital of {entity}."

    if fact_value.attribute == "winner":
        if evidence and re.search(r"\|\s*1\s*\|", evidence):
            if temporal and temporal not in evidence:
                return (
                    f"{fact_value.value} finished in first position in the "
                    f"drivers' championship standings."
                )
            label = entity if temporal in entity else f"{temporal} {entity}".strip()
            return f"{fact_value.value} finished in first position in the {label}."
        comp = entity if temporal in entity else f"{temporal} {entity}".strip()
        return f"{fact_value.value} won the {comp}."

    if fact_value.attribute == "revenue":
        currency_sym = "$" if fact_value.currency == "USD" else ""
        unit_str = f" {fact_value.unit}" if fact_value.unit else ""
        period = temporal or "the specified period"
        return (
            f"{entity} reported {currency_sym}{fact_value.value}{unit_str} "
            f"in revenue for {period}{qual_suffix}."
        )

    if fact_value.attribute == "ceo":
        return f"{fact_value.value} is the CEO of {entity}."

    if fact_value.attribute == "date":
        return f"{entity} was released on {fact_value.value}."

    return f"{fact_value.value} — {fact_value.attribute} of {entity}."


def detect_value_conflicts(
    values: list[StructuredFactValue], target: AnswerTarget
) -> str | None:
    """Detect irreconcilable conflicting target values."""
    valid_values = [v for v in values if v is not None]
    if len(valid_values) < 2:
        return None

    normalized = []
    for v in valid_values:
        key = v.value.lower().strip()
        if v.value_type == FactValueType.NUMBER:
            key = f"{v.value}|{v.temporal_scope}|{v.currency}|{v.unit}"
        normalized.append(key)

    unique = set(normalized)
    if len(unique) > 1:
        vals = [v.value for v in valid_values]
        return f"Conflicting target values: {vals}"
    return None
