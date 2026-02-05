"""PII redaction utility for logging (Rule 7: Security & Data Sanctity)."""

import re
from typing import Any


# Common PII patterns
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),  # SSN
    (r'\b\d{16}\b', 'CREDIT_CARD'),  # Credit card (16 digits)
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),  # Email
    (r'\b\d{3}-\d{3}-\d{4}\b', 'PHONE'),  # Phone (US format)
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDRESS'),  # IP address
    (r'\b[A-Z]{2}\d{6}\b', 'PASSPORT'),  # Passport (simplified)
]

# API key patterns (common prefixes)
API_KEY_PATTERNS = [
    (r'sk-[A-Za-z0-9]{32,}', 'ANTHROPIC_API_KEY'),
    (r'tvly-[A-Za-z0-9]{32,}', 'TAVILY_API_KEY'),
    (r'[A-Za-z0-9]{32,}', 'GENERIC_API_KEY'),  # Generic long alphanumeric
]


def redact_pii(text: str, replacement: str = '[REDACTED]') -> str:
    """
    Redact PII and API keys from text for safe logging.

    Args:
        text: Input text that may contain PII
        replacement: String to replace PII with

    Returns:
        Text with PII redacted
    """
    redacted = text

    # Redact PII patterns
    for pattern, pii_type in PII_PATTERNS:
        redacted = re.sub(pattern, f'{replacement} ({pii_type})', redacted, flags=re.IGNORECASE)

    # Redact API keys (more aggressive)
    for pattern, key_type in API_KEY_PATTERNS:
        redacted = re.sub(pattern, f'{replacement} ({key_type})', redacted, flags=re.IGNORECASE)

    return redacted


def redact_dict(data: dict[str, Any], replacement: str = '[REDACTED]') -> dict[str, Any]:
    """
    Recursively redact PII from dictionary (for logging prompts/responses).

    Args:
        data: Dictionary that may contain PII
        replacement: String to replace PII with

    Returns:
        Dictionary with PII redacted
    """
    redacted = {}
    for key, value in data.items():
        # Skip redaction for certain safe keys
        if key in ['model', 'temperature', 'max_tokens']:
            redacted[key] = value
        elif isinstance(value, str):
            redacted[key] = redact_pii(value, replacement)
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value, replacement)
        elif isinstance(value, list):
            redacted[key] = [
                redact_dict(item, replacement) if isinstance(item, dict)
                else redact_pii(item, replacement) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            redacted[key] = value

    return redacted
