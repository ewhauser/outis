"""Safe Harbor age categories for markers and synthetic replacement."""

from __future__ import annotations

import re

from outis.replacement.names.age import parse_english_age


def age_label(value: str) -> str:
    """Return the marker category for an AGE value: AGE_90_PLUS or AGE."""
    match = re.fullmatch(
        r"\s*(\d{1,3}(?:\.\d+)?)\+?(?:[\s-]*(?:years?[\s-]*old|y/?o|years?|yrs?))?\s*",
        value,
        re.I,
    )
    age = float(match[1]) if match else parse_english_age(value)
    # Ages over 89 keep only the Safe Harbor aggregate. The request policy
    # (``ages``) already leaves ages of 89 and under in place by default, so an
    # age that reaches replacement is one the caller asked to remove.
    return "AGE_90_PLUS" if age is not None and age > 89 else "AGE"


def synthetic_age(value: str) -> str:
    """Synthetic AGE replacement."""
    return "90+" if age_label(value) == "AGE_90_PLUS" else "[AGE]"
