"""Conservative, inspectable analysis of already detected English entities.

This module never expands detector coverage or infers clinical facts. Parsing
requires the complete detected value to match a supported grammar. Normalized
dates, ages and address components are sensitive values and require opt-in.
Relationships require an explicit supported predicate between adjacent spans;
they are experimental rule matches, not clinically validated assertions.
"""

from __future__ import annotations

import re
from datetime import date

from clinical_deid.entities import checked_entities

ANALYSIS_VERSION = "1"
SUPPORTED_LOCALES = ("en-US", "en-GB")
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sept", "sep"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        1,
    )
    for name in names
}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY AS GU MP PR VI AA AE AP".split()
)
_ADDRESS = re.compile(
    r"(?P<street_number>[0-9]{1,8}[A-Za-z]?(?:-[0-9]{1,8})?) "
    r"(?P<street>[^,\r\n]{2,100}),\s*"
    r"(?P<city>[A-Za-z][A-Za-z .'-]{0,63}),\s*"
    r"(?P<state>[A-Z]{2})\s+(?P<postal_code>[0-9]{5}(?:-[0-9]{4})?)"
    r"(?:,?\s+(?P<country>USA|US|United States))?",
    re.I,
)
_UNIT = re.compile(
    r"(?P<street>.+?)\s+(?P<unit>(?:apt\.?|suite|ste\.?|unit|#)\s*[A-Za-z0-9-]+)", re.I
)


def _date_analysis(value, locale, include_sensitive):
    value = value.strip()
    result = {"kind": "date", "status": "unsupported"}
    year = month = day = None
    precision = "day"
    match = re.fullmatch(r"([0-9]{4})([-/])([0-9]{1,2})\2([0-9]{1,2})", value)
    if match:
        year, month, day = int(match[1]), int(match[3]), int(match[4])
    else:
        match = re.fullmatch(r"([0-9]{1,2})([-/])([0-9]{1,2})\2([0-9]{4})", value)
        if match:
            first, second, year = int(match[1]), int(match[3]), int(match[4])
            month, day = (first, second) if locale == "en-US" else (second, first)
            result["interpretation"] = (
                "month_first" if locale == "en-US" else "day_first"
            )
            result["locale_dependent"] = (
                first <= 12 and second <= 12 and first != second
            )
        else:
            match = re.fullmatch(
                rf"({_MONTH})\s+([0-9]{{1,2}})(?:st|nd|rd|th)?(?:,?\s+)([0-9]{{4}})",
                value,
                re.I,
            )
            if match:
                month, day, year = (
                    _MONTHS[match[1].rstrip(".").lower()],
                    int(match[2]),
                    int(match[3]),
                )
            else:
                match = re.fullmatch(
                    rf"([0-9]{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s+)([0-9]{{4}})",
                    value,
                    re.I,
                )
                if match:
                    day, month, year = (
                        int(match[1]),
                        _MONTHS[match[2].rstrip(".").lower()],
                        int(match[3]),
                    )
                else:
                    match = re.fullmatch(rf"({_MONTH})\s+([0-9]{{4}})", value, re.I)
                    if match:
                        month, year = (
                            _MONTHS[match[1].rstrip(".").lower()],
                            int(match[2]),
                        )
                        precision = "month"
                    elif re.fullmatch(r"[0-9]{4}", value):
                        year, precision = int(value), "year"
                    elif re.fullmatch(
                        r"[0-9]{1,2}[-/][0-9]{1,2}(?:[-/][0-9]{2})?", value
                    ):
                        return {
                            "kind": "date",
                            "status": "ambiguous",
                            "reason": "missing_four_digit_year",
                        }
                    else:
                        return result
    ordinal = re.search(r"(?<![0-9])([0-9]{1,2})(st|nd|rd|th)\b", value, re.I)
    if ordinal:
        number = int(ordinal[1])
        suffix = (
            "th"
            if number % 100 in {11, 12, 13}
            else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        )
        if ordinal[2].lower() != suffix:
            result["status"] = "invalid"
            return result
    try:
        parsed = date(year, month or 1, day or 1)
        # Zero is invalid, not an omitted calendar component.
        if month == 0 or day == 0:
            raise ValueError
    except (ValueError, TypeError):
        result["status"] = "invalid"
        return result
    result.update(status="parsed", precision=precision)
    if include_sensitive:
        result["normalized"] = parsed.isoformat()[
            : {"day": 10, "month": 7, "year": 4}[precision]
        ]
    return result


def _age_analysis(value, include_sensitive):
    value = value.strip().lower()
    result = {"kind": "age", "status": "unsupported"}
    older = re.fullmatch(r"([0-9]{1,3})\+\s*(?:years?(?: old)?)?", value)
    match = re.fullmatch(
        r"(?:aged?\s+)?([0-9]{1,4})(?:\s*[- ]?\s*"
        r"(years?|yrs?|y/?o|months?|mos?|weeks?|wks?|days?)"
        r"(?:\s*[- ]?\s*old)?)?",
        value,
    )
    if older:
        amount, unit = int(older[1]), "years"
        normalized = {"minimum": amount, "unit": unit, "inclusive": True}
    elif match:
        amount = int(match[1])
        raw_unit = match[2] or "years"
        unit = next(
            (
                name
                for prefix, name in (
                    ("y", "years"),
                    ("mo", "months"),
                    ("w", "weeks"),
                    ("d", "days"),
                )
                if raw_unit.startswith(prefix)
            ),
            "years",
        )
        normalized = {"value": amount, "unit": unit}
    else:
        return result
    if amount > {"years": 130, "months": 1560, "weeks": 6783, "days": 47483}[unit]:
        result["status"] = "invalid"
        return result
    result.update(status="parsed", unit=unit)
    if include_sensitive:
        result["normalized"] = normalized
    return result


def _address_analysis(value, include_sensitive):
    result = {"kind": "address", "status": "unsupported"}
    match = _ADDRESS.fullmatch(value.strip())
    if not match or match["state"].upper() not in _STATES:
        return result
    components = {key: item.strip() for key, item in match.groupdict().items() if item}
    components["state"] = components["state"].upper()
    if "country" in components:
        components["country"] = "US"
    unit = _UNIT.fullmatch(components["street"])
    if unit:
        components["street"], components["unit"] = unit["street"], unit["unit"]
    result.update(status="parsed", format="us_postal")
    if include_sensitive:
        result["components"] = components
    return result


def _luhn(value):
    if not re.fullmatch(r"[0-9][0-9 -]*[0-9]", value):
        return False
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _routing_checksum(value):
    if not re.fullmatch(r"[0-9]{9}", value) or value == "000000000":
        return False
    return (
        sum(int(digit) * weight for digit, weight in zip(value, (3, 7, 1) * 3)) % 10
        == 0
    )


_RELATIONS = (
    ("NAME", {"DATE", "DOB"}, r"\s+(?:was\s+)?born\s+on\s+", "born_on"),
    ("NAME", {"ADDRESS"}, r"\s+(?:lives|resides)\s+at\s+", "resides_at"),
    (
        "NAME",
        {"PHONE", "EMAIL"},
        r"\s+(?:can|may)\s+be\s+reached\s+at\s+",
        "contact_at",
    ),
    ("NAME", {"AGE"}, r"\s+(?:is\s+aged|is\s+age|is)\s+", "has_age"),
    ("NAME", {"CONDITION"}, r"\s+(?:has|was\s+diagnosed\s+with)\s+", "has_condition"),
    ("NAME", {"MEDICATION"}, r"\s+(?:takes|was\s+prescribed)\s+", "takes_medication"),
)


def _asserted_clause(text, left, right):
    """Reject visible uncertainty rather than guessing its syntactic scope."""
    before = re.split(r"[.!?;\r\n]", text[max(0, left.start - 512) : left.start])[-1]
    after = re.split(r"[.!;\r\n]", text[right.end : right.end + 256])[0]
    if "?" in after:
        return False
    uncertain = r"\b(?:if|unless|whether|suppose|assuming|could|might|possible|suspected|uncertain|not|no|never|denies|denied|negative)\b"
    if re.search(uncertain, before + " " + after, re.I):
        return False
    return not re.search(
        r"\b(?:rule[ds]?\s+out|family|mother|father|sibling|brother|sister)\b",
        before + " " + after,
        re.I,
    )


def analyze_entities(
    text, entities, *, locale="en-US", include_sensitive=False
) -> dict:
    """Augment entities; input/output offsets are Python Unicode character offsets.

    Entity values are omitted by default, including normalized sensitive values.
    Validation results establish only syntax/checksum validity, never issuance,
    identity, account ownership, or clinical truth.
    """
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("Invalid analysis text")
    if locale not in SUPPORTED_LOCALES or type(include_sensitive) is not bool:
        raise ValueError("Invalid analysis options")
    entities = checked_entities(text, entities)
    analyzed = []
    for entity in entities:
        item = entity.as_dict()
        value = text[entity.start : entity.end]
        if entity.label in {"DATE", "DOB"}:
            item["analysis"] = _date_analysis(value, locale, include_sensitive)
        elif entity.label == "AGE":
            item["analysis"] = _age_analysis(value, include_sensitive)
        elif entity.label == "ADDRESS":
            item["analysis"] = _address_analysis(value, include_sensitive)
        elif entity.label == "CREDIT_CARD":
            item["analysis"] = {
                "kind": "credit_card",
                "status": "validated",
                "checksum_valid": _luhn(value),
                "algorithm": "luhn",
            }
        elif entity.label == "ROUTING_NUMBER":
            item["analysis"] = {
                "kind": "routing_number",
                "status": "validated",
                "checksum_valid": _routing_checksum(value),
                "algorithm": "aba",
            }
        analyzed.append(item)
    relationships = []
    for index, (left, right) in enumerate(zip(entities, entities[1:])):
        # Never infer cross-line associations or copy free text into metadata.
        between = text[left.end : right.start]
        if len(between) > 64 or "\n" in between or "\r" in between:
            continue
        for left_label, right_labels, predicate, relation in _RELATIONS:
            if (
                left.label == left_label
                and right.label in right_labels
                and re.fullmatch(predicate, between, re.I)
            ):
                if not _asserted_clause(text, left, right):
                    continue
                relationships.append(
                    {
                        "type": relation,
                        "source_entity": index,
                        "target_entity": index + 1,
                        "source": "rule:explicit_relation",
                    }
                )
                break
    return {
        "entities": analyzed,
        "relationships": relationships,
        "metadata": {
            "analysis_version": ANALYSIS_VERSION,
            "locale": locale,
            "sensitive_values_included": include_sensitive,
            "relationship_extraction": "explicit_templates",
            "relationship_status": "experimental",
        },
    }
