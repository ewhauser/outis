"""Versioned request policies, kept separate from the trained NER taxonomy.

Custom expressions run in the timeout-capable ``regex`` engine. A policy is
bounded by text, expression, match, and cumulative execution budgets. A failed
policy raises a generic error and never returns a partially filtered result.
"""

from __future__ import annotations

from collections.abc import Iterable
from time import monotonic
from typing import Literal

import regex
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outis.entities import Entity
from outis.types import ENTITY_TYPES

POLICY_VERSION = "1"
MAX_POLICY_TEXT = 1_000_000
MAX_POLICY_MATCHES = 10_000
REGEX_TIMEOUT = 0.025
POLICY_TIMEOUT = 0.25
EXTENDED_TYPES = (
    "CREDIT_CARD",
    "CVV",
    "ROUTING_NUMBER",
    "BANK_ACCOUNT",
    "PASSPORT",
    "DRIVER_LICENSE",
    "PASSWORD",
    "USERNAME",
    "DOB",
)
MEDICAL_TYPES = ("CONDITION", "MEDICATION")
KNOWN_LABELS = frozenset((*ENTITY_TYPES, *EXTENDED_TYPES, *MEDICAL_TYPES))
_IDENTIFIERS = frozenset(ENTITY_TYPES) - {"DATE", "AGE", "LOCATION", "FACILITY"}
_LABEL = regex.compile(r"[A-Z][A-Z0-9_]{0,47}\Z")
_REPETITIONS = regex.compile(r"\{([0-9]*)(?:,([0-9]*))?\}")


def _compile_custom(value):
    # A timeout bounds matching, not compilation. Large counted repetitions
    # can exhaust memory during compilation despite a short source expression.
    # Verbose mode can hide digits behind arbitrary spaces/comments, defeating
    # a source-level repetition bound. It is deliberately unsupported; this
    # catches global/scoped flags and mixed version/ASCII/case flag forms.
    if regex.search(r"\(\?[A-Za-z0-9-]*x[A-Za-z0-9-]*(?=[:)])", value):
        raise ValueError("Verbose custom expressions are unsupported")
    expansion = 1
    for count in _REPETITIONS.finditer(value):
        bound = max((int(part) for part in count.groups() if part), default=1)
        expansion *= max(bound, 1)
        if bound > 10_000 or expansion > 100_000:
            raise ValueError("Custom expression repetition limit exceeded")
    return regex.compile(value)


class CustomEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    label: str = Field(min_length=1, max_length=48)
    pattern: str = Field(min_length=1, max_length=512)

    @field_validator("label")
    @classmethod
    def safe_label(cls, value):
        if not _LABEL.fullmatch(value):
            raise ValueError("Invalid entity label")
        return value

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value):
        try:
            compiled = _compile_custom(value)
            if compiled.search("", timeout=REGEX_TIMEOUT) is not None:
                raise ValueError("Empty custom matches are unsupported")
        except (regex.error, TimeoutError, RecursionError, OverflowError):
            raise ValueError("Invalid custom expression") from None
        return value


class DetectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    profile: Literal["clinical_default", "identifiers_only", "extended"] = (
        "clinical_default"
    )
    include: list[str] | None = Field(default=None, max_length=128)
    exclude: list[str] = Field(default_factory=list, max_length=128)
    allowlist: list[str] = Field(default_factory=list, max_length=128)
    blocklist: list[str] = Field(default_factory=list, max_length=128)
    custom_entities: list[CustomEntity] = Field(default_factory=list, max_length=32)
    medical_terms: dict[Literal["CONDITION", "MEDICATION"], list[str]] = Field(
        default_factory=dict
    )
    # The detector finds every age. "over_89" (default) leaves ages of 89 and
    # under untouched, as the Safe Harbor age rule allows; "all" removes every age.
    ages: Literal["over_89", "all"] = "over_89"

    @field_validator("allowlist", "blocklist")
    @classmethod
    def bounded_terms(cls, values):
        if any(not value or len(value) > 256 for value in values):
            raise ValueError("Policy terms must have 1 to 256 characters")
        return list(dict.fromkeys(values))

    @field_validator("medical_terms")
    @classmethod
    def bounded_medical_terms(cls, value):
        if sum(map(len, value.values())) > 128:
            raise ValueError("Too many medical terms")
        for terms in value.values():
            cls.bounded_terms(terms)
        return value

    @model_validator(mode="after")
    def known_labels(self):
        labels = KNOWN_LABELS | {entity.label for entity in self.custom_entities}
        if any(label not in labels for label in (*(self.include or []), *self.exclude)):
            raise ValueError("Unknown policy entity label")
        return self


def validate_policy(policy: DetectionPolicy | dict | None) -> DetectionPolicy:
    """Validate even constructed/mutated models, without reflecting submitted data."""
    try:
        if isinstance(policy, DetectionPolicy):
            policy = policy.model_dump()
        return DetectionPolicy.model_validate({} if policy is None else policy)
    except (
        ValueError,
        TypeError,
        regex.error,
        TimeoutError,
        RecursionError,
        OverflowError,
    ):
        raise ValueError("Invalid detection policy") from None


def policy_metadata(policy: DetectionPolicy | dict | None) -> dict:
    policy = validate_policy(policy)
    return {
        "profile": policy.profile,
        "version": POLICY_VERSION,
        "custom_entity_count": len(policy.custom_entities),
        "allowlist_count": len(policy.allowlist),
        "blocklist_count": len(policy.blocklist),
        "medical_dictionary_labels": sorted(policy.medical_terms),
        "ages": policy.ages,
    }


def age_value(value: str) -> float | None:
    """Numeric value of an age span such as ``72``, ``72-year-old`` or ``seventy-two``; None if unparsed."""
    from outis.replacement.names.age import parse_english_age

    match = regex.fullmatch(
        r"\s*(\d{1,3}(?:\.\d+)?)\+?(?:[\s-]*(?:years?[\s-]*old|y/?o|years?|yrs?))?\s*",
        value,
        regex.I,
    )
    if match:
        return float(match[1])
    parsed = parse_english_age(value)
    return None if parsed is None else float(parsed)


def _keep_age(text: str, span, ages: str) -> bool:
    """Ages of 89 and under are left in place unless the policy removes all ages; unparsed ages stay removed."""
    if span.label != "AGE" or ages == "all":
        return True
    value = age_value(text[span.start : span.end])
    return value is None or value > 89


# Context-bound patterns deliberately avoid interpreting arbitrary clinical
# numbers as credentials or banking details. The captured value is redacted;
# the public field heading remains outside the entity.
_FIELD = r"[ \t]*(?:number|no\.?|id)?[ \t]*[:=#][ \t]*"
_CODE = r"(?P<value>[A-Z0-9][A-Z0-9._/-]{1,63})(?![\w./-])"
_DATE = (
    r"(?P<value>(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}))(?![\w/-])"
)
_EXTENDED_PATTERNS = [
    ("CREDIT_CARD", r"(?<![\w\d])(?P<value>(?:[0-9][ -]?){12,18}[0-9])(?![\w\d])"),
    (
        "CVV",
        r"\b(?:cvv2?|cvc2?|card security code)"
        + _FIELD
        + r"(?P<value>[0-9]{3,4})(?!\w)",
    ),
    (
        "ROUTING_NUMBER",
        r"\b(?:routing|aba)(?:[ \t]+number)?" + _FIELD + r"(?P<value>[0-9]{9})(?!\w)",
    ),
    ("BANK_ACCOUNT", r"\bbank account" + _FIELD + _CODE),
    # IBANs commonly contain spaces. Capture the complete field; length and
    # checksum validation must never truncate a sensitive identifier.
    (
        "BANK_ACCOUNT",
        r"\biban" + _FIELD + r"(?P<value>[A-Z]{2}[0-9]{2}(?:[ \t]*[A-Z0-9])+)(?!\w)",
    ),
    ("PASSPORT", r"\bpassport" + _FIELD + _CODE),
    (
        "DRIVER_LICENSE",
        r"\b(?:driver(?:'s|’s)?[ -]+licen[cs]e|driving[ -]+licen[cs]e)"
        + _FIELD
        + _CODE,
    ),
    # Credentials may be quoted or contain spaces and punctuation. Once an
    # explicit credential heading is present, cover its full line rather than
    # exposing a suffix that happens to look like ordinary prose.
    (
        "PASSWORD",
        r"\b(?:password|passwd|passphrase|api[ _-]?key|access[ _-]?token)"
        + _FIELD
        + r"(?P<value>[^\r\n]+)",
    ),
    (
        "USERNAME",
        r"\b(?:username|user[ -]?name|login)"
        + _FIELD
        + r"(?P<value>[\w@.+-]{1,128})(?![\w@.+-])",
    ),
    ("DOB", r"\b(?:dob|date[ -]+of[ -]+birth|birth[ -]+date)" + _FIELD + _DATE),
]
_EXTENDED_PATTERNS = [
    (label, regex.compile(pattern, regex.I)) for label, pattern in _EXTENDED_PATTERNS
]


def _copy_entity(span) -> Entity:
    return Entity(
        span.start,
        span.end,
        span.label,
        span.score,
        span.source,
        getattr(span, "entity_id", None),
    )


def _merge_coverage(entities: list[Entity]) -> list[Entity]:
    """Union every connected overlap; choosing a label never drops a suffix.

    Explicit forced rules take precedence over custom expressions, which take
    precedence over finer built-in types and model labels. Equal-priority
    overlaps prefer the longest detection, then lexical label for stability.
    A medical concept label never wins over an overlapping identifier: the
    concept types are publicly restorable and may be kept, so letting one
    relabel a merged span would expose the identifier it covers.
    """
    precedence = {
        "policy:blocklist": 4,
        "policy:custom": 3,
        "policy:extended": 2,
        "policy:medical_dictionary": 2,
    }

    def choose(group):
        candidates = [item for item in group if item.label not in MEDICAL_TYPES]
        winner = max(
            candidates or group,
            key=lambda item: (
                precedence.get(item.source, 0),
                item.end - item.start,
                item.score,
                item.label,
            ),
        )
        start, end = min(item.start for item in group), max(item.end for item in group)
        if len(group) == 1:
            return winner
        return Entity(start, end, winner.label, winner.score, winner.source, None)

    result, group = [], []
    edge = -1
    for entity in sorted(entities, key=lambda item: (item.start, item.end, item.label)):
        if group and entity.start >= edge:
            result.append(choose(group))
            group = []
        group.append(entity)
        edge = max(edge, entity.end)
    if group:
        result.append(choose(group))
    return result


def apply_policy(
    text: str, spans: Iterable, policy: DetectionPolicy | dict | None = None
) -> list[Entity]:
    policy = validate_policy(policy)
    if not isinstance(text, str) or len(text) > MAX_POLICY_TEXT:
        raise ValueError("Policy text exceeds the supported limit")
    original = []
    try:
        for span in spans:
            if len(original) >= MAX_POLICY_MATCHES:
                raise ValueError("Detection count exceeds the supported limit")
            original.append(_copy_entity(span))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Invalid detection output") from None
    if any(span.end > len(text) for span in original):
        raise ValueError("Invalid detection offsets")
    original = [span for span in original if _keep_age(text, span, policy.ages)]
    if policy == DetectionPolicy():
        return original

    enabled = set(ENTITY_TYPES)
    if policy.profile == "identifiers_only":
        enabled = set(_IDENTIFIERS) | set(EXTENDED_TYPES) - {"DOB"}
    elif policy.profile == "extended":
        enabled.update(EXTENDED_TYPES)
    enabled.update(item.label for item in policy.custom_entities)
    enabled.update(policy.medical_terms)
    if policy.include is not None:
        enabled = set(policy.include)
    enabled.difference_update(policy.exclude)
    exemptions = set(policy.allowlist)
    entities = [
        span
        for span in original
        if span.label in enabled and text[span.start : span.end] not in exemptions
    ]
    deadline = monotonic() + POLICY_TIMEOUT

    def matches(compiled, *, value_group=False):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ValueError("Policy processing budget exceeded")
        try:
            for match in compiled.finditer(text, timeout=min(REGEX_TIMEOUT, remaining)):
                if monotonic() >= deadline:
                    raise ValueError("Policy processing budget exceeded")
                start, end = match.span("value" if value_group else 0)
                if start == end:
                    raise ValueError("Empty custom matches are unsupported")
                yield start, end
        except TimeoutError:
            raise ValueError("Policy processing budget exceeded") from None

    def add(start, end, label, source):
        if source != "policy:blocklist" and text[start:end] in exemptions:
            return
        if len(entities) >= MAX_POLICY_MATCHES:
            raise ValueError("Detection count exceeds the supported limit")
        entities.append(Entity(start, end, label, 1.0, source))

    for label, compiled in _EXTENDED_PATTERNS:
        if label in enabled:
            for start, end in matches(compiled, value_group=True):
                add(start, end, label, "policy:extended")
    for custom in policy.custom_entities:
        if custom.label in enabled:
            for start, end in matches(_compile_custom(custom.pattern)):
                add(start, end, custom.label, "policy:custom")
    for label, terms in policy.medical_terms.items():
        if label in enabled and terms:
            compiled = regex.compile(
                r"(?<!\w)(?:"
                + "|".join(
                    regex.escape(term)
                    for term in sorted(set(terms), key=len, reverse=True)
                )
                + r")(?!\w)",
                regex.I,
            )
            for start, end in matches(compiled):
                add(start, end, label, "policy:medical_dictionary")
    for literal in policy.blocklist:
        for start, end in matches(regex.compile(regex.escape(literal))):
            add(start, end, "ID", "policy:blocklist")
    return _merge_coverage(entities)
