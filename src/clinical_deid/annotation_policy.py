"""Versioned annotation contract and executable, occurrence-level examples.

This is an authoring/evaluation policy, not a string blacklist or detector rule.
Offsets are Python Unicode character offsets into the exact authored text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .types import ENTITY_TYPES


POLICY_VERSION = "clinical-phi-v2"
COHORTS = ("realistic", "unusual_name", "long_document")
REQUIRED_METADATA = (
    "annotation_policy", "identity_group", "pair_id", "template_family",
    "phenomena", "cohort",
)
ANNOTATION_POLICY = {
    "version": POLICY_VERSION,
    "offset_unit": "unicode_character",
    "span_end": "exclusive",
    "labels": list(ENTITY_TYPES),
    "cohorts": list(COHORTS),
    "rules": {
        "NAME": "Annotate personal naming occurrences; exclude ordinary pronouns, titles and credentials. Retain internal proper-name punctuation.",
        "DATE": "Annotate patient-specific dates, including partial dates and years; exclude public publication dates and durations.",
        "AGE": "Annotate only the age value when greater than 89, including words and compact forms; exclude units, sex markers and suffixes.",
        "LOCATION": "Annotate the maximal authored locality expression, retaining internal commas; keep facility names separate.",
        "FACILITY": "Annotate the authored facility name, separately from adjacent locality/address spans.",
        "ADDRESS": "Annotate the complete authored address as one span without nesting LOCATION.",
        "IDENTIFIERS": "Use reserved synthetic network identifiers and fictional US phone blocks; IDs are invented, never copied from records.",
    },
    "context_rule": "The same spelling may be PHI or non-PHI in different occurrences; adjudicate context, never blacklist a spelling.",
    "legacy_policy_compatibility": "not_certified",
    "qualification_status": "development_only; independent sealed bank compatibility requires independent policy review",
}


def canonical_identity_group(full_name: str) -> str:
    """Group a canonical full identity and all its aliases before splitting."""
    if not isinstance(full_name, str) or not full_name.strip():
        raise ValueError("Canonical identity must be a nonempty string")
    return hashlib.sha256(unicodedata.normalize("NFC", full_name).casefold().encode("utf-8")).hexdigest()


def identity_split(identity_group: str) -> str:
    """Assign comparable hashed identity pools, without alphabet-based splits."""
    if not isinstance(identity_group, str) or not re.fullmatch(r"[0-9a-f]{64}", identity_group):
        raise ValueError("Identity group must be a lowercase SHA-256 digest")
    return "train" if int(identity_group[:8], 16) % 10 < 8 else "validation"


def _example(key: str, marked: str, phenomena: list[str], cohort: str = "realistic") -> dict:
    pieces, spans, offset, previous = [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", marked):
        literal = marked[previous:match.start()]
        pieces.extend((literal, match[2]))
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(match[2]), "label": match[1]})
        offset += len(match[2])
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError("Malformed policy fixture")
    return {"id": key, "text": text, "spans": spans, "annotation_policy": POLICY_VERSION,
            "phenomena": phenomena, "cohort": cohort}


POLICY_FIXTURES = (
    _example("ordinary_pronouns", "He answered. I, the patient, agreed. She returned for care.", ["ordinary_pronouns"]),
    _example("explicit_unusual_name", "Patient name: [[NAME|He]]. He answered the nurse.", ["explicit_nominal_name", "ordinary_pronouns"], "unusual_name"),
    _example("heading_and_name", "Care history reviewed for the patient named [[NAME|Care]].", ["same_spelling_different_context"], "unusual_name"),
    _example("eponym_and_name", "[[NAME|Mira Wilson]] was assessed for Wilson disease.", ["surname_eponym"]),
    _example("titles_and_unicode", "Dr. [[NAME|Zoë O’Rían-Soto]], MD, met [[NAME|Ana María de la Cruz]].", ["unicode_names", "name_boundaries"]),
    _example("alias_before_full", "[[NAME|Nia]] is the preferred name of [[NAME|Nia Okonkwo]]. She agreed.", ["alias_before_full", "ordinary_pronouns"]),
    _example("patient_partial_dates", "Her surgery was in [[DATE|March 2021]]; she was born in [[DATE|1930]]. A 2021 study described recovery over 14 days.", ["partial_dates", "public_dates", "durations"]),
    _example("complete_date", "Discharged on [[DATE|October 2, 2024]].", ["date_boundaries"]),
    _example("age_numeric_boundaries", "Age [[AGE|92]] years old; [[AGE|91]]-year-old patient; 89 years old at the earlier visit.", ["age_boundary", "ordinary_age"]),
    _example("age_word_boundaries", "A [[AGE|ninety-two]] year old and an eighty-nine-year-old discussed care.", ["word_age", "ordinary_age"]),
    _example("age_compact_boundaries", "[[AGE|92]]F; [[AGE|93]]yo; [[AGE|94]] y/o; 89M.", ["compact_age", "ordinary_age"]),
    _example("maximal_location", "Lives in [[LOCATION|Cedar Haven, UT]]; transferred to [[FACILITY|Silver Finch Medical Center]] in [[LOCATION|North Cedar Haven]].", ["location_boundary", "facility_boundary"]),
    _example("complete_address", "Address: [[ADDRESS|418 Imaginary Lane, Cedar Haven, UT 84000]].", ["address_boundary"]),
    _example("reserved_identifiers", "Contact [[EMAIL|mira@care.example]] or [[PHONE|+1 (202) 555-0148]]; portal [[URL|https://clinic.test/visit]]; host [[IP|192.0.2.17]].", ["reserved_identifiers"]),
)


def annotation_errors(text: object, spans: object) -> list[str]:
    """Return structural defects without guessing semantic annotations."""
    if not isinstance(text, str) or not text.strip():
        return ["text must be a nonempty string"]
    if not isinstance(spans, list):
        return ["spans must be a list"]
    errors, intervals = [], []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"span {index} must be an object")
            continue
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            errors.append(f"span {index} has invalid character offsets")
            continue
        if label not in ENTITY_TYPES:
            errors.append(f"span {index} has unknown label")
        if text[start:end] != text[start:end].strip():
            errors.append(f"span {index} includes leading or trailing whitespace")
        if "text" in span and span["text"] != text[start:end]:
            errors.append(f"span {index} text differs from the annotated substring")
        intervals.append((start, end))
    previous_end = -1
    for start, end in sorted(intervals):
        if start < previous_end:
            errors.append("spans overlap or repeat")
        previous_end = max(previous_end, end)
    return errors
