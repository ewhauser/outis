"""Confidentiality-safe redaction and patient-consistent fictional replacement.

All mappings are stateless keyed HMACs.  Neither the mapping key, patient anchor,
source substrings, nor the date shift are returned to callers.  Exact dates are
shifted by one patient-wide offset; imprecise or unsupported dates are redacted
instead of inventing chronology that the source did not establish.
"""

from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

from clinical_deid.detection.name_aliases import full_identity_aliases
from clinical_deid.replacement.nicknames import nickname_bindings
from clinical_deid.types import Span

_FIRST_NAMES = (
    "Avery",
    "Morgan",
    "Rowan",
    "Quinn",
    "Casey",
    "Riley",
    "Taylor",
    "Jordan",
    "Alex",
    "Drew",
    "Robin",
    "Cameron",
    "Sage",
    "Harper",
    "Skyler",
    "Emery",
    "Reese",
    "Parker",
    "Finley",
    "Dakota",
    "Blair",
    "Charlie",
    "Jamie",
    "Kendall",
    "Logan",
    "Hayden",
    "Ellis",
    "Remy",
    "Arden",
    "Lane",
    "Marley",
    "Sidney",
)
_LAST_NAMES = (
    "Alder",
    "Aspen",
    "Birch",
    "Cedar",
    "Cobalt",
    "Cypress",
    "Elm",
    "Fern",
    "Flint",
    "Grove",
    "Haven",
    "Juniper",
    "Laurel",
    "Linden",
    "Maple",
    "Meadow",
    "Moss",
    "North",
    "Oak",
    "Orchard",
    "Pine",
    "Quartz",
    "Reed",
    "River",
    "Stone",
    "Vale",
    "West",
    "Willow",
    "Wren",
    "Brook",
    "Cove",
    "Sable",
)
_MONTHS = tuple(calendar.month_name)
_MONTH_ABBREVIATIONS = tuple(calendar.month_abbr)
_MONTH_LOOKUP = {
    variant.casefold(): number
    for number in range(1, 13)
    for variant in (_MONTHS[number], _MONTH_ABBREVIATIONS[number])
}
_MONTH_LOOKUP["sept"] = 9
_MONTH_PATTERN = "(?:" + "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True)) + ")"
_TITLE = re.compile(r"^(?P<title>(?:dr|mr|mrs|ms|mx|miss|prof)\.?\s+)", re.I)
_NAME_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_NAME_SUFFIX = re.compile(r"(?:jr|sr|ii|iii|iv|v|vi)\.?\Z", re.I)
_NAME_PUNCTUATION = str.maketrans({"’": "'", "‐": "-", "‑": "-"})
_EXACT_IDENTITY_LABELS = frozenset({"MRN", "ACCOUNT", "ID", "EMAIL", "URL"})
_FACILITY_SPECIALTIES = (
    (
        "Pediatric",
        re.compile(r"\b(?:children(?:['’]s|s)?|pediatric|paediatric)\b", re.I),
    ),
    ("Family", re.compile(r"\bfamily\b", re.I)),
    ("Rehabilitation", re.compile(r"\brehabilitation\b", re.I)),
    ("Dialysis", re.compile(r"\bdialysis\b", re.I)),
    ("Infusion", re.compile(r"\binfusion\b", re.I)),
    ("Oncology", re.compile(r"\boncology\b", re.I)),
    ("Cardiology", re.compile(r"\bcardiology\b", re.I)),
    ("Psychiatric", re.compile(r"\b(?:psychiatric|mental\s+health)\b", re.I)),
    ("Hospice", re.compile(r"\bhospice\b", re.I)),
    ("Wound Care", re.compile(r"\bwound\s+care\b", re.I)),
    ("Wound", re.compile(r"\bwound\b(?!\s+care\b)", re.I)),
)
_FACILITY_TYPES = (
    ("House", re.compile(r"\bhospice\s+house\b", re.I)),
    ("Care Residence", re.compile(r"\bcare\s+(?:residence|lodge)\b", re.I)),
    ("Nursing Home", re.compile(r"\bnursing\s+home\b", re.I)),
    ("Hospital", re.compile(r"\bhospital\b", re.I)),
    ("Infirmary", re.compile(r"\binfirmary\b", re.I)),
    (
        "Pavilion",
        re.compile(
            r"\b(?:medical|outpatient|diagnostic|infusion|wound(?:\s+care)?)\s+pavilion\b",
            re.I,
        ),
    ),
    ("Clinic", re.compile(r"\bclinic\b", re.I)),
    ("Unit", re.compile(r"\bunit\b", re.I)),
    ("Medical Center", re.compile(r"\bmedical\s+cent(?:er|re)\b", re.I)),
    ("Center", re.compile(r"\bcent(?:er|re)\b", re.I)),
    ("Practice", re.compile(r"\bpractice\b", re.I)),
)


def _canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(
    secret: bytes, patient: str, label: str, value: str, salt: int = 0
) -> bytes:
    message = json.dumps(
        ["clinical-deid-replacement-v1", patient, label, value, salt],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


def _validated_secret(secret: bytes | str | None) -> bytes:
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError(
            "Synthetic replacement requires a secret containing at least 32 bytes"
        )
    return secret


def _validate_spans(text: str, spans: list[Span]) -> list[Span]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    previous_end = 0
    for span in ordered:
        if (
            not isinstance(span.start, int)
            or isinstance(span.start, bool)
            or not isinstance(span.end, int)
            or isinstance(span.end, bool)
            or span.start < previous_end
            or span.start < 0
            or span.end <= span.start
            or span.end > len(text)
        ):
            raise ValueError("PHI spans must be nonoverlapping valid character ranges")
        previous_end = span.end
    return ordered


def _safe_label(label: str) -> str:
    # Unknown model labels can be redacted, but never interpolated as free text.
    return (
        label
        if isinstance(label, str) and re.fullmatch(r"[A-Z_]{1,40}", label)
        else "PHI"
    )


def _name_identity(value: str) -> tuple[str, str]:
    title_match = _TITLE.match(value)
    title = title_match.group("title") if title_match else ""
    # These supported spelling variants are equivalent only for personal names.
    # Opaque identifiers and the caller's patient anchor retain their exact keys.
    body = (
        unicodedata.normalize("NFKC", value[len(title) :])
        .translate(_NAME_PUNCTUATION)
        .strip()
    )
    # Support both common full-name orders without producing two identities.
    if body.count(",") == 1:
        last, first = body.split(",", 1)
        if last.strip() and first.strip():
            if _NAME_SUFFIX.fullmatch(first.strip()) and len(last.split()) >= 2:
                # A sole suffix after a comma is decoration, not surname-first
                # order. Keep it in the HMAC identity to distinguish Jr/Sr.
                body = last.strip() + " " + first.strip()
            elif (
                full_identity_aliases(body)
                and len(first.split()) > 1
                and _NAME_SUFFIX.fullmatch(first.split()[-1])
            ):
                given, suffix = first.strip().rsplit(None, 1)
                body = given + " " + last.strip() + " " + suffix
            else:
                body = first.strip() + " " + last.strip()
    words = body.split()
    if len(words) >= 3 and _NAME_SUFFIX.fullmatch(words[-1]):
        suffix = words[-1].rstrip(".")
        # Preserve the existing conventional ASCII HMAC keys: Jr./Sr. carry a
        # period, while Roman numeral suffixes do not. Optional periods must not
        # make one person's reduced aliases appear to have competing owners.
        words[-1] = suffix + "." if suffix.casefold() in {"jr", "sr"} else suffix
        body = " ".join(words)
    return title, _canonical(body)


def _name_words(value: str) -> list[str]:
    """Read whole Unicode words; retain the exact legacy ASCII tokenization."""
    if value.isascii():
        return _NAME_WORD.findall(value)
    words = []
    position = 0
    while position < len(value):
        if not value[position].isalpha():
            position += 1
            continue
        start = position
        position += 1
        while position < len(value):
            if value[position].isalpha() or unicodedata.category(
                value[position]
            ).startswith("M"):
                position += 1
            elif (
                value[position] in "'’‐‑-"
                and position + 1 < len(value)
                and value[position + 1].isalpha()
            ):
                position += 2
            else:
                break
        words.append(value[start:position])
    return words


def _fictional_name(secret: bytes, patient: str, identity: str) -> tuple[str, str]:
    original_words = _name_words(identity)
    expressed_parts = full_identity_aliases(identity)
    family_words = _name_words(expressed_parts[-1]) if len(expressed_parts) == 2 else []
    original_first = (
        _canonical(expressed_parts[0])
        if len(expressed_parts) == 2
        else original_words[0]
        if original_words
        else None
    )
    original_last = (
        family_words[-1]
        if family_words
        else original_words[-1]
        if original_words
        else None
    )
    for salt in range(100):
        digest = _digest(secret, patient, "NAME", identity, salt)
        first = _FIRST_NAMES[int.from_bytes(digest[:4], "big") % len(_FIRST_NAMES)]
        last = _LAST_NAMES[int.from_bytes(digest[4:8], "big") % len(_LAST_NAMES)]
        # Partial follow-up mentions must also change, even if only one original
        # component happened to collide with a candidate fictional component.
        if (
            _canonical(first + " " + last) != identity
            and first.casefold() != original_first
            and last.casefold() != original_last
        ):
            return first, last
    # This path is unreachable with the current vocabulary, but remains safe if
    # a downstream caller replaces the vocabulary with a single original name.
    return "Synthetic", "Person"


def _name_aliases(text: str, spans: list[Span]) -> dict[str, tuple[str, str]]:
    """Resolve first/last mentions only when they identify one observed full name."""
    aliases: dict[str, set[tuple[str, str]]] = {}
    for span in spans:
        if span.label != "NAME":
            continue
        _, identity = _name_identity(text[span.start : span.end])
        parts = full_identity_aliases(identity)
        # Initials, ambiguous multipart names and a shared first/family spelling
        # cannot establish which component a reduced mention represents.
        if len(parts) == 2:
            aliases.setdefault(_canonical(parts[0]), set()).add((identity, "first"))
            aliases.setdefault(_canonical(parts[1]), set()).add((identity, "last"))
    return {
        alias: next(iter(matches))
        for alias, matches in aliases.items()
        if len(matches) == 1
    }


def _numeric_date(value: str, offset: int) -> tuple[str | None, str | None]:
    iso = re.fullmatch(
        r"(?P<y>\d{4})(?P<s>[-/.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})", value
    )
    american = re.fullmatch(
        r"(?P<m>\d{1,2})(?P<s>[-/.])(?P<d>\d{1,2})(?P=s)(?P<y>\d{4})", value
    )
    match = iso or american
    if not match:
        return None, None
    # With year last, an initial value above 12 establishes day/month order.
    # Values that could be either convention retain the documented US policy.
    day_first = american is not None and int(match["m"]) > 12
    month = int(match["d"] if day_first else match["m"])
    day = int(match["m"] if day_first else match["d"])
    try:
        shifted = date(int(match["y"]), month, day) + timedelta(days=offset)
    except (ValueError, OverflowError):
        return None, "Invalid or out-of-range date was redacted"
    month_width = len(match["d"] if day_first else match["m"])
    source_day = match["m"] if day_first else match["d"]
    day_width = (
        2
        if len(source_day) == 2 and (source_day.startswith("0") or month_width == 2)
        else 1
    )
    values = {
        "y": str(shifted.year).zfill(4),
        "m": str(shifted.month).zfill(month_width),
        "d": str(shifted.day).zfill(day_width),
    }
    order = (
        ("y", "m", "d") if iso else ("d", "m", "y") if day_first else ("m", "d", "y")
    )
    warning = None
    if day_first and shifted.day <= 12 and shifted.day != shifted.month:
        # A D/M input can be unambiguous while the shifted D/M output is not.
        # The documented M/D default would then change its calendar meaning.
        # Retain the separator/padding but make the date order explicit.
        order = ("y", "m", "d")
        warning = "Shifted day/month/year date was emitted as year/month/day to avoid ambiguity"
    elif american and int(match["m"]) <= 12 and int(match["d"]) <= 12:
        warning = "Ambiguous numeric date was interpreted as month/day/year"
    rendered = match["s"].join(values[part] for part in order)
    return rendered, warning


def _datetime_date(value: str, offset: int) -> tuple[str | None, str | None]:
    match = re.fullmatch(
        r"(?P<date>\d{4}-\d{2}-\d{2})"
        r"(?P<tail>[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
        r"(?P<zone>[Zz]|[+-]\d{2}:?\d{2})?)",
        value,
    )
    if not match:
        return None, None
    try:
        datetime.fromisoformat(
            value.removesuffix("z") + ("Z" if value.endswith("z") else "")
        )
        zone = match["zone"]
        if zone and zone[0] in "+-":
            digits = zone[1:].replace(":", "")
            if int(digits[:2]) > 23 or int(digits[2:]) > 59:
                raise ValueError("Invalid time zone")
    except ValueError:
        return None, "Invalid or out-of-range date-time was redacted"
    rendered, warning = _numeric_date(match["date"], offset)
    if rendered is None:
        return None, warning or "Invalid or out-of-range date-time was redacted"
    return rendered + match["tail"], warning


def _case_like(value: str, example: str) -> str:
    if example.isupper():
        return value.upper()
    if example.islower():
        return value.lower()
    return value


def _word_date(value: str, offset: int) -> tuple[str | None, str | None]:
    month_first = re.fullmatch(
        rf"(?P<m>{_MONTH_PATTERN})(?P<dot>\.?)"
        r"(?P<space>\s+|-)(?P<d>\d{1,2})(?P<ordinal>st|nd|rd|th)?"
        r"(?P<join>,\s*|\s+|-)(?P<y>\d{4})",
        value,
        re.I,
    )
    day_first = re.fullmatch(
        rf"(?P<d>\d{{1,2}})(?P<ordinal>st|nd|rd|th)?(?P<space>\s+|-)"
        rf"(?P<m>{_MONTH_PATTERN})(?P<dot>\.?)(?P<join>,\s*|\s+|-)(?P<y>\d{{4}})",
        value,
        re.I,
    )
    match = month_first or day_first
    if not match:
        return None, None
    month = _MONTH_LOOKUP[match["m"].casefold()]
    try:
        shifted = date(int(match["y"]), month, int(match["d"])) + timedelta(days=offset)
    except (ValueError, OverflowError):
        return None, "Invalid or out-of-range date was redacted"
    hyphenated = match["space"] == "-" or match["join"] == "-"
    abbreviated = len(match["m"]) <= 4 and (
        match["m"].casefold() != "may" or hyphenated
    )
    month_text = (
        _MONTH_ABBREVIATIONS[shifted.month] if abbreviated else _MONTHS[shifted.month]
    )
    if match["m"].casefold() == "sept" and shifted.month == 9:
        month_text = "Sept"
    month_text = _case_like(month_text, match["m"]) + match["dot"]
    # Month-name styles ordinarily use an unpadded day.  Preserve an explicit
    # zero-padded style when the source actually gives evidence of that style.
    day_text = str(shifted.day).zfill(
        2 if match["d"].startswith("0") or (hyphenated and len(match["d"]) == 2) else 1
    )
    if match["ordinal"]:
        suffix = (
            "th"
            if 10 <= shifted.day % 100 <= 20
            else {1: "st", 2: "nd", 3: "rd"}.get(shifted.day % 10, "th")
        )
        day_text += _case_like(suffix, match["ordinal"])
    year_text = str(shifted.year).zfill(4)
    if month_first:
        return month_text + match["space"] + day_text + match["join"] + year_text, None
    return day_text + match["space"] + month_text + match["join"] + year_text, None


def _shift_date(value: str, offset: int) -> tuple[str, str | None, bool]:
    # Outer spacing is part of the selected range and can be safely preserved.
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    stripped = value.strip()
    for parser in (_datetime_date, _numeric_date, _word_date):
        rendered, warning = parser(stripped, offset)
        if rendered is not None:
            return leading + rendered + trailing, warning, True
        if warning is not None:
            return "[DATE]", warning, False
    return (
        "[DATE]",
        "Incomplete or unsupported date was redacted; chronology cannot be preserved",
        False,
    )


def _age(value: str, mode: str) -> str:
    from clinical_deid.detection.context.age import parse_english_age

    match = re.fullmatch(
        r"\s*(\d{1,3}(?:\.\d+)?)\+?(?:[\s-]*(?:years?[\s-]*old|y/?o|years?|yrs?))?\s*",
        value,
        re.I,
    )
    if not match:
        age = parse_english_age(value)
        if age is None:
            return "[AGE]"
    else:
        age = float(match[1])
    if age > 89:
        return "[AGE_90_PLUS]" if mode == "redact" else "90+"
    # Ages 0--89 are clinical information and are not age identifiers under the
    # Safe Harbor age rule.  Only explicitly supplied spans affect redact mode.
    return "[AGE]" if mode == "redact" else value


def _facility_setting(value: str) -> str:
    """Emit only fixed public setting terms, never source proper-name tokens."""
    specialties = [
        safe_term
        for safe_term, pattern in _FACILITY_SPECIALTIES
        if pattern.search(value)
    ]
    facility_type = next(
        (safe_term for safe_term, pattern in _FACILITY_TYPES if pattern.search(value)),
        None,
    )
    if facility_type is None:
        if specialties == ["Hospice"]:
            return "Hospice"
        facility_type = "Medical Center"
    return " ".join(specialties + [facility_type])


def _synthetic_value(
    value: str,
    label: str,
    secret: bytes,
    patient: str,
    aliases: dict[str, tuple[str, str]],
    *,
    name_binding=None,
    name_warnings=None,
) -> str:
    canonical = _canonical(value)
    # Opaque identifiers, email local parts and URL paths can distinguish case,
    # whitespace or Unicode spelling. Match the exact identity-linking keys;
    # applying name-style normalization here silently merges distinct values.
    identity = value if label in _EXACT_IDENTITY_LABELS else canonical
    if label == "NAME":
        title, identity = _name_identity(value)
        full_identity, role = (
            (name_binding.owner, name_binding.role)
            if name_binding is not None
            else aliases.get(identity, (identity, "full"))
        )
        first, last = _fictional_name(secret, patient, full_identity)
        rendered = title + (
            first if role == "first" else last if role == "last" else first + " " + last
        )
        if name_binding is not None:
            original_tokens = _name_words(identity)

            def safe_nickname(candidate):
                tokens = _name_words(_name_identity(candidate)[1])
                width = len(original_tokens)
                return bool(
                    tokens
                    and width
                    and not any(
                        tokens[i : i + width] == original_tokens
                        for i in range(len(tokens) - width + 1)
                    )
                )

            if not safe_nickname(rendered):
                # Keep the canonical person's key/mapping unchanged. Its family
                # name is a consistent alias only if no original nickname token
                # sequence survives; expanding to the full name would echo it.
                rendered = title + last
            if not safe_nickname(rendered):
                if name_warnings is not None:
                    name_warnings.append(
                        "Synthetic owner name contains the original nickname; NAME redacted."
                    )
                return "[NAME]"
        return rendered
    if label == "AGE":
        return _age(value, "synthetic")
    for salt in range(100):
        digest = _digest(secret, patient, label, identity, salt)
        token = digest.hex()[:12]
        if label == "PHONE":
            replacement = f"(202) 555-01{int.from_bytes(digest[:4], 'big') % 100:02d}"
        elif label == "EMAIL":
            replacement = f"person.{token}@example.invalid"
        elif label == "ADDRESS":
            replacement = f"{100 + int.from_bytes(digest[:4], 'big') % 900} Synthetic Way, Example City, ZZ 00000"
        elif label == "SSN":
            replacement = f"000-{int.from_bytes(digest[:4], 'big') % 100:02d}-{int.from_bytes(digest[4:8], 'big') % 10000:04d}"
        elif label in {"MRN", "ACCOUNT", "ID"}:
            replacement = f"SYN-{label}-{token.upper()}"
        elif label == "URL":
            replacement = f"https://synthetic-{token}.invalid"
        elif label == "IP":
            replacement = f"192.0.2.{1 + int.from_bytes(digest[:4], 'big') % 254}"
        elif label == "LOCATION":
            replacement = f"Example City {token[:6].upper()}"
        elif label == "FACILITY":
            replacement = f"Synthetic {_facility_setting(value)} {token[:6].upper()}"
        else:
            return f"[{_safe_label(label)}]"
        if _canonical(replacement) != canonical:
            return replacement
    return f"[{_safe_label(label)}]"


def replace(
    text: str,
    spans: list[Span],
    *,
    mode: str = "redact",
    patient_id: str | None = None,
    secret: bytes | str | None = None,
    date_shift_days: int | None = None,
) -> dict[str, Any]:
    """Replace supplied PHI spans while copying every other character exactly.

    Entity ``start``/``end`` values index the *input* Unicode string.  Synthetic
    mode requires an application-held key and a stable patient anchor.  The
    patient_id is an exact, case-sensitive opaque anchor, including whitespace.
    Full canonical identities and structured identifiers remain stable across
    requests.  Unambiguous first/last aliases are linked within a note; standalone
    partial names cannot be linked to full identities across different notes.
    Numeric dates use US month/day/year when ambiguous, day/month/year when the
    first component exceeds 12, and year/month/day when year comes first.
    A shifted day/month/year date is emitted year-first if its meaning would
    otherwise become ambiguous. Case-sensitive identifiers use exact keys.
    """
    if mode not in {"redact", "synthetic"}:
        raise ValueError("mode must be 'redact' or 'synthetic'")
    ordered = _validate_spans(text, spans)
    key = b""
    patient = ""
    offset = 0
    if mode == "synthetic":
        key = _validated_secret(secret)
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("Synthetic replacement requires a stable patient_id")
        patient = patient_id
        if date_shift_days is not None:
            if (
                isinstance(date_shift_days, bool)
                or not isinstance(date_shift_days, int)
                or not 1 <= abs(date_shift_days) <= 3650
            ):
                raise ValueError(
                    "date_shift_days must be a nonzero integer between -3650 and 3650"
                )
            offset = date_shift_days
        else:
            shift_index = (
                int.from_bytes(_digest(key, patient, "DATE_SHIFT", "")[:8], "big") % 730
            )
            offset = shift_index - 365 if shift_index < 365 else shift_index - 364

    aliases = _name_aliases(text, ordered) if mode == "synthetic" else {}
    bindings = nickname_bindings(text, ordered) if mode == "synthetic" else {}
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    warnings: list[str] = []
    cursor = 0
    shifted_dates = 0
    redacted_dates = 0
    for entity_index, span in enumerate(ordered):
        original = text[span.start : span.end]
        label = _safe_label(span.label)
        if mode == "redact":
            replacement = _age(original, mode) if label == "AGE" else f"[{label}]"
        elif label == "DATE":
            replacement, warning, shifted = _shift_date(original, offset)
            if shifted:
                shifted_dates += 1
            else:
                redacted_dates += 1
            if warning:
                warnings.append(f"Entity {entity_index} (DATE): {warning}")
        else:
            name_warnings = []
            replacement = _synthetic_value(
                original,
                label,
                key,
                patient,
                aliases,
                name_binding=bindings.get((span.start, span.end)),
                name_warnings=name_warnings,
            )
            warnings.extend(
                f"Entity {entity_index} (NAME): {warning}" for warning in name_warnings
            )
        parts.extend((text[cursor : span.start], replacement))
        entities.append(
            {
                "start": span.start,
                "end": span.end,
                "label": label,
                "replacement": replacement,
                "score": span.score,
                "source": span.source,
            }
        )
        cursor = span.end
    parts.append(text[cursor:])
    return {
        "text": "".join(parts),
        "entities": entities,
        "warnings": warnings,
        "preservation": {
            "clinical_text_outside_spans_preserved": True,
            "date_intervals_preserved": mode == "synthetic" and redacted_dates == 0,
            "exact_dates_shifted": shifted_dates,
            "unshiftable_dates_redacted": redacted_dates,
            "patient_consistent_replacements": mode == "synthetic",
        },
    }
