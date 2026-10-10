"""Note-local NAME corroboration from independently supported person anchors.

Only a supplied NAME detection beside a person-name field, professional title,
signature, or strong human predicate establishes an identity. Exact repetitions
require a human-role predicate. A complete explicit identity may also establish
its unambiguous first/family parts; shared or ambiguous aliases are not guessed.
Unknown portions of existing detections remain covered after a supported split.
Structured PHI spans own their contents and cannot be retagged as names.
"""

from __future__ import annotations

import re
import unicodedata

from outis.replacement.names.identity import (
    PUBLIC_PERSON_REFERENCE,
)

_PUBLIC_PREFIX = PUBLIC_PERSON_REFERENCE

_EPONYM_SUFFIX = re.compile(
    r"(?:['’]s)?\s+(?:disease|syndrome|sign|test|testing|maneuvers?|manoeuvres?|"
    r"criteria|score|response|phenomenon|palsy|thyroiditis|lymphoma|"
    r"granulomatosis|ataxia|point|tenderness)\b",
    re.I,
)

_CREDENTIAL = re.compile(r"(?:MD|DO|RN|NP|PA|PHD|MBBS)\.?\Z", re.I)

_NONVALUES = {
    "unknown",
    "none",
    "unavailable",
    "pending",
    "not available",
    "not recorded",
    "declined",
    "n/a",
}

_FIELD_STOPS = {
    "DOB",
    "MRN",
    "SSN",
    "AGE",
    "PHONE",
    "EMAIL",
    "DATE",
    "ACCOUNT",
    "BILLING",
    "ADDRESS",
}

_PARTICLES = {
    "de",
    "del",
    "da",
    "van",
    "von",
    "der",
    "al",
    "bin",
    "dos",
    "du",
    "la",
    "le",
}

_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*\.?", re.UNICODE)


def _normalized(text):
    """Case/NFC matching with a mapping back to original Unicode offsets."""
    chunks, starts, ends = [], [], []
    position = 0
    while position < len(text):
        stop = position + 1
        while stop < len(text) and unicodedata.category(text[stop]).startswith("M"):
            stop += 1
        piece = unicodedata.normalize("NFC", text[position:stop]).casefold()
        piece = piece.replace("’", "'").replace("‐", "-").replace("‑", "-")
        chunks.append(piece)
        starts.extend([position] * len(piece))
        ends.extend([stop] * len(piece))
        position = stop
    return "".join(chunks), starts, ends


def _pattern(surface):
    normalized = _normalized(surface)[0]
    return re.compile(
        r"[ \t]+".join(re.escape(part) for part in re.split(r"[ \t]+", normalized))
    )


def _word_character(character):
    return (
        character.isalnum()
        or character == "_"
        or unicodedata.category(character).startswith("M")
        or unicodedata.category(character) == "Cf"
    )


def _bounded(text, start, end, *, caption=False):
    if start and _word_character(text[start - 1]):
        return False
    if start > 1 and text[start - 1] in "'-" and _word_character(text[start - 2]):
        return False
    # Names cannot be extracted from an opaque handle, path, or dotted code.
    # A sentence's terminal period remains a boundary when followed by space.
    if start and text[start - 1] in "@/\\?&=%#":
        if not (caption and text[start - 1] in "=#"):
            return False
    if (
        start > 1
        and text[start - 1] == "."
        and (_word_character(text[start - 2]) or text[start - 2] == ".")
    ):
        return False
    if end < len(text) and _word_character(text[end]):
        return False
    if end < len(text) and text[end] in "@/\\?&=%#":
        return False
    if (
        end + 1 < len(text)
        and text[end] == "."
        and (_word_character(text[end + 1]) or text[end + 1] == ".")
    ):
        return False
    if (
        end < len(text)
        and text[end] == "-"
        and end + 1 < len(text)
        and _word_character(text[end + 1])
    ):
        return False
    if (
        end < len(text)
        and text[end] == "'"
        and end + 1 < len(text)
        and _word_character(text[end + 1])
    ):
        # A possessive suffix is grammar outside NAME; an internal apostrophe
        # continues the identity. James' also retains the surname's final s.
        return text[end : end + 2] == "'s" and (
            end + 2 == len(text) or not _word_character(text[end + 2])
        )
    return True


def _trailing_name_word(text, *, separators="", initial_period=False):
    """Read a context word including combining marks omitted by word regexes."""
    stop = len(text)
    while stop and text[stop - 1] in separators:
        stop -= 1
    if initial_period and stop and text[stop - 1] == ".":
        stop -= 1
    begin = stop
    while begin and (
        text[begin - 1].isalpha()
        or unicodedata.category(text[begin - 1]).startswith("M")
        or text[begin - 1] in "'’‐‑-"
    ):
        begin -= 1
    word = text[begin:stop]
    return word if any(character.isalpha() for character in word) else ""


def _clause_prefix(text, start):
    # Context is used only for boolean decisions. Span coordinates continue to
    # come from the original text and the explicit normalization offset map.
    prefix = unicodedata.normalize("NFC", text[max(0, start - 180) : start])
    boundaries = []
    for boundary in re.finditer(r"[.!?\n\r]", prefix):
        if boundary.group() == ".":
            word = _trailing_name_word(prefix[: boundary.start()])
            if word and (
                word.casefold() in {"dr", "mr", "mrs", "ms", "mx", "prof"}
                or sum(character.isalpha() for character in word) == 1
            ):
                continue
        boundaries.append(boundary)
    return prefix[boundaries[-1].end() :] if boundaries else prefix


def _public(text, start, end):
    return bool(
        _PUBLIC_PREFIX.search(_clause_prefix(text, start))
        or _EPONYM_SUFFIX.match(text, end)
    )


def _valid_surface(value):
    if (
        not value
        or value != value.strip()
        or len(value) > 100
        or "\n" in value
        or "\r" in value
    ):
        return False
    if value.casefold() in _NONVALUES or re.match(
        r"(?:Dr\.|Nurse\s|Mr\.|Mrs\.|Ms\.|Prof\.)", value, re.I
    ):
        return False
    if re.search(r",\s*(?:MD|DO|RN|NP|PA|PhD|MBBS)\.?\Z", value, re.I):
        return False
    if value.endswith(("'s", "’s")):
        return False
    return any(character.isalpha() for character in value) and all(
        character.isalpha()
        or unicodedata.category(character).startswith("M")
        or character in " .,'’‐‑-\t"
        for character in value
    )


def _continues_name(text, end, *, comma=True):
    # A neural prefix can stop immediately before an initial's full stop.
    # That punctuation belongs to the initial, including attached NFD marks.
    if text[end : end + 1] == ".":
        preceding = _trailing_name_word(text[:end])
        if preceding and sum(character.isalpha() for character in preceding) == 1:
            end += 1
    gap = re.match(r"(?:,[ \t]+|[ \t]+)" if comma else r"[ \t]+", text[end:])
    if not gap:
        return False
    token = _WORD.match(text, end + gap.end())
    if not token:
        return False
    word = token.group().rstrip(".")
    if (
        token.group().endswith(".")
        and sum(character.isalpha() for character in word) == 1
    ):
        return True
    if _CREDENTIAL.fullmatch(word) or word.upper() in _FIELD_STOPS:
        return False
    return word[:1].isupper() or word.casefold() in _PARTICLES


def _starts_inside_name(text, start):
    """Reject a suffix fragment after another name word or internal separator."""
    if start and _word_character(text[start - 1]):
        return True
    if start > 1 and text[start - 1] in "'’‐‑-" and _word_character(text[start - 2]):
        return True
    prefix = _clause_prefix(text, start)
    # These grammatical cues delimit the whole person span, regardless of
    # caption capitalization. A preceding name fragment cannot satisfy them.
    if re.search(
        r"\b(?:family\s+of|in\s+conversation[,:]|"
        r"during\s+(?:the\s+)?(?:consultation|interview)[,:]|"
        r"for\s+(?:this|the)\s+history[,:]|brought|transported|escorted|"
        r"for\s+assessment\s+by|examined\s+by|reviewed\s+by|"
        r"patient|clinician|doctor|nurse)\s+\Z",
        prefix,
        re.I,
    ):
        return False
    if not prefix or prefix[-1] not in " \t,":
        return False
    word = _trailing_name_word(prefix, separators=" \t,", initial_period=True)
    return (
        word[:1].isupper()
        or word.casefold() in _PARTICLES
        or (bool(word) and unicodedata.category(word[0]).startswith("M"))
    )
