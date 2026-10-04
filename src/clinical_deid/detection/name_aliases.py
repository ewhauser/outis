"""Conservative first/family aliases derived from complete literal identities.

No private-name vocabulary or nickname inference is used. Ambiguous multipart
identities, initials-only values and shared aliases require another proof.
"""

from __future__ import annotations

import re
import unicodedata

_PARTICLES = {"de", "del", "da", "van", "von", "der", "al", "bin", "dos", "du", "la", "le", "den", "des"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_FUNCTION_WORDS = {
    "a", "an", "the", "for", "to", "in", "on", "of", "as", "at", "by", "from",
    "have", "has", "had", "do", "does", "did", "be", "is", "are", "was", "were",
    "will", "may", "can", "no", "not", "but", "and", "or", "na", "if", "so",
}


def alias_function_word(value: str) -> bool:
    """A grammar collision needs an explicit title/naming proof at each mention."""
    return unicodedata.normalize("NFC", value).casefold() in _FUNCTION_WORDS


def _word(value: str) -> bool:
    return bool(value) and value[0].isalpha() and all(
        c.isalpha() or unicodedata.category(c).startswith("M") or c in "'’‐‑-"
        for c in value
    )


def _initial(value: str) -> bool:
    return value.endswith(".") and sum(c.isalpha() for c in value[:-1]) == 1 and all(
        c.isalpha() or unicodedata.category(c).startswith("M") for c in value[:-1]
    )


def _family(words: list[str]) -> str | None:
    if not words or not _word(words[-1]) or words[-1].casefold() in _PARTICLES:
        return None
    if any(word.casefold() not in _PARTICLES for word in words[:-1]):
        return None
    return " ".join(words)


def full_identity_aliases(value: str) -> tuple[str, ...]:
    """Return only directly expressed given/family parts of a complete identity.

    A two-part name, optional middle initials, recognized surname particles,
    generational suffix and surname-first comma order have unambiguous parts.
    Three ordinary words can encode multiple given/family names; no guess is
    made for those forms. Returned spelling retains the input's Unicode form.
    """
    if value != value.strip() or "\n" in value or "\r" in value or value.count(",") > 1:
        return ()
    comma_parts = value.split(",")
    if len(comma_parts) == 2 and comma_parts[1].strip().rstrip(".").casefold() in _SUFFIXES:
        comma_parts = comma_parts[:1]
    parts = [re.split(r"[ \t]+", part.strip()) for part in comma_parts]
    for words in parts:
        if words and words[-1].rstrip(".").casefold() in _SUFFIXES:
            words.pop()
    if len(parts) == 2:
        family_words, given_words = parts
        if not given_words or not _word(given_words[0]) or any(not _initial(word) for word in given_words[1:]):
            return ()
        first, family = given_words[0], _family(family_words)
    else:
        words = parts[0]
        if len(words) < 2 or not _word(words[0]) or words[0].casefold() in _PARTICLES:
            return ()
        first, cursor = words[0], 1
        while cursor < len(words) and _initial(words[cursor]):
            cursor += 1
        family = _family(words[cursor:])
    if family is None or first.casefold() in _PARTICLES:
        return ()
    return tuple(dict.fromkeys((first, family)))
