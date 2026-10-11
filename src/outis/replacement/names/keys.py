"""Name identity normalization for surrogate keys and in-note alias linking."""

from __future__ import annotations

import re
import unicodedata

from outis.replacement.names.aliases import full_identity_aliases
from outis.replacement.primitives import canonical_form
from outis.types import Span

_TITLE = re.compile(r"^(?P<title>(?:dr|mr|mrs|ms|mx|miss|prof)\.?\s+)", re.I)
_NAME_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_NAME_SUFFIX = re.compile(r"(?:jr|sr|ii|iii|iv|v|vi)\.?\Z", re.I)
_NAME_PUNCTUATION = str.maketrans({"’": "'", "‐": "-", "‑": "-"})


def name_identity(value: str) -> tuple[str, str]:
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
    return title, canonical_form(body)


def name_words(value: str) -> list[str]:
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


def name_aliases(text: str, spans: list[Span]) -> dict[str, tuple[str, str]]:
    """Resolve first/last mentions only when they identify one observed full name."""
    aliases: dict[str, set[tuple[str, str]]] = {}
    for span in spans:
        if span.label != "NAME":
            continue
        _, identity = name_identity(text[span.start : span.end])
        parts = full_identity_aliases(identity)
        # Initials, ambiguous multipart names and a shared first/family spelling
        # cannot establish which component a reduced mention represents.
        if len(parts) == 2:
            aliases.setdefault(canonical_form(parts[0]), set()).add((identity, "first"))
            aliases.setdefault(canonical_form(parts[1]), set()).add((identity, "last"))
    return {
        alias: next(iter(matches))
        for alias, matches in aliases.items()
        if len(matches) == 1
    }
