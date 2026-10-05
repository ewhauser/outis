"""Shared human identity grammar, without a vocabulary of private identities.

An explicit name field, self-identification, signature, or person title can
establish a single-token identity. A public or non-human caption cannot do so.
Pronoun collisions need this same proof rather than a generic human predicate.
"""

from __future__ import annotations

import re
import unicodedata

PUBLIC_PERSON_REFERENCE = re.compile(
    r"\b(?:public(?:ation)?|published|pubmed|pmid|journal|bibliograph\w*|"
    r"textbook|cited\s+(?:author|paper|article)|research\s+paper|"
    r"teaching\s+example|hypothetical|fictional)\b",
    re.I,
)

_NON_HUMAN_CAPTION = re.compile(
    r"\b(?:drug|medication|medicine|product|device|equipment|instrument|"
    r"test|assay|procedure|study|trial|publication|journal|citation|article|"
    r"paper|specimen|sample|catalog(?:ue)?|model)(?:['’]s)?"
    r"(?:[ \t]+(?:brand|generic|registered|official|recorded|supplied|current|full|name|identity))*"
    r"[ \t:=|#>—-]*\Z",
    re.I,
)

_PRONOUN_DEICTIC = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themself",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
}


def _clause_prefix(text: str, start: int) -> str:
    prefix = text[max(0, start - 180) : start]
    boundary = 0
    for match in re.finditer(r"[.;!?|\r\n]", prefix):
        if match.group() == ".":
            begin = match.start()
            while begin and (
                prefix[begin - 1].isalpha()
                or unicodedata.category(prefix[begin - 1]).startswith("M")
            ):
                begin -= 1
            word = prefix[begin : match.start()]
            if word and (
                sum(character.isalpha() for character in word) == 1
                or unicodedata.normalize("NFC", word).casefold()
                in {"dr", "mr", "mrs", "ms", "mx", "prof"}
            ):
                continue
        boundary = match.end()
    return prefix[boundary:]


def human_person_context(text: str, start: int) -> bool:
    """A bounded non-human caption owns its value even if it looks name-like."""
    prefix = _clause_prefix(text, start)
    return not (
        PUBLIC_PERSON_REFERENCE.search(prefix) or _NON_HUMAN_CAPTION.search(prefix)
    )


def pronoun_or_deictic(surface: str) -> bool:
    """Flag a singleton grammar collision, never forbid a literal private name."""
    return unicodedata.normalize("NFC", surface).casefold() in _PRONOUN_DEICTIC
