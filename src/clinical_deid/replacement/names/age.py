"""English cardinal ages with exact original offsets and bounded age context."""

from __future__ import annotations

import re

_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def parse_english_age(value: str) -> int | None:
    """Parse canonical cardinal words from zero through nine hundred ninety-nine."""
    words = re.split(r"[\s\-\u2010\u2011]+", value.strip().casefold())
    if not words or any(not word for word in words):
        return None

    def small(parts):
        if len(parts) == 1:
            return _SMALL.get(parts[0], _TENS.get(parts[0]))
        if len(parts) == 2 and parts[0] in _TENS and 0 < _SMALL.get(parts[1], -1) < 10:
            return _TENS[parts[0]] + _SMALL[parts[1]]
        return None

    if len(words) > 1 and words[1] == "hundred" and 0 < _SMALL.get(words[0], -1) < 10:
        tail = words[2:]
        if tail and tail[0] == "and":
            tail = tail[1:]
            if not tail:
                return None
        remainder = small(tail) if tail else 0
        return None if remainder is None else 100 * _SMALL[words[0]] + remainder
    return small(words)
