"""Measured record-context additions from retained clinical identity spans.

Caller supplies final retained spans after identity and boundary refinement.
This returns additions only and never removes or relabels baseline coverage.
Public/quoted/unknown scope causes abstention, not a negative PHI assertion.
Clinical-handover temporal references are an explicitly uncertain heuristic.
"""

from __future__ import annotations

import re

_RECORD_HEADER = re.compile(
    r"(?im)^[ \t]*(?:(?:current|actual|new|next|other|another|different|separate|second|third)[ \t]+)?patient[ \t]+(?:record|chart|encounter)[ \t]*:\s*(?=$|\n)"
)

_PATIENT_HEADER = re.compile(
    r"(?im)^[ \t]*(?:patient(?:['’]s)?(?:[ \t]+(?:full[ \t]+)?(?:name|identity))?|person(?:['’]s)?[ \t]+name)[ \t]*[:=|]"
)

_NEUTRAL_HEADING = re.compile(
    r"(?i)(?:(?:clinical|medical|patient|care|current|actual)[ \t]+)*(?:assessment|plan|update|history|handover|record|chart|encounter|review|follow[- ]?up|medication[ \t]+teaching):"
)

_BLOCK_HEADING = re.compile(
    r"(?i)\b(?:teaching[ \t]+(?:example|chart|case|material)|public[ \t]+(?:source|teaching|record|chart|material)|source[ \t]+(?:excerpt|material|quotation)|quoted[ \t]+(?:source|record)|references|bibliography|scenario|simulation|fictional|hypothetical)\b"
)

_FICTIONAL_DIRECTIVE = re.compile(
    r"(?i)^[ \t]*(?:imagine|invent|pretend|suppose|consider[ \t]+(?:a|this)[ \t]+(?:fictional|hypothetical))\b"
)

_END_BLOCK = re.compile(
    r"(?i)^[ \t]*(?:end[ \t]+(?:of[ \t]+)?(?:the[ \t]+)?(?:quotation|quote|excerpt|source|teaching[ \t]+example|scenario|simulation)|\[/(?:quote|source|example)\])[ .:]*$"
)


def excluded_ranges(text, *, include_quotes=True):
    """Conservative structural/source exclusions, without an expiration window."""
    excluded = []
    blocked = None
    fence = None
    for line in re.finditer(
        r"[^\r\n\u0085\u2028\u2029]*(?:\r\n|[\r\n\u0085\u2028\u2029]|$)", text
    ):
        if not line.group():
            continue
        value = line.group().strip()
        if value.startswith(("```", "~~~")):
            if fence is None:
                fence = line.start()
            else:
                excluded.append((fence, line.end()))
                fence = None
        if value.startswith(">"):
            excluded.append((line.start(), line.end()))
        if blocked is not None and _END_BLOCK.fullmatch(value):
            excluded.append((blocked, line.end()))
            blocked = None
            continue
        heading = value.endswith(":")
        unsafe = _FICTIONAL_DIRECTIVE.search(value) or (
            heading and _BLOCK_HEADING.search(value)
        )
        unknown = heading and not (
            _NEUTRAL_HEADING.fullmatch(value) or _RECORD_HEADER.fullmatch(value + "\n")
        )
        if blocked is None and (unsafe or unknown):
            blocked = line.start()
    if blocked is not None:
        excluded.append((blocked, len(text)))
    if fence is not None:
        excluded.append((fence, len(text)))
    if not include_quotes:
        return sorted(excluded)
    # Quoted content stays unresolved, including private speech; preserve native
    # detections but do not use it to seed or receive this extra propagation.
    pairs = {'"': '"', "“": "”", "«": "»", "'": "'", "‘": "’"}
    position = 0
    while position < len(text):
        character = text[position]
        if character not in pairs or (
            character in "'‘" and position and text[position - 1].isalnum()
        ):
            position += 1
            continue
        stop = text.find(pairs[character], position + 1)
        while (
            stop >= 0
            and pairs[character] in "'’"
            and stop
            and stop + 1 < len(text)
            and text[stop - 1].isalnum()
            and text[stop + 1].isalnum()
        ):
            stop = text.find(pairs[character], stop + 1)
        stop = len(text) if stop < 0 else stop + 1
        excluded.append((position, stop))
        position = stop
    return sorted(excluded)


_OTHER_RECORD = re.compile(
    r"(?i)\b(?:other|another|different|separate|second|third|next|new)[ \t]+(?:patient|person|record|chart|case)\b"
)
