"""BIOES tag set and constrained decoding shared by the encoder detector and its trainer.

The decoder turns per-token emissions into character-offset spans. It is part
of the model's output, not a rule layer: there are no regex extractors or
context guards in the detection path.
"""

from __future__ import annotations

import re
import unicodedata

import torch

from outis.types import ENTITY_TYPES, Span

# Apostrophes and hyphens (ASCII, U+2010, and the non-breaking U+2011 that Word
# emits) that join the letters of one name, place, or facility word.
_NAME_JOINERS = "'’-\u2010\u2011"
TAGS = ["O"] + [f"{boundary}-{label}" for label in ENTITY_TYPES for boundary in "BIES"]


def _transition_constraints():
    """Fixed BIOES legality: 0 for an allowed transition, -inf otherwise."""
    c = len(TAGS)
    legal = torch.full((c, c), float("-inf"))
    for prev in range(c):
        for nxt in range(c):
            p, q = TAGS[prev], TAGS[nxt]
            if p == "O" or p[0] in "ES":
                valid = q == "O" or q[0] in "BS"
            else:
                valid = q != "O" and q[0] in "IE" and p[2:] == q[2:]
            if valid:
                legal[prev, nxt] = 0
    initial = torch.tensor(
        [0 if t == "O" or t[0] in "BS" else float("-inf") for t in TAGS]
    )
    final = torch.tensor(
        [0 if t == "O" or t[0] in "ES" else float("-inf") for t in TAGS]
    )
    return legal, initial, final


# Built once per process. The decoder only reads these; never mutate them.
_LEGAL, _INITIAL, _FINAL = _transition_constraints()


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_bioes(logits, offsets, text):
    """Constrained Viterbi with valid BIOES transitions and exact character offsets."""
    scores = logits.detach().float().cpu()
    n, c = scores.shape
    if n == 0:
        return []
    if not torch.isfinite(scores).all():
        raise RuntimeError("Invalid model logits; no deidentified output released")
    if c != len(TAGS):
        raise RuntimeError(f"Expected {len(TAGS)} tag logits, got {c}")
    labels = _viterbi(scores)
    spans = []
    start = None
    for i, label in enumerate(labels):
        tag = TAGS[label]
        if tag == "O":
            continue
        if tag[0] in "BS":
            start = i
        if tag[0] in "ES" and start is not None:
            entity = tag[2:]
            a, b = _trim_edges(text, offsets[start][0], offsets[i][1], entity)
            if entity in {"NAME", "FACILITY", "LOCATION"}:
                # Word completion stops at the edge of an adjacent decoded entity.
                left_limit = (
                    offsets[start - 1][1] if start > 0 and labels[start - 1] != 0 else 0
                )
                right_limit = (
                    offsets[i + 1][0] if i + 1 < n and labels[i + 1] != 0 else len(text)
                )
                a, b = _refine_word_entity(text, a, b, entity, left_limit, right_limit)
            if b > a:
                confidence = _span_confidence(scores, labels, start, i + 1)
                spans.append(Span(a, b, entity, confidence, "encoder"))
            start = None
    return _merge_overlaps(spans)


def _viterbi(scores):
    """Return the best legal BIOES tag index for each token."""
    dp = scores[0] + _INITIAL
    back = []
    for i in range(1, len(scores)):
        best, bp = (dp[:, None] + _LEGAL).max(0)
        dp = best + scores[i]
        back.append(bp)
    last = int((dp + _FINAL).argmax())
    labels = [last]
    for bp in reversed(back):
        last = int(bp[last])
        labels.append(last)
    labels.reverse()
    return labels


def _span_confidence(scores, labels, start, stop):
    """Mean softmax probability of the decoded tags for tokens start:stop."""
    return (
        scores[start:stop]
        .softmax(-1)[torch.arange(stop - start), torch.tensor(labels[start:stop])]
        .mean()
        .item()
    )


def _merge_overlaps(spans):
    """Merge overlapping spans, keeping the label of the higher-scoring one."""
    # A UTF-8 character can occupy several tokenizer tokens. Projecting legal
    # token spans to characters can therefore create duplicates/overlaps.
    merged = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if merged and span.start < merged[-1].end:
            prior = merged.pop()
            preferred = span if span.score > prior.score else prior
            merged.append(
                Span(
                    prior.start,
                    max(prior.end, span.end),
                    preferred.label,
                    max(prior.score, span.score),
                    "encoder",
                )
            )
        else:
            merged.append(span)
    return merged


def _trim_edges(text, a, b, entity):
    """Drop whitespace and tokenizer-merged separators from a span's edges."""
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    # Tokenizers can attach terminal punctuation to a PHI token.
    while b > a and text[b - 1] in ",;:":
        b -= 1
    # BPE can merge an email's first letter with a preceding contact
    # separator (e.g. ';m'). BIOES cannot express a subtoken boundary.
    # These characters cannot start an unquoted email local part.
    if entity == "EMAIL":
        while a < b and text[a] in ";,:[(<":
            a += 1
    return a, b


# Name-boundary refinement. NAME, FACILITY and LOCATION spans are widened to
# whole words; NAME spans then shed possessive grammar and paired quotes.


def _letter(char):
    return unicodedata.category(char)[0] in {"L", "M"}


def _refine_word_entity(text, a, b, entity, left_limit, right_limit):
    a, b = _complete_words(text, a, b, left_limit, right_limit)
    if entity == "NAME":
        b = _strip_possessive(text, a, b)
        a, b = _strip_quotes(text, a, b)
    return a, b


def _complete_words(text, a, b, left_limit, right_limit):
    """Widen a:b to whole words within left_limit:right_limit.

    NER must not disclose the remaining letters of a partially predicted word.
    Include Unicode combining marks and internal name apostrophes/hyphens,
    without crossing whitespace.
    """
    while a > left_limit and a < b:
        if (
            _letter(text[a - 1])
            and _letter(text[a])
            or (
                text[a] in _NAME_JOINERS
                and a + 1 < b
                and _letter(text[a - 1])
                and _letter(text[a + 1])
            )
        ):
            a -= 1
        elif (
            text[a - 1] in _NAME_JOINERS
            and a - 2 >= left_limit
            and _letter(text[a - 2])
            and _letter(text[a])
        ):
            a -= 2
        else:
            break
    while 0 < b < right_limit:
        if (
            _letter(text[b - 1])
            and _letter(text[b])
            or (
                text[b - 1] in _NAME_JOINERS
                and b - 2 >= a
                and _letter(text[b - 2])
                and _letter(text[b])
            )
        ):
            b += 1
        elif (
            text[b] in _NAME_JOINERS
            and b + 1 < right_limit
            and _letter(text[b - 1])
            and _letter(text[b + 1])
        ):
            b += 2
        else:
            break
    return a, b


def _period_ends_name(value):
    """Whether a trailing period belongs to the name (Jr., Sr., or an initial)."""
    if re.search(r"\b(?:Jr|Sr)\.$", value, re.I):
        return True
    # A terminal personal initial can include Unicode combining marks. A
    # period after a longer word is not enough evidence to remove following
    # grammar.
    if not value.endswith("."):
        return False
    index = len(value) - 2
    while index >= 0 and unicodedata.category(value[index])[0] == "M":
        index -= 1
    return (
        index >= 0
        and unicodedata.category(value[index])[0] == "L"
        and (index == 0 or value[index - 1].isspace() or value[index - 1] == ".")
    )


def _strip_possessive(text, a, b):
    """Return the end of a NAME span without English possessive grammar.

    An English possessive is grammar outside the person's name. Normalize after
    word completion: BPE may attach the suffix, or a partial prediction may stop
    inside it. A true lexical ending is required so O'Sullivan and
    apostrophe/hyphen compounds remain complete, including when another decoded
    entity prevents right expansion.
    """
    possessive_end = b
    while possessive_end > a and text[possessive_end - 1] in ".!?":
        possessive_end -= 1
    terminal = possessive_end == len(text) or not (
        text[possessive_end].isalnum()
        or _letter(text[possessive_end])
        or text[possessive_end] == "_"
        or (
            text[possessive_end] in _NAME_JOINERS
            and possessive_end + 1 < len(text)
            and _letter(text[possessive_end + 1])
        )
    )
    suffix_before_possessive = _period_ends_name(text[a : possessive_end - 2])
    if (
        terminal
        and possessive_end - a > 2
        and text[possessive_end - 2] in "'’"
        and text[possessive_end - 1] in "sS"
        and (_letter(text[possessive_end - 3]) or suffix_before_possessive)
    ):
        return possessive_end - 2
    if (
        b > a
        and text[b - 1] in "'’"
        and _period_ends_name(text[a : b - 1])
        and text[b : b + 1] in {"s", "S"}
        and (
            b + 1 == len(text)
            or not (
                _letter(text[b + 1])
                or text[b + 1].isalnum()
                or text[b + 1] == "_"
                or (
                    text[b + 1] in "'’‐‑-"
                    and b + 2 < len(text)
                    and _letter(text[b + 2])
                )
            )
        )
    ):
        # BPE can end immediately after the apostrophe in "Jr.'s" or "M.'s".
        # Keep the period, excluding possessive grammar only when the complete
        # following s is proven.
        return b - 1
    if (
        terminal
        and possessive_end - a > 1
        and text[possessive_end - 1] in "'’"
        and text[possessive_end - 2] in "sS"
    ):
        # Plural possessives such as James' retain the final s.
        return possessive_end - 1
    return b


def _strip_quotes(text, a, b):
    """Exclude a quotation mark pair from the edges of a NAME span.

    An edge apostrophe can be a quotation delimiter or part of a name. Require a
    paired quote in this paragraph and preserve ambiguous leading elisions such
    as 't Hooft.
    """
    opening = a if text[a : a + 1] in {"'", "‘"} else a - 1
    if not (
        opening >= 0
        and text[opening] in "'‘"
        and (
            opening == 0
            or not (
                _letter(text[opening - 1])
                or text[opening - 1].isalnum()
                or text[opening - 1] == "_"
            )
        )
    ):
        return a, b
    close_char = "’" if text[opening] == "‘" else "'"
    first = opening + 1
    after_initial = first + 1
    while (
        after_initial < len(text)
        and unicodedata.category(text[after_initial])[0] == "M"
    ):
        after_initial += 1
    leading_elision = (
        first < len(text)
        and unicodedata.category(text[first])[0] == "L"
        and after_initial < len(text)
        and (text[after_initial].isspace() or text[after_initial] in "-‐‑")
    )
    if leading_elision:
        return a, b
    for closing in range(first, len(text)):
        char = text[closing]
        if char in "\r\n\u0085  ":
            break
        if char not in {close_char, text[opening]}:
            continue
        before = text[closing - 1] if closing > first else " "
        after = text[closing + 1 : closing + 2]
        after_is_word = bool(after) and (
            _letter(after)
            or after.isalnum()
            or after == "_"
            or (
                after in "-‐‑"
                and closing + 2 < len(text)
                and _letter(text[closing + 2])
            )
        )
        if char == close_char and not before.isspace() and not after_is_word:
            # A distant apostrophe after a word could instead be a plural
            # possessive. Only a punctuated utterance establishes a pair beyond
            # the name's own closing edge.
            if closing in {b - 1, b} or (closing > b and before in ".,!?"):
                if a == opening:
                    a += 1
                if b - 1 == closing:
                    b -= 1
            break
        if (
            char == text[opening]
            and (before.isspace() or before in "([{=:;")
            and after_is_word
        ):
            # A new opener means the original mark was not demonstrably
            # paired; keep it intact.
            break
    return a, b
