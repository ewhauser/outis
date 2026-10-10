"""Recover personal names from explicit note-local identity relations.

No nickname dictionary or unrestricted lowercase NER is used. Supplied complete
NAME detections establish possible identities; an independent mention must
corroborate the owner of every recovered occurrence. Newly inferred names do not
seed this call or propagate elsewhere in the note.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left

from outis.replacement.names.aliases import full_identity_aliases
from outis.replacement.names.identity import (
    human_person_context,
    pronoun_or_deictic,
)
from outis.replacement.names.person import (
    _bounded,
    _continues_name,
    _normalized,
    _pattern,
    _public,
    _starts_inside_name,
    _valid_surface,
)
from outis.types import Span

_WRITE_NAME = re.compile(
    r"\b(?:the[ \t]+)?(?:patient|person|individual|caregiver|guardian|clinician|doctor|nurse)"
    r"[ \t]+(?:writes?|wrote|spells?|spelled|renders?|rendered|records?|recorded)"
    r"[ \t]+(?:(?:his|her|their|the)[ \t]+)?(?:full[ \t]+|personal[ \t]+)?name"
    r"[ \t]+as[ \t]+",
    re.I,
)

_WRITE_TRAILER = re.compile(
    r"(?:[ \t]*(?:[.;!?\r\n]|\Z)|[ \t]*(?:,[ \t]*)?(?:in|on)[ \t]+"
    r"(?:(?:a|an|the|their|his|her)[ \t]+)?"
    r"(?:portal[ \t]+messages?|messages?|correspondence|forms?|records?|charts?|"
    r"letters?|documents?|paperwork|registration|signature[ \t]+(?:line|block))\b)",
    re.I,
)

_NICKNAME_RELATION = re.compile(
    r"[ \t]+(?:uses?|used|prefers?|preferred|responds?[ \t]+to|responded[ \t]+to|"
    r"(?:is|was)[ \t]+known[ \t]+by)[ \t]+"
    r"(?:that|this|the[ \t]+same)[ \t]+(?:(?:lowercase|upper-case|lower-case|uppercase)[ \t]+)?"
    r"(?P<personal>personal[ \t]+)?(?P<kind>nickname|alias)\b",
    re.I,
)

_ROLE_OWNER_PREFIX = re.compile(
    r"\b(?:the[ \t]+)?(?:patient|person|individual|caregiver|guardian|clinician|doctor|nurse)[ \t]+\Z",
    re.I,
)

_HUMAN_PREDICATE = re.compile(
    r"[ \t]+(?:asked|requests?|requested|reports?|reported|denies|denied|"
    r"answers?|answered|confirms?|confirmed|describes?|described|"
    r"attends?|attended|prefers?|preferred|wants?|wanted)[ \t]+[^.;!?\r\n]{1,140}\Z",
    re.I,
)

_UNSAFE_RELATION = re.compile(
    r"\b(?:not|never|neither|without|hypothetical|fictional|quoted|quotation|example|"
    r"public|published|teaching|device|medication|drug|test|study|trial|specimen|model)\b",
    re.I,
)

_AMBIGUOUS_CLAUSE = re.compile(r"\b(?:and|or|but|while|whereas)\b|[;,:\"“”]", re.I)

_COMPATIBLE = {"NAME", "FACILITY", "LOCATION"}

_QUOTES = {"'": "'", '"': '"', "‘": "’", "“": "”"}

_INDEFINITE_SUBJECTS = {
    "someone",
    "somebody",
    "anyone",
    "anybody",
    "everyone",
    "everybody",
    "nobody",
    "noone",
    "no-one",
    "something",
    "anything",
    "everything",
    "nothing",
    "one",
    "others",
    "each",
    "either",
    "neither",
    "both",
}

_OTHER_OWNER = re.compile(
    r"\b(?:(?:for|of)(?![ \t]+(?:himself|herself|themself|themselves)\b)|"
    r"belonging[ \t]+to|referring[ \t]+to|to[ \t]+refer[ \t]+to)\b",
    re.I,
)

_PERSONAL_ALIAS_DECLARATION = re.compile(
    # Start once per complete token; retrying inside a long opaque token would
    # make a failed declaration scan quadratic before geometric guards run.
    r"(?<![^\s;:,.!?\"“”()\[\]{}])"
    r"(?P<alias>[^\s;:,.!?\"“”()\[\]{}]+)[ \t]+(?:is|was)[ \t]+"
    r"(?:the[ \t]+)?(?:(?:preferred|chosen|personal|short)[ \t]+){1,3}"
    r"(?:name|nickname|alias)[ \t]+(?:of|used[ \t]+by|belonging[ \t]+to)[ \t]+",
    re.I,
)

_HUMAN_OBJECT_PREFIX = re.compile(
    r"(?:\A|[.;!?\r\n])[ \t]*(?:the[ \t]+)?"
    r"(?:nurse|doctor|clinician|physician|caregiver|guardian|therapist|patient)"
    r"[ \t]+(?:asked|invited|encouraged|reminded|helped)[ \t]+\Z",
    re.I,
)

_NONLITERAL_ALIAS_SCOPE = re.compile(
    r"\b(?:public|published|teaching|hypothetical|fictional|example|quotation|"
    r"quoted|sentence|script|device|medication|drug|specimen|model|test|study|trial)\b",
    re.I,
)

_HUMAN_INFINITIVE = re.compile(
    r"[ \t]+to[ \t]+(?:"
    r"(?:choose|select)[ \t]+(?:(?:the|a|an|their)[ \t]+)?"
    r"(?:(?:first|next|preferred|appropriate)[ \t]+)?"
    r"(?:activity|option|exercise|meal|appointment|time|position)|"
    r"(?:describe|explain|demonstrate)[ \t]+(?:(?:the|a|an|their)[ \t]+)?"
    r"(?:(?:care|treatment|breathing|walking|planned)[ \t]+)?"
    r"(?:exercise|technique|plan|symptoms|concerns|procedure|movement)|"
    r"(?:answer)[ \t]+(?:the[ \t]+)?(?:question|questions)|"
    r"(?:walk|sit|stand|breathe|return|attend|consent|decide|begin|practice|practise)"
    r"(?:[ \t]+(?:slowly|carefully|comfortably|safely|independently|"
    r"with[ \t]+(?:assistance|support)))?)"
    r"[ \t]*(?=[.;!?]|\Z)",
    re.I,
)

_PERSONAL_ALIAS_HEADING = re.compile(
    r"[ \t]*(?:(?:(?:care|clinical|patient|visit|discharge|follow[- ]?up)[ \t]+)?"
    r"(?:coordination|arrangements|planning|registration|preferences)|"
    r"(?:(?:patient|personal|confirmed|preferred)[ \t]+)?(?:identity|name))"
    r"[ \t]*[:—][ \t]*\Z",
    re.I,
)

_SOURCE_ALIAS_PREFACE = re.compile(
    r"\b(?:public|published|teaching|hypothetical|fictional|quoted|invented)"
    r"(?:[ \t]+[^\W\d_]+){0,3}[ \t]+"
    r"(?:example|account|narrative|script|sentence|passage|biography)\b",
    re.I,
)


def _alias_source_context(text, start):
    prefix = text[:start]
    # An enclosing quoted or source-labelled block remains untrusted as a
    # naming assertion. This conservative recovery abstains after such a block.
    if prefix.count('"') % 2 or prefix.count("“") > prefix.count("”"):
        return False
    if re.search(r"(?:\A|[\s:=—])['‘](?=\s|[^\W\d_])", prefix):
        return False
    if _SOURCE_ALIAS_PREFACE.search(prefix):
        return False
    for heading in re.finditer(
        r"(?:\A|[.!?\r\n])([^\r\n.!?]+)[:—][ \t]*(?=[\r\n]|\Z)", prefix
    ):
        if not _PERSONAL_ALIAS_HEADING.fullmatch(heading.group().lstrip(".!?\r\n")):
            return False
    return True


def _literal_alias_context(text, start):
    if not _alias_source_context(text, start):
        return False
    prefix = text[:start]
    boundary = max((prefix.rfind(character) for character in ".;!?\r\n"), default=-1)
    local = prefix[boundary + 1 :]
    if not local.strip():
        return True
    # A closed short heading may precede an assertion. Arbitrary introductory
    # prose ('denies that', 'it is untrue that') is not a naming declaration.
    if not _PERSONAL_ALIAS_HEADING.fullmatch(local):
        return False
    return not _NONLITERAL_ALIAS_SCOPE.search(local)


def _declared_personal_aliases(
    text, candidates, identities, normalized, starts, ends, add
):
    """Read a literal alias relationship and separately prove each later use.

    The complete detected owner's spelling corroborates the relationship. A
    declared alias alone never authorizes replacement of all matching words.
    """
    declarations = []
    owners = {}
    for cue in _PERSONAL_ALIAS_DECLARATION.finditer(text):
        start, end = cue.span("alias")
        value = cue["alias"]
        if (
            not _alias_surface(value)
            or not _bounded(text, start, end)
            or _starts_inside_name(text, start)
            or not human_person_context(text, start)
            or _public(text, start, end)
            or not _literal_alias_context(text, start)
            or _owned(start, end, candidates)
        ):
            continue
        # Negated/quoted/example relations cannot establish a literal identity.
        prefix = re.split(r"[.;!?\r\n]", text[max(0, start - 180) : start])[-1]
        if _UNSAFE_RELATION.search(prefix) or any(c in prefix for c in '"“”'):
            continue
        options = []
        for key, identity in identities.items():
            bounds = _original_match(
                identity["pattern"], normalized, starts, ends, cue.end()
            )
            if bounds is None or not re.match(
                r"[ \t]*(?:[.;!?]|\Z)", text[bounds[1] :]
            ):
                continue
            witness = next(
                (s for s in identity["seeds"] if (s.start, s.end) == bounds), None
            )
            if witness is not None:
                options.append((key, witness))
        if len(options) != 1:
            continue
        key, witness = options[0]
        alias = _normalized(value)[0]
        owners.setdefault(alias, set()).add(key)
        declarations.append((start, end, alias, key, witness))
    for start, end, alias, key, witness in declarations:
        add((start, end), key, witness, "explicit-personal-alias-declaration")
        if len(owners[alias]) != 1 or _grammar_subject(text[start:end]):
            continue
        for match in _pattern(text[start:end]).finditer(normalized):
            left, right = starts[match.start()], ends[match.end() - 1]
            if (
                left <= end
                or right - end > 2000
                or _original_match(
                    _pattern(text[start:end]), normalized, starts, ends, left
                )
                != (left, right)
                or not human_person_context(text, left)
                or _public(text, left, right)
                or not _alias_source_context(text, left)
                or not _HUMAN_OBJECT_PREFIX.search(text[max(0, left - 180) : left])
                or not _HUMAN_INFINITIVE.match(text, right)
            ):
                continue
            add((left, right), key, witness, "declared-alias-human-object")


def _grammar_subject(value: str) -> bool:
    return pronoun_or_deictic(value) or _normalized(value)[0] in _INDEFINITE_SUBJECTS


def _unsafe_relation_tail(text: str, end: int) -> bool:
    trailer = re.split(r"[.;!?\r\n]", text[end : end + 160], maxsplit=1)[0]
    return bool(_UNSAFE_RELATION.search(trailer) or _OTHER_OWNER.search(trailer))


def _alias_surface(value: str) -> bool:
    def letter(character):
        return character.isalpha() or unicodedata.category(character).startswith("M")

    return bool(
        value
        and value[0].isalpha()
        and letter(value[-1])
        and all(
            letter(character)
            or (
                character in "'’‐‑-"
                and index
                and index + 1 < len(value)
                and letter(value[index - 1])
                and value[index + 1].isalpha()
            )
            for index, character in enumerate(value)
        )
    )


def _complete_identity(value: str) -> bool:
    # Matching the complete detected identity avoids guessing which words are
    # first/family components in multipart or multicultural names.
    words = re.split(r"[ \t,]+", value)
    return (
        _valid_surface(value)
        and 2 <= len(words) <= 7
        and any(sum(c.isalpha() for c in word) > 1 for word in words)
    )


def _owned(start: int, end: int, candidates: list[Span]) -> bool:
    return any(
        s.label not in _COMPATIBLE and s.start < end and start < s.end
        for s in candidates
    )


def _identity_seeds(text: str, candidates: list[Span]):
    result = {}
    normalized, starts, ends = _normalized(text)
    for span in candidates:
        if span.label != "NAME" or not 0 <= span.start < span.end <= len(text):
            continue
        value = text[span.start : span.end]
        if (
            not _complete_identity(value)
            or _owned(span.start, span.end, candidates)
            or not human_person_context(text, span.start)
            or _public(text, span.start, span.end)
            or _continues_name(text, span.end)
            or _starts_inside_name(text, span.start)
            or _original_match(_pattern(value), normalized, starts, ends, span.start)
            != (span.start, span.end)
        ):
            continue
        key = _normalized(value)[0]
        result.setdefault(key, {"pattern": _pattern(value), "seeds": []})[
            "seeds"
        ].append(span)
    return result


def _original_match(pattern, normalized, starts, ends, start):
    position = bisect_left(starts, start)
    if position >= len(starts) or starts[position] != start:
        return None
    match = pattern.match(normalized, position)
    if not match or not _bounded(normalized, match.start(), match.end()):
        return None
    # A normalization expansion must be consumed as a complete source grapheme.
    if (match.start() and starts[match.start() - 1] == starts[match.start()]) or (
        match.end() < len(starts) and starts[match.end()] == starts[match.end() - 1]
    ):
        return None
    return starts[match.start()], ends[match.end() - 1]


def _nickname_antecedent(text: str, owner_start: int):
    prefix = text[max(0, owner_start - 190) : owner_start]
    role = _ROLE_OWNER_PREFIX.search(prefix)
    if role is not None:
        prefix = prefix[: role.start()]
        owner_start -= len(role.group())
    separator = re.search(r"[;.][ \t]*\Z", prefix)
    if separator is None:
        return None
    clause_stop = owner_start - len(prefix) + separator.start()
    # The pair must begin at a sentence/line boundary. A second antecedent,
    # speaker turn, quotation, or paragraph crossing is not resolved by guess.
    before = text[max(0, clause_stop - 170) : clause_stop]
    boundary = max((m.end() for m in re.finditer(r"[.!?\r\n]", before)), default=0)
    if not boundary and clause_stop > len(before):
        return None
    clause_start = clause_stop - len(before) + boundary
    clause = text[clause_start:clause_stop]
    left = len(clause) - len(clause.lstrip(" \t"))
    clause_start += left
    clause = clause[left:]
    if (
        not clause
        or _AMBIGUOUS_CLAUSE.search(clause)
        or _UNSAFE_RELATION.search(clause)
    ):
        return None
    subject = re.match(r"\S+", clause)
    if subject is None:
        return None
    value = subject.group()
    if (
        not _alias_surface(value)
        or any(c in '.,"“”' for c in value)
        or not _HUMAN_PREDICATE.fullmatch(clause[subject.end() :])
    ):
        return None
    start, end = clause_start, clause_start + subject.end()
    if (
        not _bounded(text, start, end)
        or not human_person_context(text, start)
        or _public(text, start, end)
    ):
        return None
    return start, end


def contextual_person_names(
    text: str, candidates: list[Span], *, evidence: list | None = None
) -> list[Span]:
    """Return locally proved spans without modifying or recursively seeding input."""
    normalized, starts, ends = _normalized(text)
    identities = _identity_seeds(text, candidates)
    found = {}

    def add(bounds, owner, witness, relation):
        start, end = bounds
        if (start, end) in found or _owned(start, end, candidates):
            return
        found[(start, end)] = Span(start, end, "NAME", 1.0, "rules:field")
        if evidence is not None:
            evidence.append(
                {
                    "start": start,
                    "end": end,
                    "owner": owner,
                    "witness_start": witness.start,
                    "witness_end": witness.end,
                    "relation": relation,
                }
            )

    _declared_personal_aliases(
        text, candidates, identities, normalized, starts, ends, add
    )

    for cue in _WRITE_NAME.finditer(text):
        if not human_person_context(text, cue.start()):
            continue
        options = []
        for key, identity in identities.items():
            opening = text[cue.end() : cue.end() + 1]
            quoted = opening in _QUOTES
            bounds = _original_match(
                identity["pattern"], normalized, starts, ends, cue.end() + int(quoted)
            )
            if bounds is None:
                continue
            trailer_start = bounds[1]
            if quoted:
                if text[trailer_start : trailer_start + 1] != _QUOTES[opening]:
                    continue
                trailer_start += 1
            if not _WRITE_TRAILER.match(text, trailer_start):
                continue
            if _unsafe_relation_tail(text, trailer_start):
                continue
            witnesses = [
                s
                for s in identity["seeds"]
                if s.end <= cue.start() or s.start >= bounds[1]
            ]
            if witnesses:
                options.append((bounds, key, witnesses[0]))
        if len(options) == 1:
            bounds, key, witness = options[0]
            add(bounds, key, witness, "explicit-personal-name-spelling")

    for key, identity in identities.items():
        for match in identity["pattern"].finditer(normalized):
            owner_start, owner_end = starts[match.start()], ends[match.end() - 1]
            if _original_match(
                identity["pattern"], normalized, starts, ends, owner_start
            ) != (owner_start, owner_end):
                continue
            relation = _NICKNAME_RELATION.match(text, owner_end)
            if relation is None or not human_person_context(text, owner_start):
                continue
            # Ambiguous/public qualifications in the same relation clause stop
            # recovery even when a complete NAME was predicted elsewhere.
            trailer = re.split(
                r"[.;!?\r\n]", text[relation.end() : relation.end() + 120], maxsplit=1
            )[0]
            if _unsafe_relation_tail(text, relation.end()) or _AMBIGUOUS_CLAUSE.search(
                trailer
            ):
                continue
            witness = next(
                (
                    s
                    for s in identity["seeds"]
                    if s.end <= owner_start or s.start >= relation.end()
                ),
                None,
            )
            bounds = _nickname_antecedent(text, owner_start)
            literal_alias = bool(
                relation["personal"]
                and _ROLE_OWNER_PREFIX.search(
                    text[max(0, owner_start - 60) : owner_start]
                )
            )
            if literal_alias and bounds is not None:
                alias = _normalized(text[bounds[0] : bounds[1]])[0]
                parts = {
                    _normalized(part)[0]
                    for part in full_identity_aliases(text[owner_start:owner_end])
                }
                if alias in parts:
                    witness = next(
                        (
                            s
                            for s in identity["seeds"]
                            if (s.start, s.end) == (owner_start, owner_end)
                        ),
                        witness,
                    )
                else:
                    literal_alias = False
            if (
                bounds is not None
                and _grammar_subject(text[bounds[0] : bounds[1]])
                and not literal_alias
            ):
                continue
            if witness is not None and bounds is not None:
                add(bounds, key, witness, "explicit-local-nickname-coreference")
    return [found[bounds] for bounds in sorted(found)]
