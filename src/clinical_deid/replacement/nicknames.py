"""Occurrence-bound nickname ownership for replacement, never new detections.

The existing deterministic personal-name parser supplies witnesses. A witness
and its mention must both survive as exact NAME spans. Learned owner scores,
entity IDs, and matching strings cannot establish a relationship. The returned
bindings are request-local sensitive data, not API output or persistent memory.
"""

from dataclasses import dataclass, field
import re
from types import MappingProxyType

from clinical_deid.detection.context.nickname import (
    _complete_identity,
    contextual_person_names,
)
from clinical_deid.detection.context.record import (
    _OTHER_RECORD,
    _PATIENT_HEADER,
    _RECORD_HEADER,
    excluded_ranges,
)


@dataclass(frozen=True)
class NameBinding:
    owner: str = field(repr=False)
    role: str
    witness: tuple[int, int]
    relation: str


_RELATIONS = frozenset(
    {
        "explicit-personal-alias-declaration",
        "declared-alias-human-object",
        "explicit-local-nickname-coreference",
    }
)
_RELATION_CUE = re.compile(r"\b(?:nickname|alias)\b", re.I)
_IDENTITY_CLAIM = re.compile(r"\b(?:nickname|alias|identity|naming)\b", re.I)
_CLAIM_REFERENCE = re.compile(
    r"\b(?:this|that|the\s+(?:preceding|previous|above))\s+"
    r"(?:statement|assertion|declaration|claim|entry|relationship|association)\b",
    re.I,
)
_WITHDRAWN_CLAIM = re.compile(
    r"\b(?:retract(?:ed|ion)?|withdrawn|incorrect|mistaken|false|erroneous|denied)\b",
    re.I,
)


def nickname_bindings(text, spans):
    """Return exact-occurrence owner bindings; unresolved detections are kept.

    Explicitly withdrawn/false identity claims conservatively disable nickname
    linking in this caller-authorized context. This does not remove PHI,
    invalidate former names, or assert that quotations are non-PHI. Literal
    legacy aliases and all unproved occurrences retain their prior behavior.
    """
    # Deferred import avoids a module cycle while preserving legacy name keys.
    from clinical_deid.replacement.surrogates import _name_identity

    if not _RELATION_CUE.search(text):
        return MappingProxyType({})
    for clause in re.split(r"[.!?\r\n\u2028\u2029]", text):
        if (
            _IDENTITY_CLAIM.search(clause) or _CLAIM_REFERENCE.search(clause)
        ) and _WITHDRAWN_CLAIM.search(clause):
            return MappingProxyType({})
    retained = {(s.start, s.end): s for s in spans if s.label == "NAME"}
    evidence = []
    contextual_person_names(text, spans, evidence=evidence)
    excluded = excluded_ranges(text)
    supplied_owners = {}
    for bounds, span in retained.items():
        entity_id = getattr(span, "entity_id", None)
        value = text[bounds[0] : bounds[1]]
        if entity_id is not None and _complete_identity(value):
            supplied_owners.setdefault(entity_id, set()).add(_name_identity(value)[1])
    proofs = []
    for proof in evidence:
        bounds = proof["start"], proof["end"]
        witness = proof["witness_start"], proof["witness_end"]
        if (
            proof["relation"] not in _RELATIONS
            or bounds not in retained
            or witness not in retained
            or bounds == witness
        ):
            continue
        owner = _name_identity(text[witness[0] : witness[1]])[1]
        proofs.append((proof, bounds, witness, owner))
        for endpoint in (bounds, witness):
            entity_id = getattr(retained[endpoint], "entity_id", None)
            if entity_id is not None:
                supplied_owners.setdefault(entity_id, set()).add(owner)
    eligible = []
    for proof, bounds, witness, owner in proofs:
        left, right = sorted((bounds, witness))
        if any(a < right[1] and left[0] < b for a, b in excluded):
            continue
        bridge = text[left[1] : right[0]]
        if (
            _OTHER_RECORD.search(bridge)
            or _PATIENT_HEADER.search(bridge)
            or _RECORD_HEADER.search(bridge)
        ):
            # Caller sharing permits one lookup context, not a cross-record
            # identity assertion. Reuse the detection scope boundary contract
            # solely as a veto; never run its extraction or seed its additions.
            continue
        mention, owner_span = retained[bounds], retained[witness]
        # Caller-provided IDs are preserved, but contradictory IDs cannot be
        # silently reinterpreted as authority for a new ownership relationship.
        mention_id = getattr(mention, "entity_id", None)
        owner_id = getattr(owner_span, "entity_id", None)
        if mention_id is not None and owner_id is not None and mention_id != owner_id:
            continue
        if any(
            entity_id is not None and supplied_owners[entity_id] != {owner}
            for entity_id in (mention_id, owner_id)
        ):
            continue
        eligible.append((proof, bounds, witness, owner))
    choices = {}
    for proof, bounds, witness, owner in eligible:
        same_owner_bounds = {
            endpoint
            for _, mention, seed, identity in eligible
            if identity == owner
            for endpoint in (mention, seed)
        }
        left, right = sorted((bounds, witness))
        if any(
            s.label == "NAME"
            and left[1] <= s.start < right[0]
            and (s.start, s.end) not in same_owner_bounds
            and _name_identity(text[s.start : s.end])[1] != owner
            for s in spans
        ):
            # Unproved intervening people stop inheritance. Independently
            # proved occurrences of this owner do not become new seed evidence.
            continue
        choices.setdefault(bounds, set()).add(
            NameBinding(owner, "first", witness, proof["relation"])
        )
    return MappingProxyType(
        {
            bounds: min(options, key=lambda p: (p.witness, p.relation))
            for bounds, options in choices.items()
            if len({option.owner for option in options}) == 1
        }
    )
