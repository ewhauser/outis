"""Configurable replacement, conservative identity linking, and explicit restoration.

Offsets count Python Unicode characters. Numbered identities are request-local;
keyed identities use a patient-scoped HMAC. Restoration mappings are sensitive
caller-owned values and are never retained by this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace as replace_entity
import hashlib
import hmac
import json
import re
from string import Formatter
from typing import Literal
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clinical_deid.entities import Entity, checked_entities
from clinical_deid.detection.name_aliases import full_identity_aliases
from clinical_deid.replacement.nicknames import NameBinding, nickname_bindings
from clinical_deid.replacement.surrogates import (
    _canonical,
    _digest,
    _name_aliases,
    _name_identity,
    _shift_date,
    _synthetic_value,
    _validated_secret,
)
from clinical_deid.types import ENTITY_TYPES

_MAX_TEXT = 100_000
_MAX_OUTPUT = 1_000_000
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_PUBLIC_CONCEPT_TYPES = frozenset({"CONDITION", "MEDICATION"})
Action = Literal["marker", "mask", "synthetic", "keep"]


def _labels(values):
    if any(
        not isinstance(value, str) or not _LABEL.fullmatch(value) for value in values
    ):
        raise ValueError("Invalid entity label")
    return values


class ReplacementOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["marker", "mask", "synthetic"] = "marker"
    pattern: str = Field(default="[{label}_{id}]", min_length=1, max_length=256)
    mask_character: str = Field(default="*", min_length=1, max_length=1)
    per_type: dict[str, Action] = Field(default_factory=dict, max_length=256)
    include_mapping: bool = False
    entity_ids: Literal["numbered", "keyed"] = "numbered"

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value):
        try:
            for _, field, spec, conversion in Formatter().parse(value):
                if field is not None and (
                    field not in {"label", "id", "index"} or spec or conversion
                ):
                    raise ValueError
            rendered = value.format(label="NAME", id="e1", index=1)
            if not rendered.strip() or any(ord(char) < 32 for char in rendered):
                raise ValueError
        except (ValueError, KeyError, IndexError, AttributeError):
            raise ValueError(
                "Marker pattern permits only label, id, and index fields without formatting"
            ) from None
        return value

    @field_validator("mask_character")
    @classmethod
    def valid_mask(cls, value):
        if value.isspace() or not value.isprintable():
            raise ValueError("mask_character must be one visible character")
        return value

    @field_validator("per_type")
    @classmethod
    def valid_types(cls, value):
        return _labels(value)


class RestorationMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    marker: str = Field(min_length=1, max_length=_MAX_TEXT)
    original: str = Field(min_length=1, max_length=_MAX_TEXT)
    label: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value):
        _labels([value])
        return value


class ReidentifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(max_length=_MAX_OUTPUT)
    mappings: list[RestorationMapping] = Field(max_length=20_000)
    allowed_types: list[str] | None = Field(default=None, max_length=256)
    restore_sensitive: bool = False

    @field_validator("allowed_types")
    @classmethod
    def valid_types(cls, value):
        return _labels(value) if value is not None else None

    @model_validator(mode="after")
    def bounded_mappings(self):
        if sum(len(item.marker) for item in self.mappings) > 200_000:
            raise ValueError("Restoration markers exceed the size limit")
        if (
            sum(len(item.marker) + len(item.original) for item in self.mappings)
            > 2 * _MAX_OUTPUT
        ):
            raise ValueError("Restoration mappings exceed the size limit")
        return self


def _patient_key(secret, patient_id):
    key = _validated_secret(secret)
    if (
        not isinstance(patient_id, str)
        or not patient_id.strip()
        or len(patient_id) > 256
    ):
        raise ValueError(
            "Keyed identities and synthetic replacement require a stable patient_id"
        )
    return key, patient_id


def _identity_keys(text, spans):
    aliases = _name_aliases(text, spans)
    bindings = nickname_bindings(text, spans)
    owners: dict[str, set[str]] = {}
    for span in spans:
        if span.label == "NAME":
            _, identity = _name_identity(text[span.start : span.end])
            parts = full_identity_aliases(identity)
            if len(parts) == 2:
                for part in parts:
                    owners.setdefault(_canonical(part), set()).add(identity)
    keys = []
    for span in spans:
        value = text[span.start : span.end]
        if span.label == "NAME":
            _, value = _name_identity(value)
            binding = bindings.get((span.start, span.end))
            if binding is not None:
                value = binding.owner
            elif value in aliases:
                value = aliases[value][0]
            elif len(owners.get(value, ())) > 1:
                # A shared reduced name has no proven owner, including another
                # ambiguous reduced mention elsewhere in the same request.
                value = json.dumps([value, "unresolved", span.start, span.end])
        keys.append((span.label, value))
    return keys


def link_entities(text, spans, *, secret=None, patient_id=None) -> list[Entity]:
    """Assign identities using exact values and unambiguous literal name aliases.

    Pass both secret and patient_id for patient-scoped keyed identities. Existing
    IDs are retained; numbered IDs skip identifiers already in use.
    """
    if not isinstance(text, str) or len(text) > _MAX_OUTPUT:
        raise ValueError("Invalid identity context")
    ordered = checked_entities(text, spans)
    key, patient = (
        _patient_key(secret, patient_id)
        if secret is not None or patient_id is not None
        else (None, None)
    )
    keys = _identity_keys(text, ordered)
    reserved = {span.entity_id for span in ordered if span.entity_id is not None}
    provided: dict[tuple[str, str], set[str]] = {}
    for span, identity in zip(ordered, keys):
        if span.entity_id is not None:
            provided.setdefault(identity, set()).add(span.entity_id)
    assigned = {
        identity: next(iter(ids)) for identity, ids in provided.items() if len(ids) == 1
    }
    counter = 0
    result = []
    for span, identity in zip(ordered, keys):
        entity_id = span.entity_id or assigned.get(identity)
        if entity_id is None:
            if key is not None:
                message = json.dumps(
                    ["clinical-deid-entity-v1", patient, *identity],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                entity_id = (
                    "k" + hmac.new(key, message, hashlib.sha256).hexdigest()[:32]
                )
                if entity_id in reserved:
                    raise ValueError(
                        "Assigned entity identity conflicts with a keyed identity"
                    )
            else:
                counter += 1
                while f"e{counter}" in reserved:
                    counter += 1
                entity_id = f"e{counter}"
            reserved.add(entity_id)
            assigned[identity] = entity_id
        result.append(replace_entity(span, entity_id=entity_id))
    return result


@dataclass(frozen=True)
class PreparedContext:
    """Request-local immutable identity and alias indexes for linked segments.

    This contains source identifiers and must not be persisted or logged. Use
    prepare_context to build it once, then reuse it for every selected field.
    """

    text: str = field(repr=False)
    spans: tuple[Entity, ...] = field(repr=False)
    identities: Mapping[tuple[str, str], frozenset[str]] = field(repr=False)
    aliases: Mapping[str, tuple[str, str]] = field(repr=False)
    bindings: Mapping[tuple[int, int], NameBinding] = field(repr=False)
    occurrence_ids: Mapping[tuple[int, int, str], str] = field(repr=False)


def prepare_context(text, spans, *, secret=None, patient_id=None) -> PreparedContext:
    """Validate and index all linked entities once, preserving assigned IDs."""
    linked = link_entities(text, spans, secret=secret, patient_id=patient_id)
    identities: dict[tuple[str, str], set[str]] = {}
    for span, identity in zip(linked, _identity_keys(text, linked)):
        identities.setdefault(identity, set()).add(span.entity_id)
    return PreparedContext(
        text=text,
        spans=tuple(linked),
        identities=MappingProxyType(
            {key: frozenset(ids) for key, ids in identities.items()}
        ),
        aliases=MappingProxyType(_name_aliases(text, linked)),
        bindings=nickname_bindings(text, linked),
        occurrence_ids=MappingProxyType({(s.start, s.end, s.label): s.entity_id for s in linked}),
    )


class _MarkerMatcher:
    """Linear scan over bounded literal keys; never executes caller regexes."""

    def __init__(self, markers):
        self.markers = markers
        self.edges = [{}]
        self.fail = [0]
        self.terminal = [None]
        self.suffix = [None]
        for index, marker in enumerate(markers):
            node = 0
            for char in marker:
                child = self.edges[node].get(char)
                if child is None:
                    child = len(self.edges)
                    self.edges[node][char] = child
                    self.edges.append({})
                    self.fail.append(0)
                    self.terminal.append(None)
                    self.suffix.append(None)
                node = child
            if self.terminal[node] is not None:
                raise ValueError("Duplicate restoration markers")
            self.terminal[node] = index
        queue = deque(self.edges[0].values())
        while queue:
            node = queue.popleft()
            for char, child in self.edges[node].items():
                queue.append(child)
                fallback = self.fail[node]
                while fallback and char not in self.edges[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.edges[fallback].get(char, 0)
                failure = self.fail[child]
                self.suffix[child] = (
                    failure
                    if self.terminal[failure] is not None
                    else self.suffix[failure]
                )

    def matches(self, text):
        node = 0
        for end, char in enumerate(text, 1):
            while node and char not in self.edges[node]:
                node = self.fail[node]
            node = self.edges[node].get(char, 0)
            output = node
            while output is not None:
                index = self.terminal[output]
                if index is not None:
                    yield end - len(self.markers[index]), end, index
                output = self.suffix[output]


def _mapping_matcher(mappings):
    markers = [item.marker for item in mappings]
    matcher = _MarkerMatcher(markers)
    for index, marker in enumerate(markers):
        for _, _, match in matcher.matches(marker):
            if match != index:
                raise ValueError("Overlapping restoration markers")
    return matcher


def reidentify(text, mappings, *, allowed_types=None, restore_sensitive=False):
    """Restore literal, unambiguous markers once; copied values are not rescanned.

    Identifier restoration requires restore_sensitive=True. Only explicitly
    allowlisted CONDITION/MEDICATION labels can restore without that opt-in.
    """
    request = ReidentifyRequest.model_validate(
        dict(
            text=text,
            mappings=mappings,
            allowed_types=allowed_types,
            restore_sensitive=restore_sensitive,
        )
    )
    matcher = _mapping_matcher(request.mappings)
    # No marker contains another, so match ends and starts have the same
    # ordering. Stream matches instead of materializing a list proportional to
    # the number of occurrences in the caller's text.
    allowed = set(request.allowed_types) if request.allowed_types is not None else None
    parts, cursor, count, output_length = [], 0, 0, 0
    previous_end = 0
    for start, end, index in matcher.matches(request.text):
        if start < previous_end:
            raise ValueError("Overlapping restoration markers in text")
        previous_end = end
        item = request.mappings[index]
        permitted = (allowed is None or item.label in allowed) and (
            request.restore_sensitive
            or (allowed is not None and item.label in _PUBLIC_CONCEPT_TYPES)
        )
        if not permitted:
            continue
        unchanged = request.text[cursor:start]
        output_length += len(unchanged) + len(item.original)
        if output_length > _MAX_OUTPUT:
            raise ValueError("Restored output exceeds the size limit")
        parts.extend((unchanged, item.original))
        cursor, count = end, count + 1
    if output_length + len(request.text) - cursor > _MAX_OUTPUT:
        raise ValueError("Restored output exceeds the size limit")
    parts.append(request.text[cursor:])
    return {"text": "".join(parts), "restored_count": count}


def render(
    text,
    spans,
    *,
    options,
    mode="redact",
    patient_id=None,
    secret=None,
    date_shift_days=None,
    context_text=None,
    context_spans=None,
    prepared_context=None,
    index_offset=0,
    context_offset=None,
):
    """Render configured replacements with input/output character offsets.

    The options' type selects the default action. Context supplies all linked
    messages for alias ownership; preassigned identities are always preserved.
    For many linked segments, prepare context once and pass prepared_context
    instead of context_text/context_spans to avoid rebuilding the global index.
    context_offset identifies this exact segment in shared text; arbitrary
    nickname proofs never transfer by spelling or by a supplied entity ID.
    Reversible output requires unambiguous literal replacement tokens: patterns
    using {index} support distinct spellings of the same linked identity.
    """
    options = ReplacementOptions.model_validate(options)
    if type(index_offset) is not int or not 0 <= index_offset <= 1_000_000:
        raise ValueError("index_offset must be a nonnegative bounded integer")
    if mode not in {"redact", "synthetic"}:
        raise ValueError("mode must be redact or synthetic")
    if not isinstance(text, str) or len(text) > _MAX_TEXT:
        raise ValueError("Text must contain at most 100000 characters")
    ordered = checked_entities(text, spans)
    if prepared_context is not None:
        if not isinstance(prepared_context, PreparedContext):
            raise ValueError("Invalid prepared replacement context")
        if context_text is not None or context_spans is not None:
            raise ValueError(
                "Use either prepared_context or context_text/context_spans"
            )
    elif (context_text is None) != (context_spans is None):
        raise ValueError("Both context_text and context_spans are required")
    shared_context = prepared_context is not None or context_text is not None
    synthetic = options.type == "synthetic" or "synthetic" in options.per_type.values()
    key, patient = (
        _patient_key(secret, patient_id)
        if synthetic or options.entity_ids == "keyed"
        else (None, None)
    )
    if date_shift_days is not None and (
        not synthetic
        or type(date_shift_days) is not int
        or not 1 <= abs(date_shift_days) <= 3650
    ):
        raise ValueError(
            "date_shift_days requires synthetic replacement and a nonzero integer between -3650 and 3650"
        )
    offset = 0
    if synthetic:
        if date_shift_days is not None:
            offset = date_shift_days
        else:
            shift_index = (
                int.from_bytes(_digest(key, patient, "DATE_SHIFT", "")[:8], "big") % 730
            )
            offset = shift_index - 365 if shift_index < 365 else shift_index - 364
    link_kwargs = (
        dict(secret=key, patient_id=patient) if options.entity_ids == "keyed" else {}
    )
    if prepared_context is None:
        prepared_context = prepare_context(
            text if context_text is None else context_text,
            ordered if context_spans is None else context_spans,
            **link_kwargs,
        )
    context = prepared_context.text
    context_ids = prepared_context.identities
    aliases = prepared_context.aliases
    if context_offset is None and context == text:
        context_offset = 0
    if context_offset is not None and (
        type(context_offset) is not int or context_offset < 0
        or context_offset + len(text) > len(context)
        or context[context_offset:context_offset + len(text)] != text
    ):
        raise ValueError("context_offset must identify the exact text segment")
    bindings = prepared_context.bindings if context_offset is not None else {}

    if shared_context:
        enriched = []
        for span in ordered:
            value = text[span.start : span.end]
            if span.label == "NAME":
                _, value = _name_identity(value)
                value = aliases.get(value, (value, "full"))[0]
            if context_offset is not None:
                global_key = (span.start + context_offset, span.end + context_offset, span.label)
                exact_id = prepared_context.occurrence_ids.get(global_key)
                ids = {exact_id} if exact_id is not None else set()
            else:
                ids = context_ids.get((span.label, value), set())
            if span.entity_id is None:
                if len(ids) != 1:
                    raise ValueError(
                        "Context does not establish a unique identity; supply assigned local entity IDs"
                    )
                span = replace_entity(span, entity_id=next(iter(ids)))
            enriched.append(span)
        ordered = enriched
    ordered = link_entities(text, ordered, **link_kwargs)
    parts, entities, warnings, mappings = [], [], [], []
    by_marker = {}
    cursor, output_length, shifted_dates, redacted_dates = 0, 0, 0, 0
    all_dates_shifted = synthetic
    patient_consistent = synthetic
    for index, span in enumerate(ordered):
        original = text[span.start : span.end]
        action = options.per_type.get(span.label, options.type)
        marker = options.pattern.format(
            label=span.label, id=span.entity_id, index=index_offset + index + 1
        )
        if span.source == "context_fragment" and action == "synthetic":
            action = "marker"
        if action == "keep":
            replacement = original
        elif action == "mask":
            replacement = options.mask_character * len(original)
        elif action == "marker":
            replacement = marker
        elif span.label == "DATE":
            replacement, warning, shifted = _shift_date(original, offset)
            shifted_dates += int(shifted)
            redacted_dates += int(not shifted)
            if warning:
                warnings.append(f"Entity {index} (DATE): {warning}")
        elif span.label in ENTITY_TYPES:
            binding = bindings.get((span.start + (context_offset or 0),
                                    span.end + (context_offset or 0)))
            # A supplied conflicting local ID cannot borrow a global proof.
            expected_id = prepared_context.occurrence_ids.get(
                (span.start + (context_offset or 0), span.end + (context_offset or 0), span.label)
            )
            if binding is not None and span.entity_id != expected_id:
                binding = None
            name_warnings = []
            replacement = _synthetic_value(original, span.label, key, patient, aliases,
                                           name_binding=binding, name_warnings=name_warnings)
            warnings.extend(f"Entity {index} (NAME): {warning}" for warning in name_warnings)
        else:
            replacement = marker
        if span.label == "DATE" and (action != "synthetic" or not shifted):
            all_dates_shifted = False
        if action != "synthetic" or span.label not in ENTITY_TYPES:
            patient_consistent = False
        unchanged = text[cursor : span.start]
        processed_start = output_length + len(unchanged)
        output_length = processed_start + len(replacement)
        if output_length > _MAX_OUTPUT:
            raise ValueError("Replacement output exceeds the size limit")
        parts.extend((unchanged, replacement))
        entities.append(
            {
                **span.as_dict(),
                "replacement": replacement,
                "processed_start": processed_start,
                "processed_end": output_length,
            }
        )
        if options.include_mapping and replacement != original:
            mapping = {"marker": replacement, "original": original, "label": span.label}
            previous = by_marker.get(replacement)
            if previous is not None and previous != mapping:
                raise ValueError(
                    "Ambiguous restoration mapping; use distinct markers such as a pattern containing index"
                )
            if previous is None:
                by_marker[replacement] = mapping
                mappings.append(mapping)
        cursor = span.end
    if output_length + len(text) - cursor > _MAX_OUTPUT:
        raise ValueError("Replacement output exceeds the size limit")
    parts.append(text[cursor:])
    result = {
        "text": "".join(parts),
        "entities": entities,
        "warnings": warnings,
        "preservation": {
            "clinical_text_outside_spans_preserved": True,
            "date_intervals_preserved": all_dates_shifted,
            "exact_dates_shifted": shifted_dates,
            "unshiftable_dates_redacted": redacted_dates,
            "patient_consistent_replacements": patient_consistent,
        },
    }
    if options.include_mapping:
        request = ReidentifyRequest(text=result["text"], mappings=mappings)
        _mapping_matcher(request.mappings)
        # Literal C-level searches avoid walking the entire shared context in
        # Python once per selected field. No caller-supplied regex is executed.
        if any(
            item.marker in context or (context != text and item.marker in text)
            for item in request.mappings
        ):
            raise ValueError("Restoration marker collides with original text")
        # Validate adjacent replacements do not create overlapping tokens.
        reidentify(result["text"], mappings)
        result["mappings"] = mappings
    return result
