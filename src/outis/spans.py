"""Caller-supplied spans: validation, keyed text fingerprints and span files.

Detection and replacement are separate stages. ``/v1/detect`` (or any other
detector) produces spans, a reviewer may correct them, and ``/v1/replace``
applies them. Errors here name only an entity index or a file line and a fixed
reason; they never echo text, span contents, note IDs or patient IDs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import replace as replace_entity
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from outis.entities import Entity
from outis.replacement.primitives import validated_secret

SPAN_FORMAT = "outis-spans-v1"
SUPPLIED_SOURCE = "supplied"
# Spans a reviewer added. Like policy rules and site dictionaries, they are an
# explicit choice, so the ages policy never drops them.
REVIEW_SOURCE = "review"
# Same rule as policy custom_entities labels; every clinical label matches it.
LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,47}\Z")
FINGERPRINT = re.compile(r"tf1-[0-9a-f]{32}\Z")
MAX_ENTITIES = 20_000
_ENTITY_FIELDS = frozenset({"start", "end", "label", "score", "source", "entity_id"})
_RECORD_FIELDS = frozenset(
    {
        "format",
        "note_id",
        "patient_id",
        "entities",
        "model_version",
        "policy",
        "text_fingerprint",
        "edited",
    }
)
_NOTE_ID = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")
# Versioned tag; changing it invalidates every fingerprint already issued.
_FINGERPRINT_DOMAIN = b"outis-text-fingerprint-v1"


class SpanError(ValueError):
    """An invalid supplied span or span file; the message is safe to return."""


class ReplacePolicy(BaseModel):
    """The policy fields that act on supplied spans.

    Every other detection policy field acts while detecting, so it already
    shaped the supplied spans. ``ages`` drops supplied AGE spans of 89 and under
    exactly as detection would.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    ages: Literal["over_89", "all"] = "over_89"


def validate_replace_policy(policy) -> ReplacePolicy:
    try:
        if isinstance(policy, ReplacePolicy):
            policy = policy.model_dump()
        return ReplacePolicy.model_validate({} if policy is None else policy)
    except (ValueError, TypeError):
        raise ValueError("Invalid replacement policy") from None


def replace_policy_metadata(policy) -> dict:
    from outis.policy import POLICY_VERSION

    return {"version": POLICY_VERSION, "ages": validate_replace_policy(policy).ages}


def _field(entity, name):
    if isinstance(entity, dict):
        return entity.get(name)
    return getattr(entity, name, None)


def supplied_entities(text: str, entities) -> list[Entity]:
    """Validate supplied spans and return them sorted.

    Offsets are zero-based Unicode code points with an exclusive end. Start, end
    and label drive replacement. A valid score is carried through, entity_id is
    ignored, and source is kept only for ``filter_ages``; it defaults to
    ``supplied``.
    """
    if not isinstance(text, str):
        raise SpanError("Text must be a string")
    if not isinstance(entities, (list, tuple)):
        raise SpanError("entities must be a list")
    if len(entities) > MAX_ENTITIES:
        raise SpanError(f"At most {MAX_ENTITIES} entities are supported")
    checked = []
    for index, entity in enumerate(entities):
        if isinstance(entity, dict) and set(entity) - _ENTITY_FIELDS:
            raise SpanError(f"Entity {index}: unsupported field")
        start, end = _field(entity, "start"), _field(entity, "end")
        label, score = _field(entity, "label"), _field(entity, "score")
        source = _field(entity, "source")
        if type(start) is not int or type(end) is not int:
            raise SpanError(f"Entity {index}: start and end must be integers")
        if not 0 <= start < end <= len(text):
            raise SpanError(f"Entity {index}: offsets are outside the text")
        if not isinstance(label, str) or not LABEL.match(label):
            raise SpanError(f"Entity {index}: invalid label")
        if score is None:
            score = 1.0
        elif (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise SpanError(f"Entity {index}: score must be between 0 and 1")
        if source is None:
            source = SUPPLIED_SOURCE
        elif not isinstance(source, str) or not 1 <= len(source) <= 128:
            raise SpanError(f"Entity {index}: invalid source")
        checked.append((start, end, index, Entity(start, end, label, score, source)))
    checked.sort(key=lambda item: item[:2])
    for previous, current in zip(checked, checked[1:]):
        if current[0] < previous[1]:
            raise SpanError(f"Entity {current[2]}: overlaps another entity")
    return [item[3] for item in checked]


def _ages_apply(source: str) -> bool:
    # Detection applies the ages policy to detector spans only, before policy
    # rules and site dictionaries add theirs; reviewer additions are explicit.
    from outis.site_dictionaries import SOURCE_PREFIX

    return not (
        source.startswith(("policy:", SOURCE_PREFIX)) or source == REVIEW_SOURCE
    )


def filter_ages(text: str, entities: list[Entity], ages: str) -> list[Entity]:
    """Apply the ages policy as detection does, then mark every span supplied.

    AGE spans from a detector (source ``model``, any other source, or none) of
    89 and under are left in place under ``over_89``. Spans a policy rule, a
    site dictionary or a reviewer chose are always replaced, so replacing the
    spans ``/v1/detect`` returned matches ``/v1/deidentify``.
    """
    from outis.policy import keep_ages

    return [
        replace_entity(entity, source=SUPPLIED_SOURCE)
        for entity in entities
        if not _ages_apply(entity.source) or keep_ages(text, [entity], ages)
    ]


def text_fingerprint(secret, text: str) -> str:
    """A keyed fingerprint binding spans to the exact text they index.

    A bare hash of a short note could be inverted by guessing, so this is an
    HMAC under a key derived from the server secret. It identifies text only to
    holders of that secret and reveals nothing else about it.
    """
    try:
        secret = validated_secret(secret)
    except ValueError:
        raise ValueError(
            "Text fingerprints require a secret containing at least 32 bytes"
        ) from None
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    key = hmac.new(secret, _FINGERPRINT_DOMAIN, hashlib.sha256).digest()
    digest = hmac.new(key, text.encode("utf-8", "surrogatepass"), hashlib.sha256)
    return "tf1-" + digest.hexdigest()[:32]


def check_fingerprint(secret, text: str, fingerprint) -> None:
    if not isinstance(fingerprint, str) or not FINGERPRINT.match(fingerprint):
        raise SpanError("Invalid text_fingerprint")
    if not hmac.compare_digest(text_fingerprint(secret, text), fingerprint):
        raise SpanError("text_fingerprint does not match the text")


# Span files: one JSON object per note, written by detection and read by
# replacement and review. They hold offsets and labels, never note text, but
# note and patient IDs can be identifiers, so they are written owner-only.


def span_record(
    note_id: str,
    detection: dict,
    *,
    patient_id: str | None = None,
) -> dict:
    """A span file record from a detect result (``Deidentifier.detect`` output)."""
    record = {"format": SPAN_FORMAT, "note_id": note_id}
    if patient_id is not None:
        record["patient_id"] = patient_id
    record["entities"] = [
        {
            key: entity[key]
            for key in ("start", "end", "label", "score", "source")
            if key in entity
        }
        for entity in detection["entities"]
    ]
    for key in ("model_version", "policy", "text_fingerprint"):
        if detection.get(key) is not None:
            record[key] = detection[key]
    return record


def _check_record(record, line: int) -> dict:
    where = f"Span file line {line}"
    if not isinstance(record, dict):
        raise SpanError(f"{where}: expected a JSON object")
    if set(record) - _RECORD_FIELDS:
        raise SpanError(f"{where}: unsupported field")
    if record.get("format") != SPAN_FORMAT:
        raise SpanError(f"{where}: format must be {SPAN_FORMAT}")
    note_id = record.get("note_id")
    if not isinstance(note_id, str) or not _NOTE_ID.match(note_id):
        raise SpanError(f"{where}: invalid note_id")
    patient = record.get("patient_id")
    if patient is not None and (
        not isinstance(patient, str) or not 1 <= len(patient) <= 256
    ):
        raise SpanError(f"{where}: invalid patient_id")
    entities = record.get("entities")
    if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
        raise SpanError(f"{where}: entities must be a list")
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or set(entity) - _ENTITY_FIELDS:
            raise SpanError(f"{where}: entity {index} is invalid")
    fingerprint = record.get("text_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not FINGERPRINT.match(fingerprint)
    ):
        raise SpanError(f"{where}: invalid text_fingerprint")
    version = record.get("model_version")
    if version is not None and not isinstance(version, str):
        raise SpanError(f"{where}: invalid model_version")
    if record.get("policy") is not None and not isinstance(record["policy"], dict):
        raise SpanError(f"{where}: invalid policy")
    if record.get("edited") is not None and type(record["edited"]) is not bool:
        raise SpanError(f"{where}: invalid edited flag")
    return record


def read_span_file(path) -> dict[str, dict]:
    """Read a span file into records keyed by note ID, checking every line."""
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (ValueError, RecursionError):
                raise SpanError(f"Span file line {line}: invalid JSON") from None
            record = _check_record(record, line)
            if record["note_id"] in records:
                raise SpanError(f"Span file line {line}: duplicate note_id")
            records[record["note_id"]] = record
    return records


def private_write(path, content: str) -> None:
    """Write UTF-8 text readable and writable by the owner only."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        # O_CREAT's mode does not apply to a file that already existed.
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(content)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def jsonl(records: Iterable[dict]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def read_notes(
    path,
    *,
    id_column: str = "note_id",
    text_column: str = "text",
    patient_column: str | None = None,
) -> list[dict]:
    """Read a JSONL note table into ``note_id``/``text``/``patient_id`` rows."""
    notes, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            where = f"Input line {line}"
            try:
                row = json.loads(raw)
            except (ValueError, RecursionError):
                raise SpanError(f"{where}: invalid JSON") from None
            if not isinstance(row, dict):
                raise SpanError(f"{where}: expected a JSON object")
            note_id, text = row.get(id_column), row.get(text_column)
            if not isinstance(note_id, str) or not _NOTE_ID.match(note_id):
                raise SpanError(f"{where}: missing or invalid note ID")
            if note_id in seen:
                raise SpanError(f"{where}: duplicate note ID")
            if not isinstance(text, str):
                raise SpanError(f"{where}: missing or invalid text")
            patient = row.get(patient_column) if patient_column else None
            if patient_column and (
                not isinstance(patient, str) or not 1 <= len(patient) <= 256
            ):
                raise SpanError(f"{where}: missing or invalid patient ID")
            seen.add(note_id)
            notes.append(
                {"note_id": note_id, "text": text, "patient_id": patient, "line": line}
            )
    return notes


def pair_notes(notes: list[dict], records: dict[str, dict]) -> list[tuple[dict, dict]]:
    """Match every note to exactly one span record, and fill in its patient ID."""
    pairs = []
    for note in notes:
        record = records.get(note["note_id"])
        if record is None:
            raise SpanError(f"Input line {note['line']}: no span record for this note")
        recorded = record.get("patient_id")
        if note["patient_id"] is None:
            note = {**note, "patient_id": recorded}
        elif recorded is not None and recorded != note["patient_id"]:
            raise SpanError(
                f"Input line {note['line']}: patient ID differs from the span record"
            )
        pairs.append((note, record))
    if len(pairs) != len(records):
        raise SpanError(f"{len(records) - len(pairs)} span records match no input note")
    return pairs
