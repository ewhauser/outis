"""Detect, then replace supplied spans: parity, validation, span files, review.

Every value here is invented. No model is loaded.
"""

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient

from outis import batch, cli
from outis.api import create_app
from outis.review import MODEL_SCRIPT, build_report
from outis.service import Deidentifier
from outis.spans import (
    SPAN_FORMAT,
    SpanError,
    pair_notes,
    read_notes,
    read_span_file,
    span_record,
    supplied_entities,
    text_fingerprint,
)
from outis.types import Span

SECRET = b"outis-supplied-spans-test-secret-0123456789"
OTHER_SECRET = b"outis-supplied-spans-other-secret-9876543210"
VALUES = {
    "Mara Calder": "NAME",
    "Calder": "NAME",
    "Exampleford Rehabilitation Hospital": "FACILITY",
    "Brindlemoor": "LOCATION",
    "12 Quarry Lane": "ADDRESS",
    "555-867-5309": "PHONE",
    "mara.calder@example.org": "EMAIL",
    "A12345": "MRN",
    "03/02/2024": "DATE",
    "2024-03-09": "DATE",
    "93": "AGE",
    "45": "AGE",
}
TEXT = (
    "Dr. Mara Calder saw the patient at Exampleford Rehabilitation Hospital in "
    "Brindlemoor on 03/02/2024. Address 12 Quarry Lane. Call 555-867-5309 or "
    "mara.calder@example.org. MRN A12345. Age 93; sister age 45. Badge QX-7781.\n"
    "Calder followed up on 2024-03-09."
)
SECOND = "Calder phoned from Brindlemoor on 2024-03-09 about MRN A12345."
PATTERN = re.compile(
    "|".join(
        rf"(?<![\w.]){re.escape(value)}(?![\w])"
        for value in sorted(VALUES, key=len, reverse=True)
    )
)


class Detector:
    def detect_batch(self, texts):
        return [
            [
                Span(match.start(), match.end(), VALUES[match[0]], 0.8125)
                for match in PATTERN.finditer(text)
            ]
            for text in texts
        ]


class NoDetector:
    def detect_batch(self, texts):
        raise AssertionError("replace must not run detection")


def service(secret=SECRET):
    return Deidentifier(secret=secret, detector=Detector())


@pytest.fixture
def api():
    with ExitStack() as stack:
        yield lambda deid: stack.enter_context(TestClient(create_app(deid)))


BADGE = {"custom_entities": [{"label": "BADGE", "pattern": r"QX-\d{4}"}]}
CASES = [
    {},
    {"mode": "synthetic", "patient_id": "patient-a"},
    {"replacement": {"type": "marker"}},
    {"replacement": {"type": "mask", "mask_character": "#"}},
    {
        "replacement": {"pattern": "[{label}_{index}]", "include_mapping": True},
        "policy": {"ages": "all"},
    },
    {
        "mode": "synthetic",
        "patient_id": "patient-a",
        "replacement": {"type": "synthetic", "entity_ids": "keyed"},
    },
    {
        "mode": "synthetic",
        "patient_id": "patient-b",
        "date_shift_days": -41,
        "replacement": {"type": "synthetic", "style": "realistic"},
    },
    {
        "mode": "synthetic",
        "patient_id": "patient-a",
        "replacement": {
            "type": "synthetic",
            "style": "realistic",
            "keep_region": False,
            "scope": "project",
            "entity_ids": "keyed",
        },
        "project_id": "trial-a",
    },
    {
        "mode": "synthetic",
        "patient_id": "patient-c",
        "replacement": {"type": "synthetic", "scope": "project"},
        "project_id": "trial-b",
        "policy": {"ages": "all"},
    },
    {
        "replacement": {"per_type": {"BADGE": "keep", "DATE": "mask"}},
        "policy": BADGE,
    },
]


def _detect_policy(case):
    return case.get("policy")


def _replace_policy(case):
    policy = case.get("policy")
    if policy is None:
        return None
    return {"ages": policy["ages"]} if "ages" in policy else {}


def _comparable(result):
    entities = [
        {key: value for key, value in entity.items() if key != "source"}
        for entity in result["entities"]
    ]
    rest = {
        key: value
        for key, value in result.items()
        if key not in {"entities", "policy", "model_version"}
    }
    return rest, entities


@pytest.mark.parametrize("case", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("text", [TEXT, SECOND], ids=["note", "short"])
def test_replacing_detected_spans_matches_deidentify(case, text):
    deid = service()
    options = {key: value for key, value in case.items() if key != "policy"}
    expected = deid.deidentify(text, policy=case.get("policy"), **options)
    detected = deid.detect(
        text,
        policy=_detect_policy(case),
        project_id=case.get("project_id", "main"),
        include_fingerprint=True,
    )
    # A fresh replace-only instance proves no detection runs.
    applier = Deidentifier.for_replacement(secret=SECRET)
    result = applier.replace(
        text,
        detected["entities"],
        policy=_replace_policy(case),
        text_fingerprint=detected["text_fingerprint"],
        **options,
    )
    assert result["text"] == expected["text"]
    assert _comparable(result) == _comparable(expected)
    assert {entity["source"] for entity in result["entities"]} <= {"supplied"}
    assert result["model_version"] == "supplied"
    assert ("policy" in result) == ("policy" in expected)
    if "policy" in result:
        assert result["policy"] == {
            "version": "1",
            "ages": expected["policy"]["ages"],
        }


def test_http_detect_then_replace_matches_deidentify(api):
    client = api(service())
    for case in CASES:
        options = {key: value for key, value in case.items() if key != "policy"}
        body = {"text": TEXT, **options}
        if case.get("policy") is not None:
            body["policy"] = case["policy"]
        expected = client.post("/v1/deidentify", json=body)
        detect_body = {"text": TEXT, "include_fingerprint": True}
        if case.get("policy") is not None:
            detect_body["policy"] = case["policy"]
        if "project_id" in case:
            detect_body["project_id"] = case["project_id"]
        detected = client.post("/v1/detect", json=detect_body).json()
        replace_body = {
            **body,
            "entities": detected["entities"],
            "text_fingerprint": detected["text_fingerprint"],
        }
        if case.get("policy") is not None:
            replace_body["policy"] = _replace_policy(case)
        result = client.post("/v1/replace", json=replace_body)
        assert expected.status_code == result.status_code == 200
        assert _comparable(result.json()) == _comparable(expected.json())
        # The same body through the batch endpoint and the WebSocket.
        batch = client.post("/v1/replace/batch", json={"items": [replace_body]})
        assert batch.json() == [result.json()]
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"operation": "replace", "payload": replace_body})
            assert socket.receive_json()["result"] == result.json()


def _gaps(text, entities, start="start", end="end"):
    gaps, cursor = [], 0
    for entity in entities:
        gaps.append(text[cursor : entity[start]])
        cursor = entity[end]
    return [*gaps, text[cursor:]]


@pytest.mark.parametrize(
    "first, second",
    [
        ({"project_id": "trial-a"}, {"project_id": "trial-b"}),
        ({"secret": SECRET}, {"secret": OTHER_SECRET}),
        ({"style": "obvious"}, {"style": "realistic"}),
    ],
)
def test_reapplying_under_another_key_changes_only_surrogates(first, second):
    entities = service().detect(TEXT)["entities"]

    def run(secret=SECRET, project_id="trial-a", style="obvious"):
        return Deidentifier.for_replacement(secret=secret).replace(
            TEXT,
            entities,
            mode="synthetic",
            patient_id="patient-a",
            project_id=project_id,
            replacement={"type": "synthetic", "scope": "project", "style": style},
        )

    a, b = run(**first), run(**second)
    assert a["text"] != b["text"]
    assert [(e["start"], e["end"], e["label"]) for e in a["entities"]] == [
        (e["start"], e["end"], e["label"]) for e in b["entities"]
    ]
    changed = [
        x["replacement"] != y["replacement"]
        for x, y in zip(a["entities"], b["entities"])
        if x["label"] not in {"AGE"}
    ]
    assert any(changed)
    # Every character outside a span is copied exactly, in both outputs.
    source = _gaps(TEXT, a["entities"])
    for result in (a, b):
        assert (
            _gaps(
                result["text"],
                result["entities"],
                *("processed_start", "processed_end"),
            )
            == source
        )


def test_supplied_spans_from_another_detector_and_custom_labels():
    text = "Ward Q7 badge QX-7781 for Mara Calder."
    entities = [
        {"start": 26, "end": 37, "label": "NAME"},
        {"start": 14, "end": 21, "label": "BADGE"},
    ]
    result = Deidentifier.for_replacement().replace(
        text, entities, replacement={"pattern": "<{label}>"}
    )
    assert result["text"] == "Ward Q7 badge <BADGE> for <NAME>."
    # Entities come back sorted, with the ignored fields replaced.
    assert [e["label"] for e in result["entities"]] == ["BADGE", "NAME"]
    assert all(e["score"] == 1.0 for e in result["entities"])
    assert result["entity_count"] == 2


def test_ages_policy_applies_to_supplied_spans():
    text = "Age 45. Mother age 93."
    entities = [
        {"start": 4, "end": 6, "label": "AGE"},
        {"start": 19, "end": 21, "label": "AGE"},
    ]
    deid = Deidentifier.for_replacement()
    assert deid.replace(text, entities)["text"] == "Age 45. Mother age [AGE_90_PLUS]."
    every = deid.replace(text, entities, policy={"ages": "all"})
    assert every["text"] == "Age [AGE]. Mother age [AGE_90_PLUS]."
    assert every["policy"] == {"version": "1", "ages": "all"}


def test_legacy_and_extended_shapes_match_deidentify():
    deid = service()
    entities = deid.detect(TEXT)["entities"]
    legacy = deid.replace(TEXT, entities)
    assert set(legacy["entities"][0]) == {
        "start",
        "end",
        "label",
        "replacement",
        "score",
        "source",
    }
    assert "policy" not in legacy
    extended = deid.replace(TEXT, entities, replacement={"type": "marker"})
    assert {"entity_id", "processed_start", "processed_end"} <= set(
        extended["entities"][0]
    )


SENSITIVE = ("Mara", "Calder", "Brindlemoor", "A12345", "555-867", "QX-7781")


def _assert_no_content(message):
    for fragment in SENSITIVE:
        assert fragment not in message


@pytest.mark.parametrize(
    "entities, reason",
    [
        (
            [
                {"start": 4, "end": 15, "label": "NAME"},
                {"start": 9, "end": 15, "label": "NAME"},
            ],
            "Entity 1: overlaps another entity",
        ),
        (
            [
                {"start": 4, "end": 15, "label": "NAME"},
                {"start": 4, "end": 15, "label": "NAME"},
            ],
            "Entity 1: overlaps another entity",
        ),
        (
            [{"start": 4, "end": 100_000, "label": "NAME"}],
            "Entity 0: offsets are outside the text",
        ),
        (
            [{"start": -1, "end": 15, "label": "NAME"}],
            "Entity 0: offsets are outside the text",
        ),
        (
            [{"start": 15, "end": 15, "label": "NAME"}],
            "Entity 0: offsets are outside the text",
        ),
        (
            [{"start": True, "end": 15, "label": "NAME"}],
            "Entity 0: start and end must be integers",
        ),
        (
            [{"start": 4.0, "end": 15, "label": "NAME"}],
            "Entity 0: start and end must be integers",
        ),
        ([{"start": 4, "end": 15, "label": "Mara Calder"}], "Entity 0: invalid label"),
        ([{"start": 4, "end": 15, "label": "N" * 49}], "Entity 0: invalid label"),
        (
            [{"start": 4, "end": 15, "label": "NAME", "score": 2}],
            "Entity 0: score must be between 0 and 1",
        ),
        (
            [{"start": 4, "end": 15, "label": "NAME", "text": "Mara Calder"}],
            "Entity 0: unsupported field",
        ),
        (
            [{"start": 0, "end": 1, "label": "ID"}] * 20_001,
            "At most 20000 entities are supported",
        ),
    ],
)
def test_invalid_spans_are_rejected_without_content(entities, reason):
    with pytest.raises(SpanError) as caught:
        Deidentifier.for_replacement().replace(TEXT, entities)
    assert str(caught.value) == reason
    _assert_no_content(str(caught.value))


def test_invalid_requests_over_http_never_echo_content(api):
    client = api(Deidentifier(secret=SECRET, detector=NoDetector()))
    fingerprint = text_fingerprint(SECRET, TEXT)
    bodies = [
        {
            "text": TEXT,
            "entities": [
                {"start": 4, "end": 15, "label": "NAME"},
                {"start": 9, "end": 20, "label": "NAME"},
            ],
        },
        {"text": TEXT, "entities": [{"start": 4, "end": 9999, "label": "NAME"}]},
        {"text": TEXT, "entities": [{"start": 4, "end": 15, "label": "Mara Calder"}]},
        {"text": TEXT, "entities": [{"start": "4", "end": 15, "label": "NAME"}]},
        {
            "text": TEXT,
            "entities": [{"start": 4, "end": 15, "label": "NAME", "value": "Calder"}],
        },
        {"text": TEXT, "entities": [], "policy": {"profile": "extended"}},
        {
            "text": TEXT,
            "entities": [],
            "policy": {"custom_entities": [{"label": "X", "pattern": "Calder"}]},
        },
        {"text": TEXT, "entities": [], "text_fingerprint": "tf1-" + "0" * 32},
        {"text": TEXT, "entities": [], "text_fingerprint": "A12345"},
        {"text": TEXT + "!", "entities": [], "text_fingerprint": fingerprint},
        {"text": TEXT, "entities": [], "mode": "synthetic"},
        {"text": TEXT, "entities": [], "replacement": {"scope": "project"}},
    ]
    for body in bodies:
        response = client.post("/v1/replace", json=body)
        assert response.status_code == 422, body
        _assert_no_content(response.text)
    assert (
        client.post(
            "/v1/replace",
            json={"text": TEXT, "entities": [], "text_fingerprint": fingerprint},
        ).status_code
        == 200
    )


def test_text_fingerprint_is_keyed_and_optional():
    fingerprint = text_fingerprint(SECRET, TEXT)
    assert re.fullmatch(r"tf1-[0-9a-f]{32}", fingerprint)
    assert fingerprint == text_fingerprint(SECRET, TEXT)
    assert fingerprint != text_fingerprint(OTHER_SECRET, TEXT)
    assert fingerprint != text_fingerprint(SECRET, TEXT + " ")
    # Never a bare digest of the text that a guess could confirm.
    for digest in (hashlib.sha256, hashlib.md5, hashlib.sha1):
        assert fingerprint[4:] not in digest(TEXT.encode()).hexdigest()
    with pytest.raises(ValueError, match="secret"):
        text_fingerprint(None, TEXT)
    # Without a secret the request fails before any detection runs.
    deid = Deidentifier(detector=NoDetector())
    with pytest.raises(ValueError):
        deid.detect(TEXT, include_fingerprint=True)
    with pytest.raises(ValueError):
        deid.replace(TEXT, [], text_fingerprint=fingerprint)


def test_existing_endpoints_are_unchanged(api):
    client = api(service())
    detected = client.post("/v1/detect", json={"text": TEXT}).json()
    assert set(detected) == {"entities", "model_version", "entity_count", "policy"}
    batch = client.post("/v1/detect/batch", json={"items": [{"text": TEXT}]}).json()
    assert "text_fingerprint" not in batch[0]
    # Analyze never gained the detect-only flag.
    assert (
        client.post(
            "/v1/analyze", json={"text": TEXT, "include_fingerprint": True}
        ).status_code
        == 422
    )
    assert service().detect(TEXT) == detected


def test_replace_only_instance_cannot_detect():
    deid = Deidentifier.for_replacement(secret=SECRET)
    with pytest.raises(RuntimeError):
        deid.detect(TEXT)
    with pytest.raises(RuntimeError):
        deid.deidentify(TEXT)


def test_supplied_entities_accepts_objects_and_keeps_only_sources():
    spans = supplied_entities(
        "Mara Calder",
        [
            Span(0, 4, "NAME", 0.5),
            {
                "start": 5,
                "end": 11,
                "label": "NAME",
                "entity_id": "k1",
                "source": "model",
            },
        ],
    )
    # The source decides whether the ages policy applies; IDs are reassigned.
    assert [(s.start, s.end, s.score, s.source, s.entity_id) for s in spans] == [
        (0, 4, 0.5, "model", None),
        (5, 11, 1.0, "model", None),
    ]
    unsourced = supplied_entities("Mara", [{"start": 0, "end": 4, "label": "NAME"}])
    assert unsourced[0].source == "supplied"
    for source in ("", 7, "x" * 129):
        with pytest.raises(SpanError, match="^Entity 0: invalid source$"):
            supplied_entities(
                "Mara", [{"start": 0, "end": 4, "label": "NAME", "source": source}]
            )


# Span files.


def _write_notes(path, notes):
    path.write_text(
        "".join(json.dumps(note, ensure_ascii=False) + "\n" for note in notes),
        encoding="utf-8",
    )


NOTES = [
    {"note_id": "note-1", "patient": "patient-a", "body": TEXT},
    {"note_id": "note-2", "patient": "patient-a", "body": SECOND},
]


def test_span_record_round_trip(tmp_path):
    detected = service().detect(TEXT, include_fingerprint=True)
    record = span_record("note-1", detected, patient_id="patient-a")
    assert record["format"] == SPAN_FORMAT
    assert set(record) == {
        "format",
        "note_id",
        "patient_id",
        "entities",
        "model_version",
        "policy",
        "text_fingerprint",
    }
    assert all(
        set(entity) == {"start", "end", "label", "score", "source"}
        for entity in record["entities"]
    )
    # The review report's corrected records add an edited flag.
    edited = {**record, "note_id": "note-2", "entities": [], "edited": True}
    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps(record) + "\n\n" + json.dumps(edited) + "\n", encoding="utf-8"
    )
    assert read_span_file(path) == {"note-1": record, "note-2": edited}


@pytest.mark.parametrize(
    "line, reason",
    [
        ("not json", "Span file line 1: invalid JSON"),
        ("[]", "Span file line 1: expected a JSON object"),
        (
            '{"format": "other", "note_id": "n", "entities": []}',
            "Span file line 1: format must be outis-spans-v1",
        ),
        (
            '{"format": "outis-spans-v1", "entities": []}',
            "Span file line 1: invalid note_id",
        ),
        (
            '{"format": "outis-spans-v1", "note_id": "n", "entities": {}}',
            "Span file line 1: entities must be a list",
        ),
        (
            '{"format": "outis-spans-v1", "note_id": "n", "entities": [], "text": "Mara Calder"}',
            "Span file line 1: unsupported field",
        ),
        (
            '{"format": "outis-spans-v1", "note_id": "n", "entities": [{"start": 0, "end": 4, "label": "NAME", "value": "Mara"}]}',
            "Span file line 1: entity 0 is invalid",
        ),
        (
            '{"format": "outis-spans-v1", "note_id": "n", "entities": [], "text_fingerprint": "Calder"}',
            "Span file line 1: invalid text_fingerprint",
        ),
    ],
)
def test_invalid_span_files_name_only_lines(tmp_path, line, reason):
    path = tmp_path / "spans.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(SpanError) as caught:
        read_span_file(path)
    assert str(caught.value) == reason


def test_duplicate_and_unmatched_records_are_rejected(tmp_path):
    record = {"format": SPAN_FORMAT, "note_id": "note-1", "entities": []}
    path = tmp_path / "spans.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(SpanError, match="line 2: duplicate note_id"):
        read_span_file(path)
    notes_path = tmp_path / "notes.jsonl"
    _write_notes(notes_path, NOTES)
    notes = read_notes(notes_path, text_column="body", patient_column="patient")
    with pytest.raises(SpanError, match="^Input line 2: no span record"):
        pair_notes(notes, {"note-1": record})
    extra = {"note-1": record, "note-2": {**record, "note_id": "note-2"}, "x": record}
    with pytest.raises(SpanError, match="^1 span records match no input note"):
        pair_notes(notes, extra)
    conflict = {
        "note-1": {**record, "patient_id": "patient-z"},
        "note-2": {**record, "note_id": "note-2"},
    }
    with pytest.raises(SpanError, match="patient ID differs") as caught:
        pair_notes(notes, conflict)
    assert "patient-" not in str(caught.value)


# Command line: detect once, replace under different keys, review.


def _run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["outis", *args])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    try:
        cli.main()
    except SystemExit as exit:
        # outis batch exits explicitly, with 0 when every row succeeded.
        if exit.code not in (0, None):
            raise
    return capsys.readouterr()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    for name in ("OUTIS_SECRET", "OUTIS_SITE_DICTIONARIES", "OUTIS_FINGERPRINT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    # outis batch --detect-only loads this detector instead of the model.
    monkeypatch.setattr(batch, "_load_detector", lambda config: Detector())
    notes = tmp_path / "notes.jsonl"
    _write_notes(notes, NOTES)
    return tmp_path, notes


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _columns():
    return ("--text-column", "body", "--patient-column", "patient")


def test_cli_detect_replace_round_trip(corpus, monkeypatch, capsys):
    tmp_path, notes = corpus
    monkeypatch.setenv("OUTIS_SECRET", SECRET.decode())
    spans = tmp_path / "spans.jsonl"
    detect = _run(
        monkeypatch,
        capsys,
        "batch",
        "--detect-only",
        "--input",
        str(notes),
        "--output",
        str(spans),
        *_columns(),
    )
    records = read_span_file(spans)
    assert list(records) == ["note-1", "note-2"]
    assert all(r["text_fingerprint"].startswith("tf1-") for r in records.values())
    assert all("text" not in r for r in records.values())
    if os.name == "posix":
        assert _mode(spans) == 0o600
    policy = tmp_path / "policy.json"
    outputs = {}
    for project in ("trial-a", "trial-b"):
        policy.write_text(
            json.dumps({"replacement": {"type": "synthetic", "scope": "project"}})
        )
        out = tmp_path / f"{project}.jsonl"
        replaced = _run(
            monkeypatch,
            capsys,
            "spans",
            "replace",
            "--input",
            str(notes),
            "--spans",
            str(spans),
            "--output",
            str(out),
            "--mode",
            "synthetic",
            "--policy",
            str(policy),
            "--project-id",
            project,
            *_columns(),
        )
        outputs[project] = [json.loads(line) for line in out.read_text().splitlines()]
        expected = [
            service().deidentify(
                note["body"],
                mode="synthetic",
                patient_id=note["patient"],
                project_id=project,
                replacement={"type": "synthetic", "scope": "project"},
            )
            for note in NOTES
        ]
        assert [row["text"] for row in outputs[project]] == [
            e["text"] for e in expected
        ]
        assert [row["note_id"] for row in outputs[project]] == ["note-1", "note-2"]
        for stream in (detect.err, detect.out, replaced.err, replaced.out):
            _assert_no_content(stream)
            assert "note-1" not in stream and "patient-a" not in stream
    assert [r["text"] for r in outputs["trial-a"]] != [
        r["text"] for r in outputs["trial-b"]
    ]


def test_cli_rejects_spans_for_edited_text(corpus, monkeypatch, capsys):
    tmp_path, notes = corpus
    monkeypatch.setenv("OUTIS_SECRET", SECRET.decode())
    spans = tmp_path / "spans.jsonl"
    _run(
        monkeypatch,
        capsys,
        "batch",
        "--detect-only",
        "--input",
        str(notes),
        "--output",
        str(spans),
        *_columns(),
    )
    _write_notes(notes, [NOTES[0], {**NOTES[1], "body": SECOND + " Calder"}])
    with pytest.raises(SystemExit) as caught:
        _run(
            monkeypatch,
            capsys,
            "spans",
            "replace",
            "--input",
            str(notes),
            "--spans",
            str(spans),
            "--output",
            str(tmp_path / "o.jsonl"),
            *_columns(),
        )
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "Input line 2: text_fingerprint does not match the text" in err
    _assert_no_content(err)
    assert not (tmp_path / "o.jsonl").exists()


def test_review_report_is_private_offline_and_marked_phi(corpus, monkeypatch, capsys):
    tmp_path, notes = corpus
    spans = tmp_path / "spans.jsonl"
    _run(
        monkeypatch,
        capsys,
        "batch",
        "--detect-only",
        "--input",
        str(notes),
        "--output",
        str(spans),
        *_columns(),
    )
    report = tmp_path / "report.html"
    out = _run(
        monkeypatch,
        capsys,
        "review",
        "--input",
        str(notes),
        "--spans",
        str(spans),
        "--output",
        str(report),
        *_columns(),
    )
    assert "WARNING" in out.err and "protected health information" in out.err
    _assert_no_content(out.err)
    if os.name == "posix":
        assert _mode(report) == 0o600
    page = report.read_text(encoding="utf-8")
    assert "protected health information" in page
    assert "Content-Security-Policy" in page and "default-src 'none'" in page
    lowered = page.lower()
    for external in (
        "http:",
        "https:",
        "//",
        "src=",
        "<link",
        "@import",
        "<iframe",
        "fetch(",
        "xmlhttprequest",
        "websocket",
    ):
        assert external not in lowered, external
    assert not re.search(r"(?<![a-z])url\(", lowered)
    # Both panes and the per-label counts are present.
    assert 'data-label="NAME"' in page and "[NAME]" in page
    assert page.count('<section class="note"') == 2
    assert 'href="#note-1"' in page


def test_review_report_escapes_note_text():
    hostile = "Mara Calder </script><script>alert(1)</script> <img src=x onerror=alert(2)>\r\n"
    entities = [{"start": 0, "end": 11, "label": "NAME"}]
    result = Deidentifier.for_replacement().replace(
        hostile, entities, replacement={"pattern": "[{label}]"}
    )
    note = {"note_id": 'n"<b>', "text": hostile, "patient_id": None}
    record = {"format": SPAN_FORMAT, "note_id": note["note_id"], "entities": entities}
    page = build_report([(note, record)], [result])
    assert page.count("<script") == 2  # the data block and the inline script
    assert "<img" not in page and "<b>" not in page
    data = re.search(
        r'<script type="application/json" id="review-data">(.*?)</script>', page, re.S
    )[1]
    assert json.loads(data)["notes"][0]["text"] == hostile
    # The CSP hashes are those of the inline style and script.
    import base64

    for kind, body in (
        ("style", r"<style>(.*?)</style>"),
        ("script", r"<script>(.*?)</script>"),
    ):
        source = re.search(body, page, re.S)[1]
        digest = base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        assert f"{kind}-src 'sha256-{digest}'" in page


# Review findings: policy-added AGE parity, the label picker, the in-page
# editor's offsets, and fingerprints across a surrogate key rotation.


class EmptyDetector:
    def detect_batch(self, texts):
        return [[] for _ in texts]


@pytest.mark.parametrize("ages", ["over_89", "all"])
def test_policy_added_ages_keep_detect_replace_parity(ages):
    text = "Age 45. Badge AGE-77. Mother age 93."
    policy = {
        "ages": ages,
        "custom_entities": [{"label": "AGE", "pattern": r"\b45\b"}],
        "blocklist": ["AGE-77"],
    }
    deid = Deidentifier(secret=SECRET, detector=EmptyDetector())
    detected = deid.detect(text, policy=policy)
    assert [e["source"] for e in detected["entities"]] == [
        "policy:custom",
        "policy:blocklist",
    ]
    for options in ({"mode": "synthetic", "patient_id": "patient-a"}, {}):
        expected = deid.deidentify(text, policy=policy, **options)
        result = deid.replace(
            text, detected["entities"], policy={"ages": ages}, **options
        )
        assert _comparable(result) == _comparable(expected)
    # Content a policy rule selected never reappears.
    assert result["text"] == "Age [AGE]. Badge [ID]. Mother age 93."


def test_ages_policy_spares_explicitly_chosen_ages():
    text = "Age 45. Sister age 52. Brother age 61."
    entities = [
        {"start": 4, "end": 6, "label": "AGE", "source": "model"},
        {"start": 19, "end": 21, "label": "AGE", "source": "review"},
        {"start": 35, "end": 37, "label": "AGE", "source": "site_dictionary:ages"},
    ]
    result = Deidentifier.for_replacement().replace(text, entities)
    # The detector's young age stays, as detection would leave it; the
    # reviewer's and the dictionary's choices are replaced.
    assert result["text"] == "Age 45. Sister age [AGE]. Brother age [AGE]."
    assert {e["source"] for e in result["entities"]} == {"supplied"}


def test_review_label_picker_offers_every_supported_label():
    text = "Seen by Mara Calder."
    entities = [{"start": 8, "end": 19, "label": "NAME"}]
    custom = [{"start": 0, "end": 4, "label": "BADGE"}]
    result = Deidentifier.for_replacement().replace(
        text, entities + custom, replacement={"pattern": "[{label}]"}
    )
    note = {"note_id": "n1", "text": text, "patient_id": None}
    record = {"format": SPAN_FORMAT, "note_id": "n1", "entities": entities + custom}
    page = build_report([(note, record)], [result])
    picker = re.search(r'<select id="label">(.*?)</select>', page, re.S)[1]
    options = re.findall(r'<option value="([A-Z0-9_]+)">', picker)
    from outis.policy import KNOWN_LABELS

    # No EMAIL, MRN or SSN was detected, yet each can still be added.
    assert set(options) == KNOWN_LABELS | {"BADGE"}
    assert options[:3] == ["NAME", "DATE", "ADDRESS"]
    assert options[-1] == "BADGE"


NODE = shutil.which("node")


def _run_model(text, entities, steps, *, failed=None):
    """Run the report's editing model under Node, as the page would."""
    harness = r"""
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const note = {
  text: input.text, entities: input.entities, record: input.record, failed: input.failed,
};
const log = [];
function check() {
  const runs = ReviewModel.runs(note);
  let at = 0;
  for (const run of runs) {
    if (run.start !== at) throw new Error("runs are not contiguous");
    at = run.end;
  }
  if (runs.map((run) => run.text).join("") !== note.text) {
    throw new Error("runs do not show the text exactly once");
  }
  return runs;
}
// Select `needle` in the displayed text. Each boundary is measured from the
// run that contains it, as the page's boundary() does.
function select(needle) {
  const runs = check();
  const shown = runs.map((run) => run.text).join("");
  const from = shown.indexOf(needle);
  if (from < 0) throw new Error("selection not displayed");
  const point = (index, atEnd) => {
    let seen = 0;
    for (const run of runs) {
      const next = seen + run.text.length;
      if (index < next || (atEnd && index === next)) {
        return ReviewModel.offset(run.start, run.text.slice(0, index - seen));
      }
      seen = next;
    }
    return ReviewModel.offset(runs[runs.length - 1].end, "");
  };
  return [point(from, false), point(from + needle.length, true)];
}
for (const step of input.steps) {
  if (step[0] === "toggle") {
    log.push(ReviewModel.toggle(note, step[1]));
  } else {
    const [start, end] = select(step[1]);
    log.push(ReviewModel.add(note, start, end, step[2]));
  }
  check();
}
process.stdout.write(JSON.stringify({ log, exported: ReviewModel.exportLines([note]) }));
"""
    payload = {
        "text": text,
        "entities": entities,
        "record": {"format": SPAN_FORMAT, "note_id": "note-1"},
        "steps": steps,
        "failed": failed,
    }
    completed = subprocess.run(
        [NODE, "-e", MODEL_SCRIPT + harness],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_review_editor_multi_edit_round_trip(tmp_path):
    # An astral character before the edits makes UTF-16 and code point
    # offsets differ.
    text = "Mara Calder \U0001f642 called Nora with a long follow-up."
    entities = [
        {"start": 0, "end": 11, "label": "NAME", "score": 0.9, "source": "model"}
    ]
    outcome = _run_model(
        text,
        entities,
        [
            ["toggle", 0],  # remove "Mara Calder"
            ["add", "Mara", "NAME"],  # replace it with a shorter span
            ["add", "Nora", "NAME"],  # then select later text
            ["add", "long", "ID"],
            ["toggle", 2],  # remove the added "long" again
        ],
    )
    assert outcome["log"] == [
        ["Span marked for removal.", True],
        None,
        None,
        None,
        ["Removed an added span.", True],
    ]
    record = json.loads(outcome["exported"])
    nora = text.index("Nora")
    assert record["entities"] == [
        {"start": 0, "end": 4, "label": "NAME", "source": "review"},
        {"start": nora, "end": nora + 4, "label": "NAME", "source": "review"},
    ]
    assert record["edited"] is True
    path = tmp_path / "spans.reviewed.jsonl"
    path.write_text(outcome["exported"], encoding="utf-8")
    applied = read_span_file(path)["note-1"]
    result = Deidentifier.for_replacement().replace(text, applied["entities"])
    assert (
        result["text"]
        == "[NAME] Calder \U0001f642 called [NAME] with a long follow-up."
    )


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_review_editor_refuses_overlaps_and_hidden_restores():
    text = "Mara Calder and Nora Vale."
    entities = [{"start": 0, "end": 11, "label": "NAME"}]
    outcome = _run_model(
        text,
        entities,
        [
            ["add", "Calder and", "NAME"],  # overlaps an active span
            ["toggle", 0],
            ["add", "Calder", "NAME"],  # drops the removed span it covers
            ["toggle", 0],  # the added span: deleted
        ],
    )
    assert outcome["log"] == [
        "The selection overlaps a span; remove that span first.",
        ["Span marked for removal.", True],
        None,
        ["Removed an added span.", True],
    ]
    assert json.loads(outcome["exported"])["entities"] == []


def _rotation_env(monkeypatch, secret, fingerprint=None):
    monkeypatch.setenv("OUTIS_SECRET", secret.decode())
    if fingerprint is None:
        monkeypatch.delenv("OUTIS_FINGERPRINT_SECRET", raising=False)
    else:
        monkeypatch.setenv("OUTIS_FINGERPRINT_SECRET", fingerprint.decode())


def test_cli_reapplies_fingerprinted_spans_after_key_rotation(
    corpus, monkeypatch, capsys
):
    tmp_path, notes = corpus
    spans = tmp_path / "spans.jsonl"
    _rotation_env(monkeypatch, SECRET)
    _run(monkeypatch, capsys, "batch", "--detect-only", "--input", str(notes),
         "--output", str(spans), *_columns())  # fmt: skip
    assert all(r.get("text_fingerprint") for r in read_span_file(spans).values())
    out = tmp_path / "out.jsonl"
    replace = ("spans", "replace", "--input", str(notes), "--spans", str(spans),
               "--output", str(out), "--mode", "synthetic", *_columns())  # fmt: skip

    # Rotated surrogate key, fingerprints still under the old one: explained.
    _rotation_env(monkeypatch, OTHER_SECRET)
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, *replace)
    err = capsys.readouterr().err
    assert "text_fingerprint does not match the text" in err
    assert "OUTIS_FINGERPRINT_SECRET" in err and "--skip-fingerprint-check" in err
    _assert_no_content(err)

    # Keeping the detection key for fingerprints only verifies and re-keys.
    _rotation_env(monkeypatch, OTHER_SECRET, fingerprint=SECRET)
    _run(monkeypatch, capsys, *replace)
    rotated = [json.loads(line)["text"] for line in out.read_text().splitlines()]
    expected = [
        Deidentifier(secret=OTHER_SECRET, detector=Detector()).deidentify(
            note["body"], mode="synthetic", patient_id=note["patient"]
        )["text"]
        for note in NOTES
    ]
    assert rotated == expected

    # Or skip the check explicitly, which is reported.
    _rotation_env(monkeypatch, OTHER_SECRET)
    skipped = _run(monkeypatch, capsys, *replace, "--skip-fingerprint-check")
    assert "not checked" in skipped.err
    assert [json.loads(line)["text"] for line in out.read_text().splitlines()] == (
        expected
    )

    # Without any key, fingerprinted records need the explicit skip.
    monkeypatch.delenv("OUTIS_SECRET")
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, "spans", "replace", "--input", str(notes),
             "--spans", str(spans), "--output", str(out), *_columns())  # fmt: skip
    assert "--skip-fingerprint-check" in capsys.readouterr().err


def test_separate_fingerprint_key_survives_surrogate_rotation(
    corpus, monkeypatch, capsys
):
    tmp_path, notes = corpus
    spans, out = tmp_path / "spans.jsonl", tmp_path / "out.jsonl"
    fingerprint_key = b"outis-fingerprint-only-test-key-0123456789"
    _rotation_env(monkeypatch, SECRET, fingerprint=fingerprint_key)
    _run(monkeypatch, capsys, "batch", "--detect-only", "--input", str(notes),
         "--output", str(spans), *_columns())  # fmt: skip
    record = read_span_file(spans)["note-1"]
    assert record["text_fingerprint"] == text_fingerprint(fingerprint_key, TEXT)
    for secret in (SECRET, OTHER_SECRET):
        _rotation_env(monkeypatch, secret, fingerprint=fingerprint_key)
        _run(monkeypatch, capsys, "spans", "replace", "--input", str(notes),
             "--spans", str(spans), "--output", str(out), "--mode", "synthetic",
             *_columns())  # fmt: skip
    # The library and API take the same separate key.
    deid = Deidentifier.for_replacement(
        secret=OTHER_SECRET, fingerprint_secret=fingerprint_key
    )
    deid.replace(TEXT, [], text_fingerprint=record["text_fingerprint"])
    with pytest.raises(SpanError):
        Deidentifier.for_replacement(secret=fingerprint_key).replace(
            TEXT, [], text_fingerprint=text_fingerprint(OTHER_SECRET, TEXT)
        )


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_review_editor_refuses_edits_to_notes_whose_detection_failed():
    outcome = _run_model(
        "Mara Calder was seen.", [], [["add", "Mara", "NAME"]], failed="out_of_memory"
    )
    assert outcome["log"] == [
        "Detection failed for this note; detect it again before reviewing."
    ]
    assert json.loads(outcome["exported"])["entities"] == []


def test_review_shows_failed_detections_and_replace_refuses_them(
    corpus, monkeypatch, capsys
):
    tmp_path, notes = corpus
    spans = tmp_path / "spans.jsonl"
    records = [
        span_record(1, service().detect(TEXT)),
        {"format": SPAN_FORMAT, "note_id": 2, "entities": [], "error": "out_of_memory"},
    ]
    spans.write_text("".join(json.dumps(r) + "\n" for r in records))
    notes.write_text(
        json.dumps({"note_id": 1, "text": TEXT})
        + "\n"
        + json.dumps({"note_id": 2, "text": SECOND})
        + "\n"
    )
    report = tmp_path / "report.html"
    _run(monkeypatch, capsys, "review", "--input", str(notes), "--spans", str(spans),
         "--output", str(report))  # fmt: skip
    page = report.read_text(encoding="utf-8")
    assert "Detection failed for this note (out_of_memory)" in page
    assert '<td class="n">failed</td>' in page
    data = json.loads(re.search(r'id="review-data">(.*?)</script>', page, re.S)[1])
    assert data["notes"][1]["failed"] == "out_of_memory"
    assert data["notes"][1]["record"]["error"] == "out_of_memory"
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, "spans", "replace", "--input", str(notes),
             "--spans", str(spans), "--output", str(tmp_path / "o.jsonl"))  # fmt: skip
    err = capsys.readouterr().err
    assert "Input line 2: detection failed for this note (out_of_memory)" in err
