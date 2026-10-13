"""``outis batch --detect-only`` and ``--spans``: detect a corpus once, replace often.

Scripted detectors stand in for the model. Every value here is fictional.
"""

import json
import os
import stat

import pytest
from test_batch import (
    NOTES,
    PHI,
    SECRET,
    FakeDetector,
    config,
    install,
    read_jsonl,
    run,
    write_jsonl,
)

from outis import batch
from outis.batch import BatchConfig, BatchError
from outis.service import Deidentifier
from outis.spans import SPAN_FORMAT, iter_span_file, text_fingerprint

OTHER_SECRET = "a-different-test-only-secret-never-use-0987654321"
FINGERPRINT_KEY = "a-fingerprint-only-test-key-never-use-1357924680"


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    monkeypatch.delenv("OUTIS_FINGERPRINT_SECRET", raising=False)


@pytest.fixture
def detector(monkeypatch):
    return install(monkeypatch, FakeDetector())


def detect(tmp_path, name="spans.jsonl", **fields):
    cfg = config(tmp_path, name=name, detect_only=True, **fields)
    return cfg, run(cfg)


def replace(tmp_path, spans, name="replaced.jsonl", **fields):
    cfg = config(tmp_path, name=name, spans=spans, **fields)
    return cfg, run(cfg)


def no_model(monkeypatch):
    def refuse(config):
        raise AssertionError("--spans must not load the model")

    monkeypatch.setattr(batch, "_load_detector", refuse)


def _comparable(row):
    return {
        **row,
        "entities": [
            {key: value for key, value in entity.items() if key != "source"}
            for entity in row["entities"]
        ],
    }


POLICIES = [
    (None, "redact", None),
    (
        {
            "policy": {
                "custom_entities": [{"label": "LAB", "pattern": r"Creatinine \d\.\d"}],
                "ages": "all",
            },
            "replacement": {"type": "marker", "per_type": {"DATE": "mask"}},
        },
        "redact",
        None,
    ),
    ({"replacement": {"type": "synthetic", "style": "realistic"}}, "synthetic", None),
    (
        {
            "replacement": {
                "type": "synthetic",
                "scope": "project",
                "entity_ids": "keyed",
            }
        },
        "synthetic",
        "trial-a",
    ),
]


@pytest.mark.parametrize("policy, mode, project", POLICIES)
def test_detect_once_then_replace_matches_a_full_run(
    tmp_path, monkeypatch, detector, policy, mode, project
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    fields = {"mode": mode, "patient_column": "patient_id", "batch_size": 3}
    if project:
        fields["project_id"] = project
    if policy is not None:
        (tmp_path / "policy.json").write_text(json.dumps(policy))
        fields["policy_file"] = tmp_path / "policy.json"
    full = config(tmp_path, name="full.jsonl", **fields)
    run(full)
    spans_cfg, summary = detect(tmp_path, **fields)
    assert summary["failed"] == 0
    no_model(monkeypatch)
    replaced, _ = replace(tmp_path, spans_cfg.output, **fields)
    rows, expected = read_jsonl(replaced.output), read_jsonl(full.output)
    assert [_comparable(r) for r in rows] == [_comparable(r) for r in expected]
    assert {e["source"] for r in rows for e in r["entities"]} <= {"supplied"}


def test_span_file_holds_offsets_not_text_and_is_owner_only(
    tmp_path, monkeypatch, detector
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    cfg, summary = detect(tmp_path, patient_column="patient_id")
    records = [record for _, record in iter_span_file(cfg.output)]
    assert [r["note_id"] for r in records] == [n["note_id"] for n in NOTES]
    assert [r["patient_id"] for r in records] == [n["patient_id"] for n in NOTES]
    assert {r["format"] for r in records} == {SPAN_FORMAT}
    assert records[0]["entities"][0] == {
        "start": 0,
        "end": 11,
        "label": "NAME",
        "score": 1.0,
        "source": "model",
    }
    for record, note in zip(records, NOTES):
        assert record["text_fingerprint"] == text_fingerprint(SECRET, note["text"])
        assert record["model_version"] == "injected-detector"
    content = cfg.output.read_text(encoding="utf-8")
    assert not any(value in content for value in PHI)
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(cfg.output).st_mode) == 0o600
    assert summary["output"] == {"ok": len(NOTES), "failed": 0, "failure_codes": {}}


def test_detect_only_without_a_key_writes_no_fingerprints(tmp_path, detector):
    import io

    stderr = io.StringIO()
    cfg = config(tmp_path, name="spans.jsonl", detect_only=True)
    run(cfg, stderr=stderr)
    assert "no text fingerprint" in stderr.getvalue()
    assert all("text_fingerprint" not in r for _, r in iter_span_file(cfg.output))


def test_failed_detections_are_recorded_and_fail_their_rows(
    tmp_path, monkeypatch, detector
):
    original = detector.detect_batch

    def flaky(texts):
        if any("Owen Pell" in text for text in texts):
            raise RuntimeError("tokenizer failure")
        return original(texts)

    detector.detect_batch = flaky
    rows = [*NOTES, {"note_id": "long", "patient_id": "p9", "text": "a" * 100_001}]
    cfg, summary = detect(tmp_path, rows=rows)
    records = {r["note_id"]: r for _, r in iter_span_file(cfg.output)}
    assert records["n2"] == {
        "format": SPAN_FORMAT,
        "note_id": "n2",
        "entities": [],
        "error": "detector_error",
    }
    assert records["long"]["error"] == "text_too_long"
    assert summary["failure_codes"] == {"detector_error": 1, "text_too_long": 1}
    no_model(monkeypatch)
    replaced, summary = replace(tmp_path, cfg.output)
    errors = {row["note_id"]: row["error"] for row in read_jsonl(replaced.output)}
    assert errors["n2"] == "detector_error" and errors["long"] == "text_too_long"
    assert summary["output"]["failed"] == 2


def test_reapplying_under_another_project_changes_only_surrogates(
    tmp_path, monkeypatch, detector
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    spans_cfg, _ = detect(tmp_path, patient_column="patient_id")
    no_model(monkeypatch)
    (tmp_path / "policy.json").write_text(
        json.dumps({"replacement": {"type": "synthetic", "scope": "project"}})
    )
    outputs = []
    for project in ("trial-a", "trial-b"):
        cfg, _ = replace(
            tmp_path,
            spans_cfg.output,
            name=f"{project}.jsonl",
            mode="synthetic",
            patient_column="patient_id",
            policy_file=tmp_path / "policy.json",
            project_id=project,
        )
        outputs.append(read_jsonl(cfg.output))
    a, b = outputs
    assert [r["text"] for r in a] != [r["text"] for r in b]
    for x, y in zip(a, b):
        assert [(e["start"], e["end"], e["label"]) for e in x["entities"]] == [
            (e["start"], e["end"], e["label"]) for e in y["entities"]
        ]


def test_reviewed_span_files_apply(tmp_path, monkeypatch, detector):
    spans_cfg, _ = detect(tmp_path)
    records = [record for _, record in iter_span_file(spans_cfg.output)]
    # A reviewer removed the phone number and added the creatinine value.
    records[0]["entities"] = [
        e for e in records[0]["entities"] if e["label"] != "PHONE"
    ]
    start = NOTES[2]["text"].index("1.2")
    records[2]["entities"].append(
        {"start": start, "end": start + 3, "label": "ID", "source": "review"}
    )
    records[0]["edited"] = records[2]["edited"] = True
    reviewed = write_jsonl(tmp_path / "reviewed.jsonl", records)
    no_model(monkeypatch)
    cfg, _ = replace(tmp_path, reviewed)
    rows = read_jsonl(cfg.output)
    assert rows[0]["text"].endswith("Call 617-555-0185.")
    assert rows[2]["text"] == "Follow-up for [NAME] on [DATE]. Creatinine [ID] mg/dL."


def test_detect_only_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    cfg = config(tmp_path, name="resumed.jsonl", batch_size=2, detect_only=True)
    fake = install(monkeypatch, FakeDetector(interrupt_on_call=3))
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    assert len(read_jsonl(cfg.output)) == 4
    fake.interrupt_on_call = None
    resumed = run(BatchConfig(**{**cfg.__dict__, "resume": True}))
    assert (resumed["skipped"], resumed["processed"]) == (4, 3)
    clean = config(tmp_path, name="clean.jsonl", batch_size=2, detect_only=True)
    run(clean)
    assert cfg.output.read_bytes() == clean.output.read_bytes()
    # A detect-only checkpoint cannot be resumed as a full run.
    with pytest.raises(BatchError, match="changed: .*stage"):
        run(BatchConfig(**{**cfg.__dict__, "resume": True, "detect_only": False}))


def test_replace_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch, detector):
    spans_cfg, _ = detect(tmp_path)
    no_model(monkeypatch)
    cfg = config(tmp_path, name="resumed.jsonl", batch_size=2, spans=spans_cfg.output)
    calls = {"n": 0}
    original = Deidentifier.replace

    def interrupt(self, *args, **kwargs):
        calls["n"] += 1
        # Call 1 is the up-front options check; call 6 is in the third batch.
        if calls["n"] == 6:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Deidentifier, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    assert len(read_jsonl(cfg.output)) == 4
    monkeypatch.setattr(Deidentifier, "replace", original)
    resumed = run(BatchConfig(**{**cfg.__dict__, "resume": True}))
    assert (resumed["skipped"], resumed["processed"]) == (4, 3)
    clean, _ = replace(tmp_path, spans_cfg.output, name="clean.jsonl", batch_size=2)
    assert cfg.output.read_bytes() == clean.output.read_bytes()
    # A changed span file cannot resume the run.
    spans_cfg.output.write_text(spans_cfg.output.read_text() + "\n")
    with pytest.raises(BatchError, match="changed: spans_sha256"):
        run(BatchConfig(**{**cfg.__dict__, "resume": True}))


def _rewrite(path, change):
    records = [record for _, record in iter_span_file(path)]
    change(records)
    write_jsonl(path, records)


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda r: r.reverse(),
            "Span file line 1: note_id differs from input line 1; records must follow",
        ),
        (lambda r: r.pop(), "fewer records than the input has rows"),
        (
            lambda r: r.append({**r[0], "note_id": "extra"}),
            "Span file line 8: no matching input row",
        ),
        (
            lambda r: r[1]["entities"].append(dict(r[1]["entities"][0])),
            "Span file line 2: Entity 4: overlaps another entity",
        ),
        (
            lambda r: r[0]["entities"][0].update(end=10_000),
            "Span file line 1: Entity 0: offsets are outside the text",
        ),
        (
            lambda r: r[2].update(patient_id="p9"),
            "Span file line 3: patient_id differs from the input",
        ),
        (lambda r: r[3].update(text="Mara"), "Span file line 4: unsupported field"),
    ],
)
def test_bad_span_files_fail_before_any_output(
    tmp_path, monkeypatch, detector, change, message
):
    spans_cfg, _ = detect(tmp_path, patient_column="patient_id")
    _rewrite(spans_cfg.output, change)
    no_model(monkeypatch)
    with pytest.raises(BatchError) as caught:
        replace(tmp_path, spans_cfg.output, patient_column="patient_id")
    assert message in str(caught.value)
    assert not any(value in str(caught.value) for value in PHI)
    assert not (tmp_path / "replaced.jsonl").exists()


def test_fingerprints_catch_edited_input_and_survive_rotation(
    tmp_path, monkeypatch, detector
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    spans_cfg, _ = detect(tmp_path, patient_column="patient_id")
    no_model(monkeypatch)
    fields = {"mode": "synthetic", "patient_column": "patient_id"}

    # The surrogate key rotated; fingerprints were made under the old one.
    monkeypatch.setenv("OUTIS_SECRET", OTHER_SECRET)
    with pytest.raises(BatchError, match="line 1: text_fingerprint does not match"):
        replace(tmp_path, spans_cfg.output, **fields)
    monkeypatch.setenv("OUTIS_FINGERPRINT_SECRET", SECRET)
    cfg, _ = replace(tmp_path, spans_cfg.output, name="rotated.jsonl", **fields)
    rotated = Deidentifier(secret=OTHER_SECRET, detector=FakeDetector())
    assert [r["text"] for r in read_jsonl(cfg.output)] == [
        rotated.deidentify(n["text"], mode="synthetic", patient_id=n["patient_id"])[
            "text"
        ]
        for n in NOTES
    ]

    # Edited input text is caught, unless the check is explicitly skipped.
    edited = [*NOTES[:4], {**NOTES[4], "text": NOTES[4]["text"] + " "}, *NOTES[5:]]
    write_jsonl(tmp_path / "notes.jsonl", edited)
    with pytest.raises(BatchError, match="line 5: text_fingerprint does not match"):
        replace(tmp_path, spans_cfg.output, name="edited.jsonl", **fields)
    cfg, _ = replace(
        tmp_path,
        spans_cfg.output,
        name="skipped.jsonl",
        skip_fingerprint_check=True,
        **fields,
    )
    assert len(read_jsonl(cfg.output)) == len(NOTES)

    # Fingerprinted records with no key at all need the explicit skip.
    monkeypatch.delenv("OUTIS_SECRET")
    monkeypatch.delenv("OUTIS_FINGERPRINT_SECRET")
    write_jsonl(tmp_path / "notes.jsonl", NOTES)
    with pytest.raises(BatchError, match="--skip-fingerprint-check"):
        replace(tmp_path, spans_cfg.output, name="nokey.jsonl")


def test_separate_fingerprint_key(tmp_path, monkeypatch, detector):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    monkeypatch.setenv("OUTIS_FINGERPRINT_SECRET", FINGERPRINT_KEY)
    spans_cfg, _ = detect(tmp_path)
    first = next(iter_span_file(spans_cfg.output))[1]
    assert first["text_fingerprint"] == text_fingerprint(
        FINGERPRINT_KEY, NOTES[0]["text"]
    )
    no_model(monkeypatch)
    monkeypatch.setenv("OUTIS_SECRET", OTHER_SECRET)
    replace(tmp_path, spans_cfg.output)
    monkeypatch.setenv("OUTIS_FINGERPRINT_SECRET", "too-short")
    with pytest.raises(BatchError, match="at least 32 bytes"):
        detect(tmp_path, name="short.jsonl")


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"detect_only": True, "spans": "spans.jsonl"}, "cannot be combined"),
        ({"skip_fingerprint_check": True}, "needs --spans"),
        ({"detect_only": True, "pass_through": ["site"]}, "--pass-through"),
        ({"detect_only": True, "include_originals": True}, "--include-originals"),
        ({"detect_only": True, "format": "parquet"}, "JSONL span file"),
        ({"spans": "missing.jsonl"}, "Span file not found"),
    ],
)
def test_stage_options_are_checked(tmp_path, detector, fields, message):
    if isinstance(fields.get("spans"), str):
        fields["spans"] = tmp_path / fields["spans"]
    with pytest.raises(BatchError, match=message):
        run(config(tmp_path, **fields))


def test_span_file_cannot_be_overwritten(tmp_path, detector):
    spans_cfg, _ = detect(tmp_path)
    with pytest.raises(BatchError, match="overwrite the span file"):
        run(
            BatchConfig(
                input=tmp_path / "notes.jsonl",
                output=spans_cfg.output,
                spans=spans_cfg.output,
                resume=True,
            )
        )


def test_integer_note_ids_round_trip(tmp_path, monkeypatch, detector):
    rows = [{**note, "note_id": index} for index, note in enumerate(NOTES, 1)]
    spans_cfg, _ = detect(tmp_path, rows=rows)
    assert [r["note_id"] for _, r in iter_span_file(spans_cfg.output)] == list(
        range(1, len(NOTES) + 1)
    )
    no_model(monkeypatch)
    cfg, _ = replace(tmp_path, spans_cfg.output)
    assert [r["note_id"] for r in read_jsonl(cfg.output)] == list(
        range(1, len(NOTES) + 1)
    )


def test_parquet_input_and_output(tmp_path, monkeypatch, detector):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "notes.parquet"
    pq.write_table(pa.Table.from_pylist(NOTES), source)
    spans_cfg = BatchConfig(
        input=source, output=tmp_path / "spans.jsonl", detect_only=True
    )
    run(spans_cfg)
    full = BatchConfig(input=source, output=tmp_path / "full.parquet")
    run(full)
    no_model(monkeypatch)
    replaced = BatchConfig(
        input=source, output=tmp_path / "replaced.parquet", spans=spans_cfg.output
    )
    run(replaced)
    got = pq.read_table(replaced.output).to_pylist()
    expected = pq.read_table(full.output).to_pylist()
    assert [r["text"] for r in got] == [r["text"] for r in expected]


def test_cli_flags(tmp_path, monkeypatch, detector, capsys):
    import sys

    from outis import cli

    write_jsonl(tmp_path / "notes.jsonl", NOTES)
    spans = tmp_path / "spans.jsonl"

    def main(*args):
        monkeypatch.setattr(sys, "argv", ["outis", "batch", *args])
        with pytest.raises(SystemExit) as caught:
            cli.main()
        return caught.value.code, capsys.readouterr()

    notes = str(tmp_path / "notes.jsonl")
    code, out = main("--input", notes, "--output", str(spans), "--detect-only")
    assert code == 0 and spans.exists()
    assert not any(value in out.err for value in PHI)
    no_model(monkeypatch)
    out_path = str(tmp_path / "out.jsonl")
    code, _ = main("--input", notes, "--output", out_path, "--spans", str(spans))
    assert code == 0
    code, out = main(
        "--input", notes, "--output", str(tmp_path / "x"), "--spans", str(spans),
        "--detect-only",
    )  # fmt: skip
    assert code == 2 and "not allowed with argument" in out.err
