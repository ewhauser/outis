"""``outis batch``: corpus-scale runs over JSONL and Parquet note tables.

The encoder is replaced by scripted detectors, so no model is downloaded and
no GPU is needed; out-of-memory errors are raised by the fake detector. All
names, dates and identifiers below are fictional.
"""

import functools
import io
import json
import re
import sys
from datetime import date, datetime

import pytest

from outis import batch, cli
from outis.batch import BatchConfig, BatchError
from outis.service import Deidentifier
from outis.types import Span

torch = pytest.importorskip("torch")

SECRET = "test-only-secret-never-use-in-production-12345"
NAMES = ("Mara Calder", "Ivo Strand", "Owen Pell", "Tamsin Vey")
PATTERN = re.compile(
    "|".join(NAMES) + r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{3}-\d{3}-\d{4}\b|\bMRN \d{7}\b"
)


def label(value):
    if value in NAMES:
        return "NAME"
    if value.startswith("MRN"):
        return "MRN"
    return "DATE" if value[4] == "-" else "PHONE"


class FakeDetector:
    """Scripted detector that records each call's note count and window batch."""

    def __init__(self, batch_size=8, *, oom=None, interrupt_on_call=None):
        self.batch_size = batch_size
        self.calls = []
        self.oom = oom
        self.interrupt_on_call = interrupt_on_call

    def detect_batch(self, texts):
        self.calls.append((len(texts), self.batch_size))
        if self.interrupt_on_call == len(self.calls):
            raise KeyboardInterrupt
        if self.oom is not None and self.oom(texts, self.batch_size):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return [
            [Span(m.start(), m.end(), label(m[0])) for m in PATTERN.finditer(text)]
            for text in texts
        ]


NOTES = [
    {
        "note_id": "n1",
        "patient_id": "p1",
        "site": "north",
        "text": "Mara Calder admitted 2024-02-28, discharged 2024-03-02. Call 617-555-0185.",
    },
    {
        "note_id": "n2",
        "patient_id": "p2",
        "site": "south",
        "text": "Ivo Strand (MRN 4482910) seen 2025-03-04 by Owen Pell.",
    },
    {
        "note_id": "n3",
        "patient_id": "p1",
        "site": "north",
        "text": "Follow-up for Mara Calder on 2024-03-10. Creatinine 1.2 mg/dL.",
    },
    {"note_id": "n4", "patient_id": "p3", "site": None, "text": "No identifiers here."},
    {
        "note_id": "n5",
        "patient_id": "p2",
        "site": "south",
        "text": "Tamsin Vey called 555-201-7788 on 2025-04-01 about Ivo Strand.",
    },
    {"note_id": "n6", "patient_id": "p3", "site": "east", "text": ""},
    {
        "note_id": "n7",
        "patient_id": "p1",
        "site": "north",
        "text": "Seen 2024-12-31 and 2025-01-02 for Mara Calder.",
    },
]
PHI = [
    *NAMES,
    "2024-02-28",
    "2025-03-04",
    "617-555-0185",
    "555-201-7788",
    "4482910",
]


@pytest.fixture(autouse=True)
def no_secret(monkeypatch):
    monkeypatch.delenv("OUTIS_SECRET", raising=False)


def install(monkeypatch, fake):
    """Load ``fake`` as the model, applying --batch-size as load_detector does."""

    def load(config):
        if config.batch_size is not None:
            fake.batch_size = config.batch_size
        return fake

    monkeypatch.setattr(batch, "_load_detector", load)
    return fake


@pytest.fixture
def detector(monkeypatch):
    """Install a FakeDetector as the model; tests may replace its attributes."""
    return install(monkeypatch, FakeDetector())


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def config(tmp_path, name="out.jsonl", rows=NOTES, **fields):
    source = tmp_path / "notes.jsonl"
    if not source.exists():
        write_jsonl(source, rows)
    return BatchConfig(input=source, output=tmp_path / name, **fields)


def run(cfg, stderr=None):
    return batch.run(cfg, stderr=stderr or io.StringIO(), progress_interval=0)


def test_jsonl_round_trip_keeps_ids_order_and_pass_through(tmp_path, detector):
    cfg = config(tmp_path, patient_column="patient_id", pass_through=["site"])
    summary = run(cfg)
    rows = read_jsonl(cfg.output)
    assert [row["note_id"] for row in rows] == [note["note_id"] for note in NOTES]
    assert [(r["patient_id"], r["site"]) for r in rows] == [
        (n["patient_id"], n["site"]) for n in NOTES
    ]
    assert rows[0]["text"] == "[NAME] admitted [DATE], discharged [DATE]. Call [PHONE]."
    assert rows[0]["entities"][0] == {
        "start": 0,
        "end": 11,
        "label": "NAME",
        "score": 1.0,
        "source": "model",
    }
    assert {row["status"] for row in rows} == {"ok"}
    assert all(row["error"] is None and row["warnings"] == [] for row in rows)
    output = cfg.output.read_text(encoding="utf-8")
    assert not any(value in output for value in PHI)
    assert summary["processed"] == len(NOTES) and summary["failed"] == 0
    assert summary["output"] == {"ok": len(NOTES), "failed": 0, "failure_codes": {}}
    assert summary["batch_size"] == 8 and detector.calls == [(len(NOTES), 8)]


def test_originals_are_copied_only_when_requested(tmp_path, detector):
    cfg = config(tmp_path, include_originals=True)
    run(cfg)
    entity = read_jsonl(cfg.output)[0]["entities"][0]
    assert entity["original"] == "Mara Calder"


def test_batches_match_the_library_per_note_output(tmp_path, detector):
    policy = {
        "policy": {
            "custom_entities": [{"label": "LAB", "pattern": r"Creatinine \d\.\d"}],
            "exclude": ["PHONE"],
        },
        "replacement": {"type": "marker", "per_type": {"DATE": "mask"}},
    }
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    cfg = config(tmp_path, batch_size=3, policy_file=tmp_path / "policy.json")
    run(cfg)
    library = Deidentifier(detector=FakeDetector())
    for note, row in zip(NOTES, read_jsonl(cfg.output)):
        expected = library.deidentify(
            note["text"], policy=policy["policy"], replacement=policy["replacement"]
        )
        assert row["text"] == expected["text"]
        assert row["entities"] == [
            {k: v for k, v in e.items() if k != "replacement"}
            for e in expected["entities"]
        ]
        assert row["warnings"] == expected["warnings"]


def test_batched_encoder_windows_match_single_note_detection(tmp_path, monkeypatch):
    from test_hf_encoder import PHONE, SENTENCE, tiny_detector

    texts = [SENTENCE, " ".join([SENTENCE] * 5), PHONE, "", f"{PHONE} {SENTENCE}"]
    rows = [{"note_id": i, "text": text} for i, text in enumerate(texts)]
    monkeypatch.setattr(
        batch, "_load_detector", lambda config: tiny_detector(batch_size=3)
    )
    cfg = config(tmp_path, rows=rows, batch_size=3)
    run(cfg)
    library = Deidentifier(detector=tiny_detector(batch_size=1))
    for text, row in zip(texts, read_jsonl(cfg.output)):
        expected = library.deidentify(text)
        assert row["text"] == expected["text"]
        assert [(e["start"], e["end"], e["label"]) for e in row["entities"]] == [
            (e["start"], e["end"], e["label"]) for e in expected["entities"]
        ]


def test_parquet_round_trip(tmp_path, detector):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [
            {
                "note_id": note["note_id"],
                "patient_id": note["patient_id"],
                "note_text": note["text"],
                "visit": index,
                "charted": datetime(2024, 1, index + 1, 8, 30),
                "unused": "never read",
            }
            for index, note in enumerate(NOTES)
        ]
    )
    pq.write_table(table, tmp_path / "notes.parquet")
    cfg = BatchConfig(
        input=tmp_path / "notes.parquet",
        output=tmp_path / "out.parquet",
        text_column="note_text",
        patient_column="patient_id",
        pass_through=["visit", "charted"],
        batch_size=2,
    )
    summary = run(cfg)
    out = pq.read_table(cfg.output)
    assert out.schema.field("note_id").type == pa.string()
    assert out.schema.field("visit").type == pa.int64()
    assert out.schema.field("charted").type == pa.string()
    assert "unused" not in out.schema.names
    rows = out.to_pylist()
    assert [row["note_id"] for row in rows] == [note["note_id"] for note in NOTES]
    assert rows[0]["note_text"].startswith("[NAME] admitted [DATE]")
    assert rows[0]["charted"] == "2024-01-01T08:30:00"
    assert rows[1]["entities"][1]["label"] == "MRN"
    assert rows[1]["entities"][1]["original"] is None
    assert summary["output"]["ok"] == len(NOTES)
    assert not (tmp_path / "out.parquet.partial.jsonl").exists()

    # The same table written as JSONL carries the same rows.
    jsonl = BatchConfig(**{**cfg.__dict__, "output": tmp_path / "out.jsonl"})
    run(jsonl)
    for row, line in zip(rows, read_jsonl(jsonl.output)):
        assert row["note_text"] == line["note_text"]
        assert [e["start"] for e in row["entities"]] == [
            e["start"] for e in line["entities"]
        ]


def test_parquet_without_the_extra_names_it(tmp_path, detector, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(BatchError, match=r"outis\[parquet\]"):
        run(config(tmp_path, name="out.parquet"))


def interrupted_then_resumed(tmp_path, monkeypatch, name, *, torn_tail):
    """Interrupt on the third detection call, then finish with --resume."""
    cfg = config(tmp_path, name=name, batch_size=2, pass_through=["site"])
    fake = FakeDetector(interrupt_on_call=3)
    install(monkeypatch, fake)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    journal = (
        cfg.output
        if name.endswith(".jsonl")
        else cfg.output.with_name(cfg.output.name + ".partial.jsonl")
    )
    assert len(read_jsonl(journal)) == 4
    if torn_tail:
        # A crash part-way through appending the next batch.
        with open(journal, "ab") as stream:
            stream.write(b'{"note_id":"n5","text":"[NA')
    with pytest.raises(BatchError, match="--resume"):
        run(cfg)
    fake.interrupt_on_call = None
    resumed = run(BatchConfig(**{**cfg.__dict__, "resume": True}))
    assert (resumed["skipped"], resumed["processed"]) == (4, 3)
    assert fake.calls[-2:] == [(2, 2), (1, 2)]
    return cfg.output


@pytest.mark.parametrize("torn_tail", [False, True])
def test_resume_after_interruption_matches_an_uninterrupted_run(
    tmp_path, monkeypatch, torn_tail
):
    resumed = interrupted_then_resumed(
        tmp_path, monkeypatch, "resumed.jsonl", torn_tail=torn_tail
    )
    install(monkeypatch, FakeDetector())
    clean = config(tmp_path, name="clean.jsonl", batch_size=2, pass_through=["site"])
    run(clean)
    assert resumed.read_bytes() == clean.output.read_bytes()
    # Resuming a complete run changes nothing.
    again = run(BatchConfig(**{**clean.__dict__, "resume": True}))
    assert again["processed"] == 0 and again["skipped"] == len(NOTES)
    assert clean.output.read_bytes() == resumed.read_bytes()


def test_parquet_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    resumed = interrupted_then_resumed(
        tmp_path, monkeypatch, "resumed.parquet", torn_tail=True
    )
    install(monkeypatch, FakeDetector())
    clean = config(tmp_path, name="clean.parquet", batch_size=2, pass_through=["site"])
    run(clean)
    assert pq.read_table(resumed).equals(pq.read_table(clean.output))


def test_resume_refuses_changed_settings(tmp_path, monkeypatch):
    fake = FakeDetector(interrupt_on_call=2)
    install(monkeypatch, fake)
    cfg = config(tmp_path, batch_size=2)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    fake.interrupt_on_call = None
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    changed = {"mode": "synthetic", "patient_column": "patient_id", "batch_size": 4}
    with pytest.raises(
        BatchError, match="changed: batch_size, mode, patient_column, secret"
    ):
        run(BatchConfig(**{**cfg.__dict__, "resume": True, **changed}))
    write_jsonl(cfg.input, NOTES[:3])
    with pytest.raises(BatchError, match="changed: input_sha256"):
        run(BatchConfig(**{**cfg.__dict__, "resume": True}))
    # With nothing to resume, --resume starts a run; an output without its
    # progress file is never appended to.
    fresh = run(config(tmp_path, name="fresh.jsonl", resume=True))
    assert (fresh["processed"], fresh["skipped"]) == (3, 0)
    (tmp_path / "fresh.jsonl.progress.json").unlink()
    with pytest.raises(BatchError, match="without its progress file"):
        run(config(tmp_path, name="fresh.jsonl", resume=True))


def test_out_of_memory_halves_the_batch_down_to_one_note(
    tmp_path, detector, monkeypatch
):
    released = []
    monkeypatch.setattr(batch, "release_memory", lambda: released.append(1))
    rows = [*NOTES, {"note_id": "huge", "text": "x " * 5000}]
    # Out of memory above three notes or two windows per pass; the huge note
    # never fits, even alone in one-window passes.
    detector.oom = lambda texts, size: (
        len(texts) > 3 or size > 2 or any(len(t) > 9000 for t in texts)
    )
    cfg = config(tmp_path, rows=rows, batch_size=8)
    summary = run(cfg)
    out = read_jsonl(cfg.output)
    assert out[-1]["status"] == "failed" and out[-1]["error"] == "out_of_memory"
    assert out[-1]["text"] is None and out[-1]["entities"] == []
    assert [row["status"] for row in out[:-1]] == ["ok"] * len(NOTES)
    assert out[0]["text"].startswith("[NAME] admitted")
    assert detector.calls[:3] == [(8, 8), (4, 4), (2, 2)]
    assert (1, 1) in detector.calls
    assert detector.batch_size == 8  # Restored for the next batch.
    assert summary["failed"] == 1 and summary["oom_retries"] == len(released) > 2
    assert summary["failure_codes"] == {"out_of_memory": 1}


def test_mps_out_of_memory_runtime_error_is_recognized():
    assert batch.is_out_of_memory(
        RuntimeError("MPS backend out of memory (MPS allocated)")
    )
    assert batch.is_out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert not batch.is_out_of_memory(RuntimeError("shape mismatch"))


def test_other_detector_errors_fail_only_their_note(tmp_path, detector):
    original = detector.detect_batch

    def flaky(texts):
        if any("Owen Pell" in text for text in texts):
            raise RuntimeError("tokenizer failure")
        return original(texts)

    detector.detect_batch = flaky
    cfg = config(tmp_path)
    summary = run(cfg)
    statuses = {row["note_id"]: row["error"] for row in read_jsonl(cfg.output)}
    assert statuses["n2"] == "detector_error"
    assert [code for code in statuses.values() if code] == ["detector_error"]
    assert summary["failure_codes"] == {"detector_error": 1}


def test_overlong_notes_fail_without_reaching_the_detector(tmp_path, detector):
    rows = [NOTES[0], {"note_id": "long", "text": "a" * 100_001}]
    cfg = config(tmp_path, rows=rows)
    run(cfg)
    assert [row["error"] for row in read_jsonl(cfg.output)] == [None, "text_too_long"]
    assert detector.calls == [(1, 8)]


def shifts(original, replaced):
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    before = [date.fromisoformat(d) for d in pattern.findall(original)]
    after = [date.fromisoformat(d) for d in pattern.findall(replaced)]
    assert len(before) == len(after)
    return {(b - a).days for a, b in zip(before, after)}


def test_synthetic_mode_shifts_each_patients_dates_consistently(
    tmp_path, detector, monkeypatch
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    cfg = config(tmp_path, mode="synthetic", patient_column="patient_id", batch_size=2)
    run(cfg)
    by_patient = {}
    for note, row in zip(NOTES, read_jsonl(cfg.output)):
        assert row["status"] == "ok"
        if PATTERN.search(note["text"]) and re.search(r"\d{4}-", note["text"]):
            by_patient.setdefault(note["patient_id"], set()).update(
                shifts(note["text"], row["text"])
            )
    # p1's dates sit in three notes spread over three batches.
    assert set(by_patient) == {"p1", "p2"}
    for found in by_patient.values():
        assert len(found) == 1 and 0 not in found
    library = Deidentifier(detector=FakeDetector(), secret=SECRET)
    expected = library.deidentify(NOTES[6]["text"], mode="synthetic", patient_id="p1")
    assert read_jsonl(cfg.output)[6]["text"] == expected["text"]


def test_synthetic_mode_requires_a_secret_and_patient_ids(
    tmp_path, detector, monkeypatch
):
    with pytest.raises(BatchError, match="--patient-column"):
        run(config(tmp_path, mode="synthetic"))
    with pytest.raises(BatchError, match="OUTIS_SECRET"):
        run(config(tmp_path, mode="synthetic", patient_column="patient_id"))
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    cfg = BatchConfig(
        input=write_jsonl(
            tmp_path / "missing.jsonl", [NOTES[0], {**NOTES[1], "patient_id": None}]
        ),
        output=tmp_path / "out.jsonl",
        mode="synthetic",
        patient_column="patient_id",
    )
    with pytest.raises(BatchError, match="Line 2: patient ID"):
        run(cfg)
    assert not (tmp_path / "out.jsonl").exists() and detector.calls == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([NOTES[0], {**NOTES[1], "note_id": "n1"}], "Line 2: duplicate note ID"),
        ([NOTES[0], {"text": "Ivo Strand"}], "Line 2: note ID must be"),
        ([NOTES[0], {**NOTES[1], "note_id": ""}], "Line 2: note ID must be"),
        ([NOTES[0], {**NOTES[1], "note_id": 7}], "Line 2: note IDs mix"),
        ([NOTES[0], {"note_id": "n2"}], "Line 2: text must be a string"),
    ],
)
def test_missing_or_duplicate_ids_are_rejected_before_any_output(
    tmp_path, detector, rows, message
):
    cfg = config(tmp_path, rows=rows)
    with pytest.raises(BatchError, match=message) as error:
        run(cfg)
    assert not cfg.output.exists() and detector.calls == []
    assert "n1" not in str(error.value) and "Ivo" not in str(error.value)


def test_cli_reports_progress_without_phi_and_exits_nonzero_on_failure(
    tmp_path, detector, monkeypatch, capsys
):
    rows = [
        *NOTES,
        {"note_id": "MRN-0042-secret", "patient_id": "p9", "text": "a" * 100_001},
    ]
    source = write_jsonl(tmp_path / "notes.jsonl", rows)
    summary_path = tmp_path / "summary.json"
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    # Report after every batch instead of every 30 seconds.
    monkeypatch.setattr(batch, "run", functools.partial(batch.run, progress_interval=0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "outis",
            "batch",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--mode",
            "synthetic",
            "--patient-column",
            "patient_id",
            "--batch-size",
            "2",
            "--summary-json",
            str(summary_path),
        ],
    )
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "4/8 notes (50.0%)" in captured.err and "tok/s" in captured.err
    assert "ETA" in captured.err and "failed 1 (text_too_long 1)" in captured.err
    for row in rows:
        for value in (row["note_id"], row["patient_id"]):
            assert value not in captured.err
    assert not any(value in captured.err for value in PHI)
    summary = json.loads(summary_path.read_text())
    assert summary["output"]["failed"] == 1 and summary["processed"] == 8
    assert not any(value in summary_path.read_text() for value in PHI)


def test_cli_input_errors_exit_2_without_values(
    tmp_path, detector, monkeypatch, capsys
):
    source = write_jsonl(
        tmp_path / "notes.jsonl", [NOTES[0], {**NOTES[2], "note_id": "n1"}]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "outis",
            "batch",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "o.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "duplicate note ID" in err and "n1" not in err and "Mara" not in err


def test_project_scope_uses_the_project_id_for_every_note(
    tmp_path, detector, monkeypatch
):
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"replacement": {"type": "synthetic", "scope": "project"}})
    )
    rows = [
        {"note_id": "a", "patient_id": "p1", "text": "Seen by Owen Pell today."},
        {"note_id": "b", "patient_id": "p2", "text": "Owen Pell signed the note."},
    ]
    fields = {
        "mode": "synthetic",
        "patient_column": "patient_id",
        "policy_file": policy,
    }
    with pytest.raises(BatchError, match="Project scope requires an explicit"):
        run(config(tmp_path, rows=rows, **fields))
    assert not (tmp_path / "out.jsonl").exists()
    cfg = config(tmp_path, rows=rows, project_id="release-1", **fields)
    run(cfg)
    first, second = read_jsonl(cfg.output)
    surrogate = first["text"][8 : first["entities"][0]["processed_end"]]
    assert "Owen Pell" not in surrogate and second["text"].startswith(surrogate)
    library = Deidentifier(detector=FakeDetector(), secret=SECRET)
    expected = library.deidentify(
        rows[1]["text"],
        mode="synthetic",
        patient_id="p2",
        replacement={"type": "synthetic", "scope": "project"},
        project_id="release-1",
    )
    assert second["text"] == expected["text"]


def test_site_dictionaries_apply_to_every_note(tmp_path, detector):
    manifest = tmp_path / "sites.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "dictionaries": [
                    {"name": "units", "label": "FACILITY", "terms": ["Wexmoor Ward"]}
                ],
            }
        )
    )
    rows = [
        {"note_id": "a", "text": "Moved to Wexmoor Ward overnight."},
        {"note_id": "b", "text": "Mara Calder stayed on Wexmoor Ward."},
    ]
    cfg = config(tmp_path, rows=rows, site_dictionaries=str(manifest))
    summary = run(cfg)
    first, second = read_jsonl(cfg.output)
    assert first["text"] == "Moved to [FACILITY] overnight."
    assert second["text"] == "[NAME] stayed on [FACILITY]."
    assert first["entities"][0]["source"] == "site_dictionary:units"
    assert summary["output"]["ok"] == 2
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy": {"site_dictionaries": ["wards"]}}))
    with pytest.raises(BatchError, match="Unknown site dictionary"):
        run(config(tmp_path, name="o.jsonl", rows=rows, policy_file=policy))
    with pytest.raises(BatchError, match="does not exist"):
        run(config(tmp_path, name="o.jsonl", site_dictionaries=str(tmp_path / "x")))


def test_policy_files_are_validated_up_front(tmp_path, detector):
    bad = tmp_path / "policy.json"
    bad.write_text(json.dumps({"policy": {"profile": "nope"}}))
    with pytest.raises(BatchError, match="Invalid detection policy"):
        run(config(tmp_path, policy_file=bad))
    bad.write_text(json.dumps({"exclude": ["DATE"]}))
    with pytest.raises(BatchError, match='"policy"'):
        run(config(tmp_path, policy_file=bad))
    assert detector.calls == []
