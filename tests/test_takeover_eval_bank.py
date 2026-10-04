"""Evaluation writes only verified, immutable development/proxy evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from clinical_deid.detection import hf_encoder
from clinical_deid.detection import rules
from clinical_deid.types import ENTITY_TYPES, Span


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        "takeover_" + name, ROOT / "scripts/takeover" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script("eval_bank")
selection = load_script("select_sources")


def bank(tmp_path, name="development", extra=None):
    corpus = tmp_path / (name + ".jsonl")
    rows = [
        {
            "id": "one",
            "text": "Zoë reports fever.",
            "spans": [{"start": 0, "end": 3, "label": "NAME"}],
            "split": "validation",
            "suite": "development",
            "cohort": "clinical",
        },
        {
            "id": "two",
            "text": "No fever.",
            "spans": [],
            "split": "validation",
            "suite": "development",
        },
    ]
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    corpus.write_bytes(payload)
    manifest = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "independent_final": False,
        "split": "validation",
        "suite": "development",
        "label_support": {label: int(label == "NAME") for label in ENTITY_TYPES},
    } | (extra or {})
    corpus.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    return corpus


def fake_cli(tmp_path, monkeypatch, *, corpus=None, predictions=True, extra=()):
    corpus = corpus or bank(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"synthetic fixture weights")
    report, prediction_path = tmp_path / "report.json", tmp_path / "predictions.json"
    args = [
        "eval_bank.py",
        "--data",
        str(corpus),
        "--model-dir",
        str(model_dir),
        "--output",
        str(report),
        "--device",
        "cpu",
        "--model-labels",
        "",
        "--base-dir",
        str(tmp_path / "base"),
    ]
    if predictions:
        args += ["--predictions", str(prediction_path)]
    args += list(extra)
    monkeypatch.setattr(sys, "argv", args)

    class Detector:
        config = {"model_labels": []}
        encoding = None

        def raw_batch(self, texts):
            return [
                [Span(0, 3, "NAME")] if text.startswith("Zoë") else [] for text in texts
            ]

    monkeypatch.setattr(hf_encoder, "load_detector", lambda *a, **k: Detector())
    monkeypatch.setattr(rules, "augment", lambda text, spans: spans)
    monkeypatch.setattr(
        evaluation,
        "neural_runtime_manifest",
        lambda *a, **k: {"tokenizer": {"name": "fixture"}},
    )
    monkeypatch.setattr(
        evaluation, "source_manifest", lambda: {"hf_encoder.py": "a" * 64}
    )
    return corpus, model_dir, report, prediction_path


def test_producer_envelope_roundtrips_through_selection(tmp_path, monkeypatch):
    corpus, _, report, predictions = fake_cli(tmp_path, monkeypatch)
    evaluation.main()
    result = json.loads(report.read_text())
    envelope = json.loads(predictions.read_text())
    assert result["model"]["typed_span"]["f1"] == 1
    assert result["neural_runtime"] == envelope["neural_runtime"]
    assert result["source_manifest"] == envelope["source_manifest"]
    assert selection.score(selection.load([f"{corpus}={predictions}"]), set())[0] == 1


@pytest.mark.parametrize("kind", ["model", "runtime", "source", "corpus"])
def test_drift_leaves_neither_output_usable(tmp_path, monkeypatch, kind):
    corpus, model_dir, report, predictions = fake_cli(tmp_path, monkeypatch)
    original = hf_encoder.load_detector

    def mutating_loader(*args, **kwargs):
        detector = original(*args, **kwargs)
        if kind == "model":
            (model_dir / "model.safetensors").write_bytes(b"changed fixture")
        elif kind == "runtime":
            monkeypatch.setattr(
                evaluation, "neural_runtime_manifest", lambda *a, **k: {"changed": True}
            )
        elif kind == "source":
            monkeypatch.setattr(
                evaluation, "source_manifest", lambda: {"changed": "b" * 64}
            )
        else:
            corpus.write_bytes(corpus.read_bytes() + b"\n")
        return detector

    monkeypatch.setattr(hf_encoder, "load_detector", mutating_loader)
    with pytest.raises((SystemExit, ValueError), match="changed|SHA-256"):
        evaluation.main()
    assert not report.exists() and not predictions.exists()


@pytest.mark.parametrize("kind", ["report", "predictions"])
def test_existing_outputs_are_immutable_and_fail_before_inference(
    tmp_path, monkeypatch, kind
):
    _, _, report, predictions = fake_cli(tmp_path, monkeypatch)
    path = report if kind == "report" else predictions
    path.write_text("original")
    monkeypatch.setattr(
        hf_encoder,
        "load_detector",
        lambda *a, **k: pytest.fail("must fail before inference"),
    )
    with pytest.raises(FileExistsError, match="immutable"):
        evaluation.main()
    assert path.read_text() == "original"
    assert not (predictions if kind == "report" else report).exists()


def test_same_prediction_and_report_destination_fails_before_inference(
    tmp_path, monkeypatch
):
    _, _, report, _ = fake_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--predictions", str(report)])
    monkeypatch.setattr(
        hf_encoder,
        "load_detector",
        lambda *a, **k: pytest.fail("must fail before inference"),
    )
    with pytest.raises(ValueError, match="different paths"):
        evaluation.main()
    assert not report.exists()


@pytest.mark.parametrize(
    "metadata", [{"suite": "final"}, {"split": "test"}, {"independent_final": True}]
)
def test_retired_filename_does_not_bypass_final_metadata(
    tmp_path, monkeypatch, metadata
):
    corpus = bank(tmp_path, name="retired_development", extra=metadata)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *a, **k: pytest.fail("must reject before corpus read"),
    )
    with pytest.raises(ValueError, match="final/independent"):
        evaluation.load_bank(corpus)


@pytest.mark.parametrize(
    "mode", ["alias", "manifest_role", "manifest_flag", "copied_alias"]
)
def test_proxy_detail_guards_follow_identity_before_read(tmp_path, monkeypatch, mode):
    corpus = tmp_path / "alias.jsonl"
    metadata = {}
    if mode == "alias":
        corpus.symlink_to(tmp_path / "proxyq_a2.jsonl")
    elif mode == "manifest_role":
        metadata["proxy_role"] = "development"
    elif mode == "manifest_flag":
        metadata["is_proxy"] = True
    else:
        metadata["corpus"] = "data/proxyq_c4.jsonl"
    corpus.with_suffix(".jsonl.manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *a, **k: pytest.fail("must reject before corpus read"),
    )
    with pytest.raises(ValueError, match="held-out proxy"):
        evaluation.load_bank(corpus, detailed=True)


def test_proxy_aggregate_report_contains_no_failures_or_predictions(
    tmp_path, monkeypatch
):
    corpus = bank(tmp_path, name="proxyq_fixture", extra={"proxy_role": "blind"})
    _, _, report, predictions = fake_cli(
        tmp_path, monkeypatch, corpus=corpus, predictions=False
    )
    evaluation.main()
    result = json.loads(report.read_text())
    assert result["failures_written"] is False
    assert "failures" not in result["model"] and "failures" not in result["hybrid"]
    assert not predictions.exists()


@pytest.mark.parametrize("slot", ["corpus", "manifest", "model"])
def test_sensitive_symlinks_are_rejected_before_any_contents_read(
    tmp_path, monkeypatch, slot
):
    corpus = tmp_path / "development.jsonl"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if slot == "corpus":
        corpus.symlink_to(tmp_path / "blind_qualification_v9_revision3.jsonl")
    elif slot == "manifest":
        corpus.with_suffix(".jsonl.manifest.json").symlink_to(
            tmp_path / "private.sealed"
        )
    else:
        (model_dir / "config.json").symlink_to(
            tmp_path / "blind_qualification_v6.jsonl"
        )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *a, **k: pytest.fail("protected bytes must not be read"),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *a, **k: pytest.fail("protected text must not be read"),
    )
    with pytest.raises(ValueError, match="protected path"):
        evaluation.guarded_model_manifest(
            model_dir
        ) if slot == "model" else evaluation.load_bank(corpus)


@pytest.mark.parametrize(
    "extra", [("--limit", "0"), ("--limit", "-1"), ("--batch-size", "0")]
)
def test_invalid_batch_or_limit_rejected(tmp_path, monkeypatch, extra):
    _, _, report, predictions = fake_cli(tmp_path, monkeypatch, extra=extra)
    with pytest.raises(ValueError, match="positive"):
        evaluation.main()
    assert not report.exists() and not predictions.exists()


def test_partial_predictions_are_refused(tmp_path, monkeypatch):
    _, _, report, predictions = fake_cli(tmp_path, monkeypatch, extra=("--limit", "1"))
    with pytest.raises(SystemExit, match="complete development bank"):
        evaluation.main()
    assert not report.exists() and not predictions.exists()


def test_detector_missing_rows_cannot_be_silently_scored(tmp_path, monkeypatch):
    _, _, report, predictions = fake_cli(tmp_path, monkeypatch)

    class ShortDetector:
        def raw_batch(self, texts):
            return [[]]

    monkeypatch.setattr(hf_encoder, "load_detector", lambda *a, **k: ShortDetector())
    with pytest.raises(ValueError, match="every input row"):
        evaluation.main()
    assert not report.exists() and not predictions.exists()
