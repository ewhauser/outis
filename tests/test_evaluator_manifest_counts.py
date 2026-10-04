"""Frozen manifest compatibility and cached metric implementation provenance."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import clinical_deid.evaluation.evaluate as evaluation
from clinical_deid.types import ENTITY_TYPES


def frozen_fixture(tmp_path, counts):
    rows = [
        {
            "id": "development-1",
            "text": "Zoë reports fever.",
            "spans": [{"start": 0, "end": 3, "label": "NAME"}],
            "split": "validation",
            "suite": "development",
        }
    ]
    path = tmp_path / "development.jsonl"
    payload = (json.dumps(rows[0], ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    manifest = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        **counts,
        "label_support": {label: int(label == "NAME") for label in ENTITY_TYPES},
    }
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(manifest))
    return path, sidecar, rows, manifest


@pytest.mark.parametrize(
    "counts", [{"rows": 1}, {"documents": 1}, {"rows": 1, "documents": 1}]
)
def test_frozen_count_aliases_preserve_original_manifest(tmp_path, counts):
    path, sidecar, expected_rows, manifest = frozen_fixture(tmp_path, counts)
    before = (path.read_bytes(), sidecar.read_bytes())
    rows, actual = evaluation.load_verified_corpus(path)
    assert rows == expected_rows
    assert actual == manifest
    assert (path.read_bytes(), sidecar.read_bytes()) == before
    assert {key for key in ("rows", "documents") if key in actual} == set(counts)


@pytest.mark.parametrize(
    "counts",
    [
        {"rows": 1, "documents": 2},
        {"rows": 2, "documents": 1},
    ],
)
def test_conflicting_frozen_counts_fail_even_if_one_is_correct(tmp_path, counts):
    path, _, _, _ = frozen_fixture(tmp_path, counts)
    with pytest.raises(ValueError, match="conflicting"):
        evaluation.load_verified_corpus(path)


@pytest.mark.parametrize(
    "counts", [{"rows": 2}, {"documents": 2}, {"rows": 2, "documents": 2}]
)
def test_declared_count_must_match_verified_payload(tmp_path, counts):
    path, _, _, _ = frozen_fixture(tmp_path, counts)
    with pytest.raises(ValueError, match="row count differs"):
        evaluation.load_verified_corpus(path)


@pytest.mark.parametrize("key", ["rows", "documents"])
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, -1])
def test_frozen_count_must_be_a_nonnegative_json_integer(tmp_path, key, value):
    path, _, _, _ = frozen_fixture(tmp_path, {key: value})
    with pytest.raises(ValueError, match="nonnegative integer"):
        evaluation.load_verified_corpus(path)


def test_missing_count_is_rejected(tmp_path):
    path, _, _, _ = frozen_fixture(tmp_path, {})
    with pytest.raises(ValueError, match="rows or documents count"):
        evaluation.load_verified_corpus(path)


@pytest.mark.parametrize("key", ["rows", "documents"])
def test_count_alias_does_not_weaken_checksum_or_label_support(tmp_path, key):
    path, sidecar, _, manifest = frozen_fixture(tmp_path, {key: 1})
    payload = path.read_bytes()
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation.load_verified_corpus(path)
    path.write_bytes(payload)
    manifest["label_support"]["NAME"] = 0
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="label support"):
        evaluation.load_verified_corpus(path)


@pytest.mark.parametrize("key", ["rows", "documents"])
def test_zero_count_preserves_existing_empty_loader_semantics(tmp_path, key):
    path, sidecar, _, manifest = frozen_fixture(tmp_path, {key: 0})
    path.write_bytes(b"")
    manifest["sha256"] = hashlib.sha256(b"").hexdigest()
    manifest["label_support"] = dict.fromkeys(ENTITY_TYPES, 0)
    sidecar.write_text(json.dumps(manifest))
    assert evaluation.load_verified_corpus(path) == ([], manifest)


def test_cached_runtime_provenance_pins_evaluator_implementation(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "v12_manifest_provenance_scorer", scripts / "score_cached_candidates.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    monkeypatch.setattr(scorer, "neural_runtime_manifest", dict)
    monkeypatch.setattr(scorer, "rules_source_manifest", dict)
    monkeypatch.setattr(
        scorer.train, "loaded_function_fingerprint", lambda function: function.__name__
    )
    metric_source = tmp_path / "evaluate.py"
    metric_source.write_text("def evaluate(): return 1\n")
    monkeypatch.setattr(evaluation, "__file__", str(metric_source))
    before = scorer.runtime_provenance()
    assert (
        before["evaluation_source_sha256"]
        == hashlib.sha256(metric_source.read_bytes()).hexdigest()
    )
    assert scorer.runtime_provenance() == before
    metric_source.write_text("def evaluate(): return 2\n")
    after = scorer.runtime_provenance()
    assert (
        after["evaluation_source_sha256"]
        == hashlib.sha256(metric_source.read_bytes()).hexdigest()
    )
    assert after["evaluation_source_sha256"] != before["evaluation_source_sha256"]
    assert {
        key: value for key, value in after.items() if key != "evaluation_source_sha256"
    } == {
        key: value for key, value in before.items() if key != "evaluation_source_sha256"
    }
