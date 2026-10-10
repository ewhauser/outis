"""Frozen manifest compatibility and cached metric implementation provenance."""

import hashlib
import json

import pytest

import outis.evaluation.evaluate as evaluation
from outis.types import ENTITY_TYPES


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
