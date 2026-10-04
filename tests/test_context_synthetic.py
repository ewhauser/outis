import hashlib
import json

from clinical_deid.data.generators.context import (
    AGE_FRAMES,
    CONTACT_BUNDLES,
    EMAIL_FRAMES,
    GEOGRAPHY_ROOTS,
    LOCATION_FRAMES,
    generate_rows,
    write_dataset,
)
from clinical_deid.data.generators.base import validate_row
from clinical_deid.types import ENTITY_TYPES


def test_context_generation_valid_and_covers_all_labels():
    rows = list(generate_rows(120))
    assert {span["label"] for row in rows for span in row["spans"]} == set(ENTITY_TYPES)
    for row in rows:
        validate_row(row)
        if row["family"].endswith("negative"):
            assert row["spans"] == []
        else:
            assert {"AGE", "LOCATION", "EMAIL", "PHONE", "URL", "IP"}.issubset(
                span["label"] for span in row["spans"]
            )
        for span in row["spans"]:
            value = row["text"][span["start"] : span["end"]]
            if span["label"] == "AGE":
                assert value.isdigit() and int(value) > 89
            if span["label"] == "EMAIL":
                assert "@" in value
                assert value[0] not in ";,|<(['"
                assert value[-1] not in ";,|>)]'"


def test_context_splits_have_separate_geographies_and_frames():
    assert set(GEOGRAPHY_ROOTS["train"]).isdisjoint(GEOGRAPHY_ROOTS["validation"])
    for frames in (AGE_FRAMES, CONTACT_BUNDLES, EMAIL_FRAMES, LOCATION_FRAMES):
        assert set(frames["train"]).isdisjoint(frames["validation"])
    train = list(generate_rows(20, seed=22))
    assert train == list(generate_rows(20, seed=22))
    assert train != list(generate_rows(20, seed=23))
    validation = list(generate_rows(20, split="validation", seed=22))
    assert {row["family"] for row in train}.isdisjoint(
        row["family"] for row in validation
    )


def test_context_manifest_has_hashes_and_authored_provenance(tmp_path):
    manifest = write_dataset(tmp_path, train_count=12, val_count=7)
    assert manifest["provenance"]["independent_final_evaluation_used"] is False
    assert manifest["provenance"]["external_datasets"] is False
    assert manifest["license"] == "CC0-1.0"
    assert json.loads((tmp_path / "context_manifest.json").read_text()) == manifest
    for split in ("train", "validation"):
        contents = (tmp_path / f"context_{split}.jsonl").read_bytes()
        assert (
            hashlib.sha256(contents).hexdigest()
            == manifest["datasets"][split]["sha256"]
        )
