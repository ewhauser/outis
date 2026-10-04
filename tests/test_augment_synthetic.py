import hashlib
import json
import re

from clinical_deid.data.generators.augment import (
    CLAUSES,
    EPONYMS,
    IDENTITIES,
    generate_rows,
    write_dataset,
)
from clinical_deid.data.generators.base import validate_row
from clinical_deid.types import ENTITY_TYPES


def test_augmentation_offsets_and_every_entity_dense():
    rows = list(generate_rows(100))
    for row in rows:
        validate_row(row)
        assert row["split"] == "train"
        assert row["suite"] == "augmented_training"
        if "dense" in row["family"]:
            assert {span["label"] for span in row["spans"]} == set(ENTITY_TYPES)
        for span in row["spans"]:
            value = row["text"][span["start"] : span["end"]]
            assert value
            if span["label"] == "AGE":
                assert value.isdigit() and int(value) > 89
            if span["label"] == "NAME":
                assert not value.startswith(("Dr.", "Nurse "))
    assert sum(not row["spans"] for row in rows) == 10
    assert any("straße" in row["text"] for row in rows)


def test_clinical_eponyms_public_identifiers_and_scores_are_not_phi():
    rows = list(generate_rows(150, seed=77))
    for row in rows:
        intervals = [(span["start"], span["end"]) for span in row["spans"]]
        for eponym in EPONYMS:
            for match in re.finditer(re.escape(eponym), row["text"]):
                assert all(
                    match.end() <= start or match.start() >= end
                    for start, end in intervals
                )
        for pattern in (
            r"PMID \d+",
            r"publication year \d{4}",
            r"score was \d{2}/30",
            r"Scale was 15/15",
            r"\d+-year-old adult",
        ):
            for match in re.finditer(pattern, row["text"]):
                assert all(
                    match.end() <= start or match.start() >= end
                    for start, end in intervals
                )


def test_augmentation_splits_and_paraphrases_disjoint_and_reproducible():
    for category in IDENTITIES["train"]:
        assert set(IDENTITIES["train"][category]).isdisjoint(
            IDENTITIES["validation"][category]
        )
    for label in ENTITY_TYPES:
        assert set(CLAUSES["train"][label]).isdisjoint(CLAUSES["validation"][label])
    train = list(generate_rows(25, seed=41))
    assert train == list(generate_rows(25, seed=41))
    assert train != list(generate_rows(25, seed=42))
    validation = list(generate_rows(25, split="validation", seed=41))
    assert {row["family"] for row in train}.isdisjoint(
        row["family"] for row in validation
    )
    assert all(row["suite"] == "augmented_validation" for row in validation)


def test_augmentation_written_manifest_matches_content(tmp_path):
    manifest = write_dataset(tmp_path, train_count=25, val_count=15, seed=30)
    assert manifest["license"] == "CC0-1.0"
    assert manifest["provenance"]["external_datasets"] is False
    assert json.loads((tmp_path / "augmentation_manifest.json").read_text()) == manifest
    for split, count in (("train", 25), ("validation", 15)):
        contents = (tmp_path / f"augmentation_{split}.jsonl").read_bytes()
        assert len(contents.splitlines()) == count
        assert (
            hashlib.sha256(contents).hexdigest()
            == manifest["datasets"][split]["sha256"]
        )
