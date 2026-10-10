import json

import pytest

from outis.data.generators.base import (
    TRAIN_TEMPLATES,
    VALIDATION_TEMPLATES,
    VOCABULARY,
    MarkedValue,
    generate_rows,
    render_marked,
    validate_row,
    write_dataset,
)
from outis.types import ENTITY_TYPES


def test_marked_offsets_with_unicode_repetition_and_regex_characters():
    text, spans = render_marked(
        "☎ {phone}; {name}\r\nRepeat: {name} / glucose {result}",
        {
            "phone": MarkedValue("PHONE", "+1 (207) 555-0101"),
            "name": MarkedValue("NAME", "Chloë O'Silver"),
            "result": "112",
        },
    )
    assert [text[s["start"] : s["end"]] for s in spans] == [
        "+1 (207) 555-0101",
        "Chloë O'Silver",
        "Chloë O'Silver",
    ]
    assert [s["label"] for s in spans] == ["PHONE", "NAME", "NAME"]
    assert text.endswith("glucose 112")


def test_splits_reproducible_and_independent():
    assert list(generate_rows(40, "train", 41)) == list(generate_rows(40, "train", 41))
    assert list(generate_rows(40, "train", 41)) != list(generate_rows(40, "train", 42))
    assert (
        list(generate_rows(5, "validation", 41))
        == list(generate_rows(40, "validation", 41))[:5]
    )


def test_validation_families_and_identity_vocabulary_are_disjoint():
    assert {t.family for t in TRAIN_TEMPLATES}.isdisjoint(
        t.family for t in VALIDATION_TEMPLATES
    )
    for key in VOCABULARY["train"]:
        assert set(VOCABULARY["train"][key]).isdisjoint(VOCABULARY["validation"][key])
    train = list(generate_rows(100, "train"))
    validation = list(generate_rows(100, "validation"))
    assert {r["family"] for r in train}.isdisjoint(r["family"] for r in validation)
    train_names = {
        r["text"][s["start"] : s["end"]]
        for r in train
        for s in r["spans"]
        if s["label"] == "NAME"
    }
    validation_names = {
        r["text"][s["start"] : s["end"]]
        for r in validation
        for s in r["spans"]
        if s["label"] == "NAME"
    }
    assert train_names.isdisjoint(validation_names)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_every_entity_and_clinical_negatives_present_with_valid_offsets(split):
    rows = list(generate_rows(500, split, 7))
    assert {s["label"] for r in rows for s in r["spans"]} == set(ENTITY_TYPES)
    assert len({r["id"] for r in rows}) == len(rows)
    assert any(not r["spans"] for r in rows)
    assert any("\r\n" in r["text"] for r in rows)
    for row in rows:
        validate_row(row)
        assert row["split"] == split
        for span in row["spans"]:
            surface = row["text"][span["start"] : span["end"]]
            assert surface.strip()
            if span["label"] == "AGE":
                assert int(surface) > 89


def test_invalid_annotations_are_rejected():
    row = {
        "id": "train-0",
        "split": "train",
        "text": "Patient Adela",
        "spans": [
            {"start": 8, "end": 13, "label": "NAME"},
            {"start": 10, "end": 13, "label": "NAME"},
        ],
    }
    with pytest.raises(ValueError, match="Invalid span"):
        validate_row(row)
    with pytest.raises(ValueError, match="Unknown template field"):
        render_marked("{missing}", {})
    with pytest.raises(ValueError):
        list(generate_rows(-1))


def test_written_dataset_manifest_and_reproducibility(tmp_path):
    manifest = write_dataset(tmp_path, train_count=75, val_count=35, seed=99)
    assert manifest["provenance"]["external_datasets"] is False
    assert manifest["license"] == "CC0-1.0"
    assert manifest["generator_code_license"] == "Apache-2.0"
    assert manifest["split_audit"]["identity_vocabulary_intersections"] == {}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    original = (tmp_path / "train.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.decode().splitlines()]
    assert len(rows) == 75
    assert manifest["datasets"]["train"]["count"] == 75
    assert manifest["datasets"]["train"]["negative_notes"] > 0
    write_dataset(tmp_path, train_count=75, val_count=35, seed=99)
    assert (tmp_path / "train.jsonl").read_bytes() == original
