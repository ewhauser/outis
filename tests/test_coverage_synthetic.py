import hashlib
import json
import re

from clinical_deid.data.generators.coverage import CARE_SUFFIXES, FRAMES, ROLE_RELATIONSHIPS, VOCAB, generate_rows, write_dataset
from clinical_deid.data.generators.base import validate_row
from clinical_deid.types import ENTITY_TYPES


def test_coverage_every_type_and_exact_offsets():
    rows = list(generate_rows(180))
    assert {span["label"] for row in rows for span in row["spans"]} == set(ENTITY_TYPES)
    assert any(not row["spans"] for row in rows)
    assert any("PATlENT" in row["text"] or "D0CT0R" in row["text"] for row in rows)
    for row in rows:
        validate_row(row)
        for span in row["spans"]:
            surface = row["text"][span["start"]:span["end"]]
            if span["label"] == "AGE":
                assert surface.isdigit() and int(surface) > 89
            if span["label"] == "NAME":
                assert not surface.startswith("Dr. ")
            if span["label"] == "FACILITY":
                assert any(surface.endswith(suffix) for suffix in CARE_SUFFIXES)


def test_coverage_name_forms_and_identifier_shapes_have_context_supervision():
    rows = list(generate_rows(180))
    names = [row["text"][span["start"]:span["end"]] for row in rows for span in row["spans"] if span["label"] == "NAME"]
    assert any(name.isupper() for name in names)
    assert any("," in name for name in names)
    assert any(re.match(r"^\w\. ", name) for name in names)
    assert any(any(ord(char) > 127 for char in name) for name in names)
    for row in rows:
        for match in re.finditer(r"(?:pain|Pain) [23]/10|(?:score|testing) \d{2}/30|(?:Scale|scale) 15/15|PMID \d+", row["text"]):
            assert all(match.end() <= span["start"] or match.start() >= span["end"] for span in row["spans"])


def test_coverage_splits_are_disjoint_and_reproducible():
    for key in VOCAB["train"]:
        assert set(VOCAB["train"][key]).isdisjoint(VOCAB["validation"][key])
    for key in FRAMES["train"]:
        assert set(FRAMES["train"][key]).isdisjoint(FRAMES["validation"][key])
    assert set(ROLE_RELATIONSHIPS["train"]).isdisjoint(ROLE_RELATIONSHIPS["validation"])
    train = list(generate_rows(12, seed=25))
    assert train == list(generate_rows(12, seed=25))
    assert train != list(generate_rows(12, seed=26))
    validation = list(generate_rows(12, split="validation", seed=25))
    assert {row["family"] for row in train}.isdisjoint(row["family"] for row in validation)


def test_coverage_manifest_hashes_and_authored_provenance(tmp_path):
    manifest = write_dataset(tmp_path, train_count=15, val_count=8)
    assert manifest["license"] == "CC0-1.0"
    assert manifest["provenance"]["external_datasets"] is False
    assert manifest["provenance"]["independent_final_evaluation_used"] is False
    assert json.loads((tmp_path / "coverage_manifest.json").read_text()) == manifest
    for split in ("train", "validation"):
        content = (tmp_path / f"coverage_{split}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest["datasets"][split]["sha256"]
