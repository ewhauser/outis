import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime

from outis.data.generators.base import validate_row
from outis.data.generators.narrative import (
    FRAMES,
    NARRATIVE_FRAMES,
    NEGATIVE_FRAMES,
    _context,
    _encoding,
    _token_statistics,
    generate_rows,
    single_token_words,
    write_dataset,
)
from outis.types import ENTITY_TYPES


def test_narrative_all_types_clean_notes_and_exact_offsets():
    rows = list(generate_rows(200))
    assert {span["label"] for row in rows for span in row["spans"]} == set(ENTITY_TYPES)
    assert sum(not row["spans"] for row in rows) == 20
    assert Counter(row["family"].rsplit("_", 1)[-1] for row in rows) == {
        "dense": 120,
        "focused": 60,
        "clean": 20,
    }
    for row in rows:
        validate_row(row)
        for span in row["spans"]:
            surface = row["text"][span["start"] : span["end"]]
            if span["label"] == "AGE":
                assert surface.isdigit() and int(surface) > 89
            if span["label"] == "NAME":
                assert not surface.startswith("Dr. ")


def test_narrative_single_token_supervision_and_bounded_lengths():
    for split in ("train", "validation"):
        tags, lengths = Counter(), []
        for row in generate_rows(200, split=split):
            length, counts = _token_statistics(row, _encoding())
            lengths.append(length)
            tags.update(counts)
        assert tags["S-NAME"] >= 300
        assert tags["S-LOCATION"] >= 200
        assert max(lengths) <= 500
        assert tags["B-NAME"] and tags["E-NAME"] and tags["O"]


def test_narrative_identity_grammar_and_frames_are_disjoint():
    encoding = _encoding()
    train_pool, validation_pool = (
        single_token_words("train"),
        single_token_words("validation"),
    )
    assert {word.casefold() for word in train_pool}.isdisjoint(
        word.casefold() for word in validation_pool
    )
    for word in train_pool + validation_pool:
        assert word.isalpha() and len(word) in (2, 3, 4)
        assert len(encoding.encode(word)) == 1 and len(encoding.encode(" " + word)) == 1
    for key in FRAMES["train"]:
        assert set(FRAMES["train"][key]).isdisjoint(FRAMES["validation"][key])
    assert set(NARRATIVE_FRAMES["train"]).isdisjoint(NARRATIVE_FRAMES["validation"])
    assert set(NEGATIVE_FRAMES["train"]).isdisjoint(NEGATIVE_FRAMES["validation"])
    training = list(generate_rows(20, seed=45))
    assert training == list(generate_rows(20, seed=45))
    assert training != list(generate_rows(20, seed=46))
    development = list(generate_rows(20, split="validation", seed=45))
    assert {row["family"] for row in training}.isdisjoint(
        row["family"] for row in development
    )


def test_narrative_names_include_multiple_forms_and_clinical_negatives_are_unmarked():
    rows = list(generate_rows(200))
    names = [
        row["text"][span["start"] : span["end"]]
        for row in rows
        for span in row["spans"]
        if span["label"] == "NAME"
    ]
    assert any(name.isupper() for name in names)
    assert any("," in name for name in names)
    assert any(re.match(r"^[A-Z]\. ", name) for name in names)
    assert any(any(ord(letter) > 127 for letter in name) for name in names)
    for row in rows:
        for match in re.finditer(
            r"PMID \d+|publication year \d{4}|published in \d{4}|Pain 3/10|strength 5/5|"
            r"MoCA score 27/30|sodium 139 mmol/L|metformin 500 mg|Cushing syndrome|Romberg testing",
            row["text"],
            re.IGNORECASE,
        ):
            assert all(
                match.end() <= span["start"] or match.start() >= span["end"]
                for span in row["spans"]
            )


def test_narrative_dates_are_coherent_and_identifier_shapes_are_shared():
    for split in ("train", "validation"):
        context = _context(random.Random(1981), split, 0)
        current, previous, following = [
            datetime.strptime(context[key].text, "%Y-%m-%d")
            for key in ("date", "past_date", "future_date")
        ]
        assert 7 <= (current - previous).days <= 28
        assert 14 <= (following - current).days <= 42
        shapes = {label: set() for label in ("mrn", "account", "identifier")}
        rng = random.Random(4812)
        for index in range(250):
            context = _context(rng, split, index)
            for label in shapes:
                # Normalize letter prefixes to one symbol: every class uses
                # numeric, zero-padded, slash, hyphen, and letter-prefix forms.
                value = re.sub(r"[A-Z]", "X", context[label].text)
                shapes[label].add(re.sub(r"\d", "0", value))
        assert shapes["mrn"] == shapes["account"] == shapes["identifier"]
        assert len(shapes["mrn"]) == 5


def test_narrative_manifests_hashes_and_authored_provenance(tmp_path):
    manifest = write_dataset(tmp_path, train_count=40, val_count=20)
    assert manifest["license"] == "CC0-1.0"
    assert manifest["provenance"]["external_datasets"] is False
    assert manifest["provenance"]["identity_lookup_dictionaries"] is False
    assert manifest["provenance"]["independent_qualification_used"] is False
    assert manifest["split_audit"]["identity_overlap_count"] == 0
    assert json.loads((tmp_path / "narrative_manifest.json").read_text()) == manifest
    for split in ("train", "validation"):
        details = manifest["datasets"][split]
        content = (tmp_path / f"narrative_{split}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == details["sha256"]
        adjacent = json.loads(
            (tmp_path / f"narrative_{split}.jsonl.manifest.json").read_text()
        )
        assert adjacent["sha256"] == details["sha256"] and adjacent["split"] == split
        from outis.evaluation.evaluate import load_verified_corpus

        verified, _ = load_verified_corpus(tmp_path / f"narrative_{split}.jsonl")
        assert len(verified) == details["count"]
        assert (
            details["token_counts"]["S-NAME"] and details["token_counts"]["S-LOCATION"]
        )
