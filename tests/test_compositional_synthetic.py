import hashlib
import json
import random
import re
from collections import Counter

import pytest

from clinical_deid.data.generators import compositional
from clinical_deid.data.generators import narrative as baseline
from clinical_deid.evaluation.evaluate import load_verified_corpus
from clinical_deid.data.generators.base import MarkedValue, render_marked, validate_row
from clinical_deid.types import ENTITY_TYPES


def _values(row):
    return Counter((span["label"], row["text"][span["start"]:span["end"]]) for span in row["spans"])


def test_compositional_changes_prose_and_preserves_every_protected_value():
    for seed in (630429, 81):
        originals = list(baseline.generate_rows(200, split="train", seed=seed))
        generated = list(compositional.generate_rows(200, seed=seed))
        assert len(generated) == len(originals) == 200
        for row, original in zip(generated, originals):
            assert _values(row) == _values(original)
            assert compositional._protected_values(row) == _values(row)
            assert row["text"] != original["text"]
            # Public references remain clinical negatives, with exactly the
            # same bibliographic values rather than newly sampled examples.
            citations = r"PMID \d+|publication year \d{4}|published in \d{4}"
            assert Counter(re.findall(citations, row["text"])) == Counter(re.findall(citations, original["text"]))
        assert Counter(row["family"].rsplit("_", 1)[-1] for row in generated) == {"dense": 120, "focused": 60, "clean": 20}
        assert sum(not row["spans"] for row in generated) == 20


def test_compositional_grammar_is_productive_without_copying_development_frames():
    for kind in baseline.FRAMES["train"]:
        rng = random.Random(8993)
        frames = {compositional._frame(kind, rng) for _ in range(100)}
        assert len(frames) > 40, kind
        assert frames.isdisjoint(baseline.FRAMES["validation"][kind])
        assert all(re.search(r"\{\w+\}", frame) for frame in frames)
    rng = random.Random(8993)
    frames = {compositional._joint_frame(rng, ("name", "clinician", "location", "facility", "date")) for _ in range(100)}
    assert len(frames) > 40
    assert frames.isdisjoint(baseline.NARRATIVE_FRAMES["validation"])
    assert all(Counter(re.findall(r"\{(\w+)\}", frame)) == {"name": 1, "clinician": 1, "location": 1, "facility": 1, "date": 1} for frame in frames)


def test_compositional_joint_roles_do_not_make_an_ambiguous_comma_name_list():
    values = {
        "name": MarkedValue("NAME", "Vovel, Ba"),
        "clinician": MarkedValue("NAME", "R. Noveba"),
        "location": MarkedValue("LOCATION", "Bela"),
        "facility": MarkedValue("FACILITY", "Vemeda Care Residence"),
        "date": MarkedValue("DATE", "2012-04-23"),
    }
    rng = random.Random(6742)
    for _ in range(100):
        for fields in (("name", "clinician", "location", "facility"), tuple(values)):
            frame = compositional._joint_frame(rng, fields)
            text, spans = render_marked(frame, values)
            assert re.search(r"patient|care recipient", frame, re.IGNORECASE)
            assert re.search(r"clinician|examining professional|care practitioner|clinical reviewer", frame, re.IGNORECASE)
            assert Counter(span["label"] for span in spans)["NAME"] == 2
            assert Counter(span["label"] for span in spans)["DATE"] == ("date" in fields)
            for first, second in zip(spans, spans[1:]):
                if first["label"] == second["label"] == "NAME":
                    bridge = text[first["end"]:second["start"]]
                    # Sentence boundaries or explicit role words can separate
                    # roles. Bare commas and whitespace cannot distinguish a
                    # second person from the comma inside a reversed full name.
                    assert bridge.strip(" ,\t\r\n")


def test_compositional_gold_aligns_and_retains_single_token_supervision():
    tags, labels, maximum = Counter(), set(), 0
    for row in compositional.generate_rows(200):
        validate_row(row)
        assert row["split"] == "train" and row["suite"] == "authored_compositional_training"
        length, row_tags = baseline._token_statistics(row, baseline._encoding())
        tags.update(row_tags)
        maximum = max(maximum, length)
        for span in row["spans"]:
            labels.add(span["label"])
            if span["label"] == "AGE":
                age = row["text"][span["start"]:span["end"]]
                assert age.isdigit() and int(age) > 89
        for match in re.finditer(r"PMID \d+|Pain 3/10|strength 5/5|MoCA score 27/30|metformin 500 mg", row["text"]):
            assert all(match.end() <= span["start"] or match.start() >= span["end"] for span in row["spans"])
    assert labels == set(ENTITY_TYPES)
    assert tags["S-NAME"] >= 300 and tags["S-LOCATION"] >= 200
    assert maximum <= 500


def test_compositional_reproducibility_and_invalid_inputs():
    assert list(compositional.generate_rows(20, seed=53)) == list(compositional.generate_rows(20, seed=53))
    assert list(compositional.generate_rows(20, seed=53)) != list(compositional.generate_rows(20, seed=54))
    assert list(compositional.generate_rows(0)) == []
    for count in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            list(compositional.generate_rows(count))
    with pytest.raises(ValueError, match="Unknown"):
        compositional._frame("unknown", random.Random(19))


def test_compositional_manifest_is_canonical_hash_verified_and_training_only(tmp_path):
    manifest = compositional.write_dataset(tmp_path, train_count=40)
    path = tmp_path / "narrative_compositional_train.jsonl"
    rows, frozen = load_verified_corpus(path)
    assert frozen == manifest
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(rows) == manifest["rows"] == manifest["experiment"]["rows_verified_against_baseline"] == 40
    assert json.loads((tmp_path / "narrative_compositional_manifest.json").read_text()) == manifest
    assert manifest["license"] == "CC0-1.0"
    assert manifest["split"] == "train"
    assert manifest["provenance"]["external_datasets"] is False
    assert manifest["provenance"]["identity_lookup_dictionaries"] is False
    assert manifest["provenance"]["development_sentences_copied"] is False
    assert manifest["provenance"]["independent_qualification_used"] is False
    assert manifest["experiment"]["per_note_protected_values_and_labels_unchanged"] is True
    assert manifest["experiment"]["development_corpora_modified"] is False
    assert manifest["length_tokens"]["max"] <= 500
