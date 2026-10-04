"""Construction invariants for independently authored development diagnostics."""
from collections import Counter, defaultdict
import hashlib
import json

import pytest
import tiktoken

from clinical_deid.evaluation.evaluate import load_verified_corpus
from clinical_deid.data.generators.remediation_development import (
    ENTITY_TYPES, POLICY, _Text, _negative, _short_positive, build_rows, generate_corpus, identity_group,
)


def fragments(row, label):
    return [row["text"][s["start"]:s["end"]] for s in row["spans"] if s["label"] == label]


def test_authored_offsets_are_character_offsets_and_occurrence_specific():
    b = _Text().add("Dr. ").add("Élodie Varela", "NAME").add(", MD saw ")
    b.add("Élodie Varela", "NAME").add("; she was well.")
    assert b.spans == [
        {"start": 4, "end": 17, "label": "NAME"},
        {"start": 26, "end": 39, "label": "NAME"},
    ]


def test_counts_metadata_pair_groups_and_development_partition():
    rows = build_rows(short_count=150, long_count=0)
    assert len(rows) == 150
    assert sum(not row["spans"] for row in rows) == 30
    pairs = defaultdict(list)
    for row in rows:
        assert row["split"] == "validation"
        assert row["suite"] == "development"
        assert row["annotation_policy"] == POLICY
        assert row["template_family"].startswith("remediation_dev.")
        assert isinstance(row["phenomena"], list)
        if row["spans"]:
            assert int(row["identity_group"][:8], 16) % 10 in {8, 9}
            assert row["identity_group"] in row["identity_groups"]
            assert all(int(group[:8], 16) % 10 in {8, 9} for group in row["identity_groups"])
            pairs[row["pair_id"]].append(row)
    assert all(len(pair) == 2 for pair in pairs.values())
    assert all(pair[0]["identity_group"] == pair[1]["identity_group"] for pair in pairs.values())
    support = Counter(span["label"] for row in rows for span in row["spans"])
    assert set(support) == set(ENTITY_TYPES)
    assert any(row["diagnostic_slice"] == "ocr_stress" for row in rows)
    assert any(row["cohort"] == "unusual_name" for row in rows)


def test_contextual_name_date_and_age_boundaries():
    row = _short_positive("Adriana Haldane", pair=4, variant=1, seed=7)
    assert "2021" in fragments(row, "DATE")
    text = row["text"]
    first, second = text.index("2021"), text.rindex("2021")
    assert first != second
    assert any(s["start"] == first and s["end"] == first + 4 for s in row["spans"])
    assert not any(s["start"] <= second < s["end"] for s in row["spans"])
    assert "101" in fragments(row, "AGE")
    assert "101yo" in text
    assert all(not name.startswith("Dr.") and not name.endswith(", MD") for name in fragments(row, "NAME"))
    condition_start = text.index("Parkinson disease")
    assert not any(s["start"] <= condition_start < s["end"] for s in row["spans"])
    assert sum(name == "Adriana Haldane" for name in fragments(row, "NAME")) == 2


def test_alias_precedes_owner_and_pronouns_stay_unannotated():
    row = _short_positive("Keiko Navarro", pair=0, variant=1, seed=7)
    assert row["cohort"] == "realistic"
    assert row["diagnostic_slice"] == "ordinary_name_contrast"
    assert fragments(row, "NAME")[0:2] == ["Hope", "Keiko Navarro"]
    assert "hope" not in fragments(row, "NAME")
    pronoun = row["text"].index("They")
    assert not any(s["start"] <= pronoun < s["end"] for s in row["spans"])
    assert fragments(row, "NAME").count("Hope") == 2
    pronoun_alias = _short_positive("Keiko Navarro", pair=84, variant=1, seed=7)
    assert pronoun_alias["cohort"] == "unusual_name"
    assert fragments(pronoun_alias, "NAME").count("They") == 2
    ordinary = pronoun_alias["text"].index("They described")
    assert not any(s["start"] <= ordinary < s["end"] for s in pronoun_alias["spans"])


def test_full_multiline_address_and_compact_written_age_boundaries():
    address_row = _short_positive("Inés Gutiérrez", pair=72, variant=0, seed=7)
    assert fragments(address_row, "ADDRESS") == ["290 Lantern Walk\nUnit 12\nLarkspur Hollow, Oregon 97072"]
    assert fragments(address_row, "LOCATION") == ["Cedar Quay, Wisconsin"]
    compact = _short_positive("Zara Petrov", pair=1, variant=0, seed=7)
    assert "92F" in compact["text"]
    assert fragments(compact, "AGE") == ["92"]
    written = _short_positive("Umar Carrillo", pair=6, variant=0, seed=7)
    assert fragments(written, "AGE") == ["one hundred and two"]
    ordinary_age = _short_positive("Zara Petrov", pair=18, variant=0, seed=7)
    assert "Patient age: 89" in ordinary_age["text"]
    assert fragments(ordinary_age, "AGE") == []
    assert "ordinary_age" in ordinary_age["phenomena"]


def test_negative_variation_scales_and_new_clinical_genres_are_present():
    negatives = [_negative(i, 72) for i in range(1000)]
    assert len({row["text"] for row in negatives}) == 1000
    assert all(not row["spans"] for row in negatives)
    assert all(int(row["identity_group"][:8], 16) % 10 in {8, 9} for row in negatives)
    assert len({row["layout"] for row in negatives}) == 4
    families = {row["family"] for row in build_rows(short_count=50, long_count=0)}
    assert {"psychiatric_review", "obstetric_followup", "dental_assessment", "genetics_counseling"} <= families


def test_long_rows_cross_all_requested_joins_and_retain_tail_occurrences():
    rows = build_rows(short_count=0, long_count=10)
    encoding = tiktoken.get_encoding("o200k_base")
    for row in rows:
        tokens = encoding.encode(row["text"])
        assert row["token_count"] == len(tokens)
        assert 1500 <= len(tokens) <= 6000
        _, offsets = encoding.decode_with_offsets(tokens)
        expected = [] if row["diagnostic_slice"] == "long_narrative" else sorted({512, *range(384, len(tokens), 256)})
        assert [check["token_boundary"] for check in row["boundary_checks"]] == expected
        for check in row["boundary_checks"]:
            boundary = offsets[check["token_boundary"]]
            assert check["span_start"] < boundary < check["span_end"]
            assert {"start": check["span_start"], "end": check["span_end"], "label": check["label"]} in row["spans"]
        assert any(span["start"] > len(row["text"]) * .95 for span in row["spans"])
        assert len(set(fragments(row, "NAME"))) == 1
        assert len(fragments(row, "NAME")) >= 4
        assert {s["label"] for s in row["spans"]} == set(ENTITY_TYPES)


def test_manifest_reproducibility_and_no_overwrite(tmp_path):
    destination = tmp_path / "development.jsonl"
    manifest = generate_corpus(destination, short_count=20, long_count=0, seed=72)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert generate_corpus(destination, short_count=20, long_count=0, seed=72) == manifest
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    rows, loaded = load_verified_corpus(destination)
    assert len(rows) == 20
    assert loaded == manifest
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == manifest["sha256"]
    declaration = json.loads(destination.with_suffix(".jsonl.author-declaration.json").read_text())
    assert declaration["used_for_checkpoint_selection"] is True
    assert declaration["used_for_training"] is False
    assert declaration["independent_final"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_corpus(destination, short_count=20, long_count=0, seed=73)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_sidecar_conflict_is_rejected_before_new_corpus_write(tmp_path):
    destination = tmp_path / "development.jsonl"
    destination.with_suffix(".jsonl.manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError):
        generate_corpus(destination, short_count=5, long_count=0)
    assert not destination.exists()


def test_identity_hash_normalizes_case_and_unicode():
    assert identity_group("Élodie Varela") == identity_group("E\u0301LODIE VARELA")


@pytest.mark.parametrize("short,long", [(0, 0), (-1, 1), (1, -1), (True, 0), (1.5, 0)])
def test_invalid_counts(short, long):
    with pytest.raises(ValueError):
        build_rows(short, long)
