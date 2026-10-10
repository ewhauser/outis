import hashlib
import json
import re
from collections import Counter, defaultdict

import pytest

from outis.data.annotation_policy import (
    canonical_identity_group,
    identity_split,
)
from outis.data.corpus_quality import audit_corpora
from outis.data.v12 import synthetic as generator
from outis.types import ENTITY_TYPES


@pytest.fixture(scope="module")
def rows():
    return generator.build_corpus(short_count=1600, long_count=2)


def test_corpus_is_deterministic_and_every_secondary_owner_is_grouped(rows):
    assert rows[:20] == generator.build_corpus(short_count=100, long_count=0)[:20]
    assert len({r["text"] for r in rows}) == len(rows)
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        generator.validate_row(row)
        assert {
            canonical_identity_group(name) for name in row["full_identities"]
        } == set(row["identity_groups"])
        assert all(identity_split(group) == "train" for group in row["identity_groups"])
    pairs = defaultdict(list)
    for row in rows:
        if "context_pair" in row["phenomena"]:
            pairs[row["pair_id"]].append(row)
    assert pairs
    for positive, negative in pairs.values():
        assert positive["identity_group"] == negative["identity_group"]
        assert positive["family"] == negative["family"]
        assert positive["spans"] and not negative["spans"]


def test_all_years_have_patient_and_public_support_and_real_same_value_contrasts(rows):
    patient, public, contrasts = Counter(), Counter(), 0
    for row in rows:
        year_positive = set()
        for span in row["spans"]:
            if span["label"] == "DATE":
                found = re.findall(
                    r"\b(?:19|20)\d{2}\b", row["text"][span["start"] : span["end"]]
                )
                patient.update(found)
                year_positive.update(found)
        year_negative = set()
        for loc in row["phenomenon_evidence"].get("public_dates", []):
            assert not any(
                s["start"] < loc["end"] and s["end"] > loc["start"]
                for s in row["spans"]
            )
            found = re.findall(
                r"\b(?:19|20)\d{2}\b", row["text"][loc["start"] : loc["end"]]
            )
            public.update(found)
            year_negative.update(found)
        if "same_year_patient_public" in row["phenomena"]:
            assert year_positive & year_negative
            contrasts += 1
    assert set(patient) == set(public) == {str(y) for y in generator.YEARS}
    assert contrasts > 300


def test_metadata_has_actual_offsets_and_no_claimed_context_without_text(rows):
    for row in rows:
        assert sorted(row["phenomenon_evidence"]) == row["phenomena"]
        for loc in row["phenomenon_evidence"].get("clinical_durations", []):
            assert re.search(r"\d+ days", row["text"][loc["start"] : loc["end"]])
        for loc in row["phenomenon_evidence"].get("ordinary_pronouns", []):
            assert re.search(
                r"\b(?:They|She|He|We|I|You)\b", row["text"][loc["start"] : loc["end"]]
            )
        assert not re.search(
            r"\b(?:PHI|non-PHI|gold label|should be labeled)\b", row["text"], re.I
        )


def test_name_and_age_annotations_retain_clinical_occurrence_boundaries(rows):
    elderly, unicode_names, clinician_names = 0, 0, 0
    for row in rows:
        text = row["text"]
        for span in row["spans"]:
            value = text[span["start"] : span["end"]]
            if span["label"] == "NAME":
                assert not value.startswith("Dr. ") and not value.endswith(
                    (", MD", ", DO")
                )
                unicode_names += not value.isascii()
            elif span["label"] == "AGE":
                assert (
                    value.isdigit()
                    and 90 <= int(value) <= 100
                    or value.startswith("ninety")
                    or value == "one hundred"
                )
                assert not re.search(r"year|[FM]$|y/o|yo", value)
                elderly += 1
        for match in re.finditer(r"\b(?:9\d|100)-year-old\b", text):
            assert any(
                s["label"] == "AGE" and s["start"] == match.start()
                for s in row["spans"]
            )
        ages = set(re.findall(r"\b(\d+)-year-old\b", text))
        # A note never silently mixes different patient ages.
        assert len(ages) <= 1
        clinician_names += "clinician_name" in row["phenomena"]
    assert elderly > 100 and unicode_names > 100 and clinician_names > 100


def test_reserved_identifiers_and_nested_address_annotations(rows):
    counts = Counter()
    for row in rows:
        for span in row["spans"]:
            label, value = span["label"], row["text"][span["start"] : span["end"]]
            counts[label] += 1
            if label == "PHONE":
                assert re.search(r"555[-. ]01\d\d", value)
            elif label == "EMAIL":
                assert value.endswith(
                    (".test", ".example", "example.org", "example.net", "example.com")
                )
            elif label == "SSN":
                assert value.startswith("000-")
            elif label == "IP":
                assert value.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
            elif label == "ADDRESS":
                assert not any(
                    other is not span and span["start"] <= other["start"] < span["end"]
                    for other in row["spans"]
                )
    assert set(counts) == set(ENTITY_TYPES)
    assert min(counts.values()) > 100


def test_stress_cohorts_and_long_token_metadata_are_explicit(rows):
    assert sum(not row["spans"] for row in rows) == 320
    assert Counter(r["cohort"] for r in rows) == {
        "realistic": 1560,
        "unusual_name": 40,
        "long_document": 2,
    }
    for row in rows:
        if row["cohort"] == "unusual_name":
            assert "explicit_unusual_name" in row["phenomena"]
        if row["cohort"] == "long_document":
            assert row["diagnostic_slice"] == "repeated_course_narrative"
            assert row["token_audit"]["token_count"] == len(
                generator.get_encoding().encode(row["text"], disallowed_special=())
            )
            assert row["token_audit"]["token_count"] > 1000
            assert "training_seam" not in row["phenomena"]


def test_writer_is_immutable_and_auditor_accepts_its_manifest(tmp_path):
    path = tmp_path / "training.jsonl"
    small = generator.build_corpus(short_count=100, long_count=0)
    manifest = generator.write_corpus(path, small)
    before = path.read_bytes()
    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["external_datasets"] == []
    audit = audit_corpora({"train": path})
    assert audit["passed"], audit["errors"]
    with pytest.raises(FileExistsError):
        generator.write_corpus(path, small)
    assert path.read_bytes() == before
    assert json.loads(path.with_suffix(".jsonl.manifest.json").read_text()) == manifest


def test_claimed_same_year_evidence_is_validated():
    row = generator.short_row(11)
    row["phenomenon_evidence"]["same_year_patient_public"] = [{"start": 0, "end": 1}]
    row["phenomena"] = sorted(row["phenomenon_evidence"])
    row["phenomenon_evidence"].pop("public_dates", None)
    row["phenomena"] = sorted(row["phenomenon_evidence"])
    with pytest.raises(ValueError, match="same-year"):
        generator.validate_row(row)
