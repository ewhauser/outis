import copy
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import tiktoken

from outis.data.generators import generalization as generator
from outis.types import ENTITY_TYPES


@pytest.fixture(scope="module")
def encoding():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    return tiktoken.get_encoding("o200k_base")


def test_original_authored_corpus_is_reproducible_and_training_only():
    first = [generator.generate_row(index, seed=67) for index in range(80)]
    assert first == [generator.generate_row(index, seed=67) for index in range(80)]
    assert first != [generator.generate_row(index, seed=68) for index in range(80)]
    assert len({row["id"] for row in first}) == len(first)
    assert len({row["text"] for row in first}) == len(first)
    for row in first:
        generator.validate_row(row)
        assert row["split"] == "train"
        assert row["suite"] == "authored_generalization_training"
        assert row["negative"] == (not row["spans"])


def test_production_scale_has_many_phi_free_notes_and_diverse_clinical_headers():
    negatives, headings, families, profiles = 0, set(), set(), set()
    contexts = defaultdict(set)
    for index in range(8000):
        row = generator.generate_row(index)
        generator.validate_row(row)
        negatives += row["negative"]
        headings.add(row["text"].splitlines()[0])
        families.add(row["family"])
        profiles.add(row["clinical_profile"])
        for context in row["authored_contexts"]:
            contexts[context["label"]].add(context["frame"])
    assert negatives == 1280
    assert len(headings) > 1000
    assert len(families) == 24
    assert profiles == set(range(len(generator.CLINICAL_PROFILES)))
    assert set(contexts) == set(ENTITY_TYPES)
    for label in ENTITY_TYPES:
        assert {f"narrative-{i}" for i in range(16)} <= contexts[label]
        assert {f"field-{i}" for i in range(8)} <= contexts[label]


def test_phi_free_clinical_information_is_unlabelled_despite_numeric_distractors():
    rows = [generator.generate_row(index) for index in range(300) if index % 25 < 4]
    assert all(not row["spans"] for row in rows)
    text = "\n".join(row["text"] for row in rows)
    assert re.search(r"The patient is (?:[1-8][0-9]) years old", text)
    assert re.search(r"(?:9[0-9]|1[01][0-9])-day monitoring interval", text)
    assert "Monitoring over one hundred" in text
    for phrase in (
        "mmHg",
        "mg/dL",
        "5/5",
        "/30",
        "Patient name: Unknown",
        "NDC",
        "PMID",
        "catalogue",
        "published",
        "Wilson disease",
        "We checked",
        "I reviewed",
        "Teach-back:",
        "A/P:",
        "Monofilament",
        "Presyncope",
    ):
        assert phrase in text
    assert "\nHistory:" in text
    assert re.search(r"\n\d+\. ", text)
    assert "\n- " in text


def test_team_pronouns_soap_and_clinical_action_grammar_are_unlabelled_in_mixed_notes():
    hits = Counter()
    for index in range(800):
        row = generator.generate_row(index)
        if row["negative"]:
            continue
        for phrase in generator.CLINICAL_LANGUAGE:
            start = row["text"].find(phrase)
            if start < 0:
                continue
            hits[phrase] += 1
            end = start + len(phrase)
            assert all(
                end <= span["start"] or start >= span["end"] for span in row["spans"]
            )
    assert sum(bool(hits[phrase]) for phrase in generator.CLINICAL_LANGUAGE) >= 20


def test_clinical_terms_and_public_references_stay_outside_mixed_note_gold():
    patterns = (
        r"BP \d+/\d+ mmHg",
        r"glucose \d+ mg/dL",
        r"creatinine \d\.\d mg/dL",
        r"PMID \d+",
        r"NCT\d+",
        r"MED-\d+",
        r"DEV-\d+",
        r"NDC [\d-]+",
        r"Wilson disease",
        r"Parkinson disease",
        r"Baker cyst",
        r"published in \d{4}",
        r"manual revised in \w+ \d{4}",
        r"(?:power|motor power) 5/5",
        r"(?:cognitive screening|screening) \d+/30",
        r"\d+-day monitoring interval",
    )
    hits = Counter()
    for index in range(400):
        row = generator.generate_row(index)
        for pattern in patterns:
            for match in re.finditer(pattern, row["text"], flags=re.IGNORECASE):
                hits[pattern] += 1
                assert all(
                    match.end() <= span["start"] or match.start() >= span["end"]
                    for span in row["spans"]
                )
    assert all(hits[pattern] for pattern in patterns)


def test_builder_owns_original_unicode_offsets_and_repeated_values():
    note = generator.Note()
    value = "A\u0301sa O’Alder"
    note.plain("Clinical history:\n")
    note.clause("Patient {v} supplied the history.", "NAME", value)
    note.plain("\n")
    note.clause("The clinician {v} reviewed the chart.", "NAME", value)
    assert len(note.spans) == 2
    assert note.spans[0]["end"] < note.spans[1]["start"]
    assert all(note.text[span["start"] : span["end"]] == value for span in note.spans)
    assert unicodedata.normalize("NFC", note.text) != note.text
    with pytest.raises(ValueError, match="one marked value"):
        note.clause("{v} and {v}", "NAME", value)
    suffix = generator.Note()
    suffix.clause("Full name is {v}.", "NAME", "A. Alder Jr.")
    assert suffix.text == "Full name is A. Alder Jr."
    assert (
        suffix.text[suffix.spans[0]["start"] : suffix.spans[0]["end"]] == "A. Alder Jr."
    )


def test_invalid_or_overlapping_gold_is_rejected():
    row = generator.generate_row(9)
    for bad in (
        {"start": -1, "end": 2, "label": "NAME"},
        {"start": 0, "end": len(row["text"]) + 1, "label": "NAME"},
        {"start": 1, "end": 1, "label": "NAME"},
        {"start": 0, "end": 2, "label": "UNKNOWN"},
    ):
        broken = copy.deepcopy(row)
        broken["spans"] = [bad]
        with pytest.raises(ValueError, match="ordered disjoint"):
            generator.validate_row(broken)
    broken = copy.deepcopy(row)
    broken["spans"].insert(1, dict(broken["spans"][0]))
    with pytest.raises(ValueError, match="ordered disjoint"):
        generator.validate_row(broken)
    broken = copy.deepcopy(row)
    broken["split"] = "validation"
    with pytest.raises(ValueError, match="exclusively"):
        generator.validate_row(broken)


def test_projection_has_all_entity_types_and_singleton_training_support(encoding):
    tags, labels, lengths = Counter(), Counter(), []
    for index in range(500):
        row = generator.generate_row(index)
        length, row_tags = generator.token_audit(row, encoding)
        assert sum(row_tags.values()) == length
        tags.update(row_tags)
        labels.update(span["label"] for span in row["spans"])
        lengths.append(length)
        for span in row["spans"]:
            value = row["text"][span["start"] : span["end"]]
            if span["label"] == "AGE":
                assert (value.isdigit() and int(value) > 89) or value.startswith(
                    ("ninety", "one hundred")
                )
            if span["label"] in {"URL", "EMAIL"}:
                assert ".example" in value
    assert set(labels) == set(ENTITY_TYPES)
    assert tags["S-NAME"] > 5 and tags["S-LOCATION"] > 5
    assert max(lengths) <= 512


def test_identifier_morphology_is_shared_but_clinical_type_depends_on_role():
    rng = random.Random(781)
    values = [generator.identity_values(rng) for _ in range(300)]
    shapes = defaultdict(set)
    for value in values:
        assert value["LOCATION"] in value["ADDRESS"]
        for label in ("MRN", "ACCOUNT", "ID"):
            shapes[label].add(re.sub(r"[A-Z]", "L", re.sub(r"\d", "D", value[label])))
    assert shapes["MRN"] == shapes["ACCOUNT"] == shapes["ID"]
    assert len(shapes["MRN"]) == 8
    assert any(
        " and " in value["AGE"] or value["AGE"].startswith("ninety") for value in values
    )
    assert any(
        "\u0301" in value["NAME"] or "\u0308" in value["NAME"] for value in values
    )
    assert any(re.search(r"(?:Jr\.|Sr\.| II| III)$", value["NAME"]) for value in values)


def test_locality_frames_require_geographic_relations():
    assert "residential district of {v}" in generator.FRAMES["LOCATION"][7]
    assert "community of {v}" in generator.FRAMES["LOCATION"][8]


def _declared(
    path, rows, *, suite="authored_test_development", split="validation", final=False
):
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    manifest = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "suite": suite,
        "split": split,
        "independent_final": final,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(
    "declaration",
    [
        {"suite": "final"},
        {"suite": "test"},
        {"split": "test"},
        {"independent_final": True},
    ],
)
def test_qualification_declaration_is_rejected_before_any_payload_read(
    tmp_path, monkeypatch, declaration
):
    regular = tmp_path / "declared-development.jsonl"
    sealed = tmp_path / "never-open-this.jsonl"
    _declared(
        regular, [{"text": "Public clinical observations", "split": "validation"}]
    )
    sealed.with_suffix(".jsonl.manifest.json").write_text(json.dumps(declaration))
    original = Path.read_bytes
    reads = []

    def spy(path):
        if path.suffix == ".jsonl":
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", spy)
    with pytest.raises(ValueError, match="Qualification"):
        generator.comparison_hashes([regular, sealed])
    assert reads == []


def test_declared_corpus_hashes_and_exact_text_overlap_are_checked(tmp_path):
    compare = tmp_path / "declared.jsonl"
    row = generator.generate_row(0)
    _declared(compare, [row], split="train", suite="authored_test_training")
    audit = generator.comparison_hashes([compare])[0]
    assert audit["text_hashes"] == {hashlib.sha256(row["text"].encode()).hexdigest()}
    with pytest.raises(ValueError, match="overlaps"):
        generator.write_dataset(
            tmp_path / "output", train_count=1, compare_corpora=[compare]
        )
    assert not (tmp_path / "output" / "generalization_train.jsonl").exists()
    assert not (tmp_path / "output" / "generalization_train.jsonl.tmp").exists()
    compare.write_text("changed payload")
    with pytest.raises(ValueError, match="declaration"):
        generator.comparison_hashes([compare])


def test_legacy_authored_aggregate_declaration_is_verified(tmp_path):
    path = tmp_path / "train.jsonl"
    payload = (
        json.dumps({"text": "Clinical measurements only", "split": "train"}) + "\n"
    )
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    manifest = {
        "datasets": {"train": {"file": "train.jsonl", "count": 1, "sha256": digest}}
    }
    aggregate = tmp_path / "manifest.json"
    aggregate.write_text(json.dumps(manifest))
    record = generator.comparison_hashes([path])[0]
    assert record["sha256"] == digest and record["rows"] == 1
    assert record["declaration"] == str(aggregate.resolve())
    manifest["datasets"]["train"]["count"] = 2
    aggregate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count"):
        generator.comparison_hashes([path])


def test_manifest_matches_immutable_payload_and_actual_annotation_audit(
    tmp_path, encoding
):
    compare = tmp_path / "declared.jsonl"
    _declared(
        compare,
        [
            {
                "text": "Public observation outside this authored corpus",
                "split": "validation",
            }
        ],
    )
    output = tmp_path / "output"
    manifest = generator.write_dataset(
        output, train_count=160, compare_corpora=[compare]
    )
    payload = output / "generalization_train.jsonl"
    rows = [json.loads(line) for line in payload.read_text().splitlines()]
    assert len(rows) == manifest["rows"] == 160
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest["sha256"]
    assert (
        json.loads(payload.with_suffix(".jsonl.manifest.json").read_text()) == manifest
    )
    assert (
        manifest["source_sha256"]
        == hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()
    )
    assert manifest["split"] == "train" and manifest["license"] == "CC0-1.0"
    assert (
        manifest["code_license"] == "Apache-2.0" and manifest["external_datasets"] == []
    )
    assert (
        manifest["used_for_training"] is True and manifest["independent_final"] is False
    )
    assert manifest["phi_free_notes"] == sum(not row["spans"] for row in rows)
    lengths, tags, labels = [], Counter(), Counter()
    for row in rows:
        generator.validate_row(row)
        length, support = generator.token_audit(row, encoding)
        lengths.append(length)
        tags.update(support)
        labels.update(span["label"] for span in row["spans"])
    assert manifest["token_length"]["max"] == max(lengths) <= 512
    assert manifest["token_length"]["mean"] == sum(lengths) / len(lengths)
    assert manifest["bioes_tag_support"] == tags
    assert manifest["label_support"] == labels
    assert manifest["comparison_audit"][0]["exact_text_overlap"] == 0
    assert "text_hashes" not in manifest["comparison_audit"][0]
    original = payload.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        generator.write_dataset(output, train_count=160)
    assert payload.read_bytes() == original


def test_age_word_values_are_bounded_and_accurate():
    assert generator.age_words(90) == "ninety"
    assert generator.age_words(99) == "ninety-nine"
    assert generator.age_words(100) == "one hundred"
    assert generator.age_words(104) == "one hundred and four"
    assert generator.age_words(110) == "one hundred and ten"
    assert generator.age_words(119) == "one hundred and nineteen"
    for age in (89, 120, -1, True):
        with pytest.raises(ValueError, match="90 to 119"):
            generator.age_words(age)


def test_public_reference_year_cannot_follow_current_clinical_episode():
    rng = random.Random(6918)
    for _ in range(50):
        values = generator.identity_values(rng)
        for index in (0, 6, 14):
            control = generator.public_control(
                rng, index, max_year=values["episode_year"]
            )
            years = re.findall(r"\b(?:19|20)\d{2}\b", control)
            assert years and all(int(year) <= values["episode_year"] for year in years)


def test_invalid_generation_count_and_token_ceiling_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        generator.write_dataset(tmp_path, train_count=0)
    with pytest.raises(ValueError, match="ceiling"):
        generator.write_dataset(tmp_path, max_tokens=99)
