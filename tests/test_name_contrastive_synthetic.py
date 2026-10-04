import copy
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import pytest
import tiktoken

from clinical_deid.data.generators import name_contrastive as generator
from clinical_deid.evaluation.evaluate import load_verified_corpus
from clinical_deid.types import ENTITY_TYPES


@pytest.fixture(scope="module")
def encoding():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    return tiktoken.get_encoding("o200k_base")


def _values(row, label="NAME"):
    return [
        row["text"][span["start"] : span["end"]]
        for span in row["spans"]
        if span["label"] == label
    ]


def _canonical(value):
    return unicodedata.normalize("NFC", value).casefold()


def _declared(path, rows, **extra):
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload)
    path.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "rows": len(rows),
                "split": "validation",
                "independent_final": False,
            }
            | extra
        )
    )


def test_authored_pairs_reproduce_exactly_and_keep_consistent_people():
    rows = [generator.generate_pair(index, seed=19) for index in range(100)]
    assert rows == [generator.generate_pair(index, seed=19) for index in range(100)]
    assert rows != [generator.generate_pair(index, seed=20) for index in range(100)]
    ids, texts = set(), set()
    for positive, negative in rows:
        assert positive["pair_id"] == negative["pair_id"]
        assert positive["person_role"] == negative["person_role"]
        assert len(_values(positive)) == 3
        assert len(set(_values(positive))) == 1
        assert positive["negative"] is False
        assert negative["negative"] is True and negative["spans"] == []
        assert positive["text"].splitlines()[0] == negative["text"].splitlines()[0]
        assert positive["text"].splitlines()[-1] == negative["text"].splitlines()[-1]
        for row in (positive, negative):
            generator.validate_row(row)
            assert row["id"] not in ids and row["text"] not in texts
            ids.add(row["id"])
            texts.add(row["text"])


@pytest.mark.parametrize("split,count", [("train", 2000), ("validation", 500)])
def test_actual_scale_has_broad_contexts_cases_layouts_and_public_contrasts(
    split, count
):
    human, captions, styles, layouts, facts, collisions = (
        set(),
        set(),
        set(),
        set(),
        set(),
        set(),
    )
    for index in range(count):
        positive, negative = generator.generate_pair(index, split=split)
        for context in positive["authored_contexts"]:
            if context.startswith("human-"):
                human.add(int(context.split("-")[1]))
            if context.startswith("caption-"):
                captions.add(int(context.split("-")[1]))
        styles.add(positive["authored_name_style"])
        layouts.add(positive["family"].split(".")[2])
        facts.add(positive["public_fact_context"])
        if positive["authored_name_style"].startswith("collision"):
            value = _values(positive)[0]
            collisions.add(value.casefold())
            assert value in negative["text"]
    assert human == set(range(len(generator.HUMAN_FRAMES[split])))
    assert captions == set(range(len(generator.CAPTIONS[split])))
    assert len(styles) == 11  # two lowercase collision slots share one style
    assert layouts == set(generator.LAYOUTS)
    assert facts == set(range(24))
    assert collisions == {value.casefold() for value in generator.COLLISIONS[split]}
    assert not set(generator.HUMAN_FRAMES["train"]) & set(
        generator.HUMAN_FRAMES["validation"]
    )
    assert not set(generator.CAPTIONS["train"]) & set(generator.CAPTIONS["validation"])


def test_private_development_values_and_ids_are_disjoint_after_unicode_normalization():
    values, ids, texts = {}, {}, {}
    for split, count in (("train", 2000), ("validation", 500)):
        values[split], ids[split], texts[split] = set(), set(), set()
        for index in range(count):
            for row in generator.generate_pair(index, split=split):
                ids[split].add(row["id"])
                texts[split].add(row["text"])
                for label in ("NAME", "LOCATION"):
                    values[split].update(
                        _canonical(value) for value in _values(row, label)
                    )
        assert len(ids[split]) == len(texts[split]) == 2 * count
    assert not values["train"] & values["validation"]
    assert not ids["train"] & ids["validation"]
    assert not texts["train"] & texts["validation"]


def test_unicode_middle_initial_suffix_and_possessive_boundaries_are_complete(encoding):
    counts, tags = Counter(), Counter()
    for index in range(240):
        row, _ = generator.generate_pair(index)
        length, support = generator.shared.token_audit(row, encoding)
        assert length < 400
        tags.update(support)
        values = _values(row)
        if index % 12 == 6:
            assert unicodedata.category(values[0][1]).startswith("M")
            assert values[0] != unicodedata.normalize("NFC", values[0])
            counts["nfd-leading"] += 1
        if index % 12 == 7:
            assert re.search(r"\s[A-Z][\u0300-\u036f]+\.\s", values[0])
            counts["nfd-middle-initial"] += 1
        if index % 12 in {4, 8, 10}:
            assert values[0].islower()
            counts["fully-lowercase"] += 1
        if index % 12 in {9, 10}:
            assert re.search(r"\b(?:jr\.?|sr\.?|ii|iii)$", values[0], re.I)
            last = [s for s in row["spans"] if s["label"] == "NAME"][-1]
            assert row["text"][last["end"] : last["end"] + 2] == "'s"
            assert not values[-1].endswith("'s")
            counts["suffix-possessive"] += 1
    assert counts == {
        "nfd-leading": 20,
        "nfd-middle-initial": 20,
        "fully-lowercase": 60,
        "suffix-possessive": 40,
    }
    assert tags["S-NAME"] > 200


def test_normal_age_doses_labs_durations_eponyms_and_public_dates_are_unlabelled():
    patterns = (
        r"(?:Na|Sodium|sodium) \d+ mmol/L",
        r"(?:BP|was) \d+/\d+ mmHg",
        r"glucose \d+ mg/dL",
        r"\d+ (?:mg|mcg)",
        r"\d+/30",
        r"\b5/5\b",
        r"\d+ days",
        r"(?:aged|is) \d+ years",
        r"Addison disease",
        r"Romberg testing",
        r"Graves disease",
    )
    hits = Counter()
    for index in range(150):
        positive, negative = generator.generate_pair(index)
        for pattern in patterns:
            for match in re.finditer(pattern, positive["text"]):
                hits[pattern] += 1
                assert all(
                    match.end() <= s["start"] or match.start() >= s["end"]
                    for s in positive["spans"]
                )
        reference = positive["text"].splitlines()[-1]
        reference_start = positive["text"].rfind("\n") + 1
        assert "Teaching examples" in reference
        assert all(s["end"] <= reference_start for s in positive["spans"])
        assert negative["spans"] == []
        assert positive["publication_year"] < positive["clinical_episode_year"]
    assert all(hits[pattern] for pattern in patterns)


def test_private_dates_keep_clinical_order_and_counterpart_contains_no_phi():
    for index in range(120):
        positive, negative = generator.generate_pair(index)
        dates = _values(positive, "DATE")
        if index % 4 == 1:
            assert len(dates) == 2
            assessment, review = map(date.fromisoformat, dates)
            assert (review - assessment).days == 28
            assert assessment.year == positive["clinical_episode_year"]
            assert dates[0] not in negative["text"] and dates[1] not in negative["text"]
        else:
            assert dates == []


def test_public_value_mentions_in_mixed_notes_are_not_relabelled_as_people():
    for index in range(0, 240, 4):
        positive, negative = generator.generate_pair(index)
        value = _values(positive)[0]
        assert value in negative["text"] and not negative["spans"]
        spans = [span for span in positive["spans"] if span["label"] == "NAME"]
        last_name_end = max(span["end"] for span in spans)
        assert value in positive["text"][last_name_end:]
        assert len(spans) == 3


def test_care_team_heading_and_pronoun_do_not_form_person_gold():
    found = set()
    for index in range(0, 600, 3):
        positive, negative = generator.generate_pair(index)
        if not positive["authored_name_style"].startswith("collision"):
            continue
        match = re.search(r"(?:Nurse|Clinician|Caregiver)\n", negative["text"])
        if match:
            found.add(match.group().strip())
            assert negative["spans"] == []
    assert found == {"Nurse", "Clinician", "Caregiver"}


@pytest.mark.parametrize(
    "mutation", ["overlap", "out-of-bounds", "label", "negative", "suite", "chronology"]
)
def test_corrupted_gold_or_role_metadata_is_rejected(mutation):
    row = copy.deepcopy(generator.generate_pair(0)[0])
    if mutation == "overlap":
        row["spans"][1]["start"] = row["spans"][0]["start"]
    elif mutation == "out-of-bounds":
        row["spans"][0]["end"] = len(row["text"]) + 1
    elif mutation == "label":
        row["spans"][0]["label"] = "DRUG"
    elif mutation == "negative":
        row["negative"] = True
    elif mutation == "suite":
        row["suite"] = "independent_final"
    else:
        row["publication_year"] = row["clinical_episode_year"]
    with pytest.raises(ValueError):
        generator.validate_row(row)


def test_manifests_match_exact_payload_and_actual_annotation_counts(tmp_path, encoding):
    compare = tmp_path / "approved.jsonl"
    _declared(
        compare,
        [
            {
                "id": "approved-private-id",
                "text": "Clinical example outside this corpus",
                "split": "validation",
            }
        ],
    )
    manifests = generator.write_dataset(
        tmp_path / "output", 240, 120, compare_corpora=[compare]
    )
    for split, suffix in (("train", "train"), ("validation", "validation")):
        path = tmp_path / "output" / f"name_contrastive_{suffix}.jsonl"
        rows, saved = load_verified_corpus(path)
        assert saved == manifests[split]
        assert (
            saved["source_sha256"]
            == hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()
        )
        assert (
            saved["shared_annotation_audit_helper_sha256"]
            == hashlib.sha256(Path(generator.shared.__file__).read_bytes()).hexdigest()
        )
        assert saved["license"] == "CC0-1.0" and saved["code_license"] == "Apache-2.0"
        assert saved["external_datasets"] == []
        assert saved["used_for_training"] is (split == "train")
        assert saved["used_for_checkpoint_selection"] is (split == "validation")
        assert saved["independent_final"] is False
        lengths, tags, support = [], Counter(), Counter()
        for row in rows:
            length, row_tags = generator.shared.token_audit(row, encoding)
            lengths.append(length)
            tags.update(row_tags)
            support.update(span["label"] for span in row["spans"])
        assert saved["bioes_tag_support"] == tags
        assert saved["label_support"] == {
            label: support[label] for label in ENTITY_TYPES
        }
        assert saved["token_length"]["max"] == max(lengths) <= 512
        assert saved["token_length"]["mean"] == sum(lengths) / len(lengths)
        assert saved["phi_free_notes"] == len(rows) // 2
        assert saved["contrast_pairs"] == len(rows) // 2
        assert saved["name_span_traits"]["total"] == 3 * len(rows) // 2
        assert saved["comparison_audit"][0]["exact_text_overlap"] == 0
        assert saved["comparison_audit"][0]["exact_id_overlap"] == 0
        assert "text_hashes" not in saved["comparison_audit"][0]
        assert sum(saved["public_fact_note_support"].values()) == len(rows)
        assert saved["cross_split_audit"] == {
            "id_overlap": 0,
            "exact_text_overlap": 0,
            "name_location_value_overlap": 0,
        }


def test_published_files_are_not_overwritten_and_seed_reproduction_matches(tmp_path):
    first = generator.write_dataset(tmp_path / "first", 48, 24, seed=45)
    second = generator.write_dataset(tmp_path / "second", 48, 24, seed=45)
    assert first == second
    path = tmp_path / "first" / "name_contrastive_train.jsonl"
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        generator.write_dataset(tmp_path / "first", 48, 24, seed=99)
    assert path.read_bytes() == before


@pytest.mark.parametrize("collision", ["text", "id"])
def test_approved_corpus_text_or_id_overlap_is_rejected_without_publication(
    tmp_path, collision
):
    row = generator.generate_pair(0)[0]
    comparison = tmp_path / "approved.jsonl"
    _declared(
        comparison,
        [
            {
                "id": row["id"] if collision == "id" else "unique-prior-id",
                "text": row["text"]
                if collision == "text"
                else "No matching authored clinical text",
                "split": "validation",
            }
        ],
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps"):
        generator.write_dataset(output, 4, 2, compare_corpora=[comparison])
    assert not list(output.iterdir())


def test_qualification_is_rejected_before_any_payload_read(tmp_path, monkeypatch):
    approved = tmp_path / "approved.jsonl"
    _declared(
        approved,
        [
            {
                "id": "allowed-id",
                "text": "Ordinary clinical review",
                "split": "validation",
            }
        ],
    )
    sealed = tmp_path / "not-created-qualified-payload.jsonl"
    sealed.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps({"split": "test", "independent_final": True})
    )
    original, reads = Path.read_bytes, []

    def spy(path):
        if path.suffix == ".jsonl":
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", spy)
    with pytest.raises(ValueError, match="Qualification"):
        generator.write_dataset(
            tmp_path / "output", 4, 2, compare_corpora=[approved, sealed]
        )
    assert reads == []


@pytest.mark.parametrize("count", [0, 1, 3, -2, True])
def test_note_counts_require_positive_complete_contrast_pairs(tmp_path, count):
    with pytest.raises(ValueError, match="positive even"):
        generator.write_dataset(tmp_path, count, 2)
