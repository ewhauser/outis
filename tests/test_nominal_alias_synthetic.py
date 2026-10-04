import copy
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

import pytest
import tiktoken

from clinical_deid.data.generators import nominal_alias as generator
from clinical_deid.evaluation.evaluate import load_verified_corpus


@pytest.fixture(scope="module")
def encoding():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    return tiktoken.get_encoding("o200k_base")


def _names(row):
    return [
        row["text"][span["start"] : span["end"]]
        for span in row["spans"]
        if span["label"] == "NAME"
    ]


def _declared(path, rows, **metadata):
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode()
    path.write_bytes(payload)
    path.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": len(rows),
                "split": "validation",
                "independent_final": False,
            }
            | metadata
        )
    )


def test_pair_reproducibility_and_controlled_anchor_ownership():
    first = [generator.generate_pair(index, seed=91) for index in range(90)]
    assert first == [generator.generate_pair(index, seed=91) for index in range(90)]
    assert first != [generator.generate_pair(index, seed=92) for index in range(90)]
    for positive, counterfactual in first:
        assert positive["pair_id"] == counterfactual["pair_id"]
        assert positive["person_identity"] == counterfactual["person_identity"]
        assert positive["other_identity"] == counterfactual["other_identity"]
        assert not positive["negative"] and not counterfactual["negative"]
        primary = positive["person_identity"]
        assert _names(positive)[0] == _names(counterfactual)[0] == primary["full"]
        assert primary["first"] in _names(positive)
        assert primary["surname"] in _names(positive)
        assert primary["first"] not in _names(counterfactual)
        assert primary["surname"] not in _names(counterfactual)
        for row in (positive, counterfactual):
            generator.validate_row(row)


def test_counterfactual_preserves_different_person_private_mentions():
    for index in range(80):
        positive, counterfactual = generator.generate_pair(index)
        other = positive["other_identity"]
        if other is None:
            continue
        assert other["full"] != positive["person_identity"]["full"]
        assert other["surname"] != positive["person_identity"]["surname"]
        assert other["full"] in _names(positive) and other["full"] in _names(
            counterfactual
        )
        assert other["surname"] in _names(positive) and other["surname"] in _names(
            counterfactual
        )


def test_shared_alias_supervision_declares_both_owners_without_inventing_identity():
    hits = 0
    for index in range(90):
        positive, counterfactual = generator.generate_pair(index)
        if not positive["shared_first_alias"]:
            continue
        hits += 1
        person, other = positive["person_identity"], positive["other_identity"]
        assert person["first"] == other["first"]
        assert person["full"] != other["full"]
        assert positive["alias_context"] < 18
        assert {"person": "shared", "kind": "first"} in positive["name_mentions"]
        assert all(
            mention["person"] != "shared" for mention in counterfactual["name_mentions"]
        )
    assert hits >= 7


@pytest.mark.parametrize("split,count", [("train", 500), ("validation", 125)])
def test_production_context_coverage_includes_bare_function_nominals(split, count):
    captions, frames, common, profiles, layouts, reference = (set() for _ in range(6))
    bare_common, pronoun_nominal = 0, 0
    for index in range(count):
        positive, counterfactual = generator.generate_pair(index, split=split)
        captions.add(positive["caption_context"])
        frames.add(positive["alias_context"])
        profiles.add(positive["clinical_profile"])
        layouts.add(positive["family"].split(".")[2])
        reference.add(positive["public_reference_context"])
        person = positive["person_identity"]
        if person["common"]:
            common.add(generator.canonical(person["first"]))
            assert person["first"] in counterfactual["text"]
            bare_common += positive["alias_context"] < 18
        if person["pronoun"]:
            pronoun_nominal += 1
            assert positive["alias_context"] in generator._frame_choices(person, split)
    assert captions == set(range(len(generator.CAPTIONS[split])))
    assert frames == set(range(24))
    assert profiles == set(range(24))
    assert layouts == set(generator.LAYOUTS)
    assert reference == set(range(8))
    assert common == {value.casefold() for value in generator.COMMON_NAMES}
    assert bare_common > count // 4 and pronoun_nominal > count // 10
    assert not set(generator.CAPTIONS["train"]) & set(generator.CAPTIONS["validation"])
    assert not set(generator.NOMINAL_FRAMES["train"]) & set(
        generator.NOMINAL_FRAMES["validation"]
    )


def test_bare_pronoun_subjects_remain_unlabelled_despite_identity_anchor():
    seen = set()
    for index in range(240):
        for row in generator.generate_pair(index):
            person = row["person_identity"]
            if not person["pronoun"]:
                continue
            seen.add(generator.canonical(person["first"]))
            assert row["alias_context"] >= 18 or (
                generator.canonical(person["first"]) in generator.SUBJECT_ONLY_PRONOUNS
                and row["alias_context"] in generator.OBJECT_NOMINAL_FRAMES
            )
            assert row["alias_context"] not in {20, 21}
            ordinary = [
                clause
                for clause in row["nonpersonal_clauses"]
                if clause["interpretation"] == "ordinary_function_or_pronoun"
            ]
            assert ordinary
            assert all(
                person["first"] in row["text"][clause["start"] : clause["end"]]
                for clause in ordinary
            )
            assert all(
                clause["end"] <= span["start"] or clause["start"] >= span["end"]
                for clause in ordinary
                for span in row["spans"]
            )
    assert seen == {value.casefold() for value in generator.PRONOUN_NAMES}


@pytest.mark.parametrize(
    "split,count,minimum_s", [("train", 500, 650), ("validation", 125, 140)]
)
def test_unanchored_short_human_names_supply_real_singleton_tags(
    split, count, minimum_s, encoding
):
    tags, names, frames = Counter(), set(), set()
    for index in range(count):
        positive, negative = generator.generate_unanchored_pair(index, split=split)
        assert positive["person_identity"] == negative["person_identity"]
        assert positive["person_identity"]["full"] is None
        assert positive["other_identity"] is None
        assert _names(positive) == [positive["person_identity"]["first"]] * 2
        assert negative["spans"] == negative["name_mentions"] == []
        assert positive["pair_id"] == negative["pair_id"]
        assert negative["negative"] and not positive["negative"]
        names.add(positive["person_identity"]["first"])
        frames.update((positive["alias_context"], positive["second_alias_context"]))
        length, counts = generator.audit.token_audit(positive, encoding)
        tags.update(counts)
        assert length <= 512
        if positive["person_identity"]["common"]:
            assert (
                positive["alias_context"] >= 18
                and positive["second_alias_context"] >= 18
            )
        if positive["person_identity"]["pronoun"]:
            assert {
                positive["alias_context"],
                positive["second_alias_context"],
            }.isdisjoint({20, 21})
        generator.validate_row(positive)
        generator.validate_row(negative)
    assert tags["S-NAME"] >= minimum_s
    assert tags["S-NAME"] + tags["B-NAME"] == count * 2
    assert len(names) > count // 2
    assert frames == set(range(24))


def test_unanchored_negatives_preserve_same_short_spelling_as_public_or_ordinary_language():
    for index in range(150):
        positive, negative = generator.generate_unanchored_pair(index)
        value = positive["person_identity"]["first"]
        assert value in negative["text"]
        assert negative["spans"] == []
        clinical_end = positive["text"].index(
            "S: comfort at rest. O: controlled movement. A/P: continue paced activity."
        ) + len(
            "S: comfort at rest. O: controlled movement. A/P: continue paced activity."
        )
        assert positive["text"][:clinical_end] == negative["text"][:clinical_end]
        assert (
            positive["text"].split("Clinical team observations: ")[-1]
            == negative["text"].split("Clinical team observations: ")[-1]
        )


def test_name_styles_and_unicode_boundaries_are_preserved_in_gold(encoding):
    traits = Counter()
    for index in range(150):
        for row in generator.generate_pair(index) + generator.generate_unanchored_pair(
            index
        ):
            length, _ = generator.audit.token_audit(row, encoding)
            assert length < 350
            for value in _names(row):
                traits["lower"] += value.islower()
                traits["unicode"] += not value.isascii()
                traits["nfd"] += value != unicodedata.normalize("NFC", value)
                traits["comma"] += ", " in value
                traits["suffix"] += bool(
                    re.search(r"\b(?:Jr\.|Sr\.|II|III)(?:,|$)", value)
                )
                traits["particle"] += bool(re.search(r"\b(?:de|van|al|von)\b", value))
                traits["middle_initial"] += bool(re.search(r"\s[^\s,]+\.\s", value))
            for span in row["spans"]:
                assert not unicodedata.category(row["text"][span["start"]]).startswith(
                    "M"
                )
                assert span["end"] == len(row["text"]) or not unicodedata.category(
                    row["text"][span["end"]]
                ).startswith("M")
    assert all(
        traits[trait] >= 10
        for trait in (
            "lower",
            "unicode",
            "nfd",
            "comma",
            "suffix",
            "particle",
            "middle_initial",
        )
    )


def test_clinical_values_and_public_reference_dates_are_never_annotations():
    patterns = (
        r"\d+/\d+ mmHg",
        r"\d+ mmol/L",
        r"\d+ mg/dL",
        r"\d+ days",
        r"adult aged \d+ years",
        r"\b5/5\b",
        r"\b28/30\b",
        r"Babinski testing",
        r"Baker cyst",
        r"Addison disease",
    )
    hits = Counter()
    for index in range(100):
        for row in generator.generate_pair(index) + generator.generate_unanchored_pair(
            index
        ):
            for exclusion in row["nonpersonal_clauses"]:
                assert all(
                    exclusion["end"] <= span["start"]
                    or exclusion["start"] >= span["end"]
                    for span in row["spans"]
                )
            for pattern in patterns:
                for match in re.finditer(pattern, row["text"]):
                    hits[pattern] += 1
                    assert all(
                        match.end() <= span["start"] or match.start() >= span["end"]
                        for span in row["spans"]
                    )
    assert all(hits[pattern] for pattern in patterns)


def test_private_dates_preserve_fourteen_day_intervals_and_prior_public_years():
    for index in range(72):
        positive, counterfactual = generator.generate_pair(index)
        for row in (positive, counterfactual):
            dates = [
                row["text"][span["start"] : span["end"]]
                for span in row["spans"]
                if span["label"] == "DATE"
            ]
            assert row["publication_year"] < row["clinical_episode_year"]
            if index % 6 == 0:
                first, second = map(date.fromisoformat, dates)
                assert (second - first).days == 14
                assert first.year == row["clinical_episode_year"]
            else:
                assert not dates


@pytest.mark.parametrize(
    "mutation",
    [
        "offset",
        "overlap",
        "label",
        "owner",
        "public",
        "unicode",
        "pronoun",
        "split",
        "negative",
    ],
)
def test_invalid_annotation_geometry_meaning_and_declarations_fail(mutation):
    row = copy.deepcopy(generator.generate_pair(9)[0])
    if mutation == "offset":
        row["spans"][0]["end"] = len(row["text"]) + 1
    elif mutation == "overlap":
        row["spans"][1]["start"] = row["spans"][0]["start"]
    elif mutation == "label":
        row["spans"][0]["label"] = "PRIVATE"
    elif mutation == "owner":
        row["name_mentions"][0]["kind"] = "first"
    elif mutation == "public":
        row["nonpersonal_clauses"][0]["start"] = row["spans"][0]["start"]
    elif mutation == "unicode":
        row["spans"][0]["start"] += 1
    elif mutation == "pronoun":
        row = copy.deepcopy(generator.generate_pair(5)[0])
        row["alias_context"] = 0
    elif mutation == "split":
        row["split"] = "test"
    else:
        row["negative"] = True
    with pytest.raises(ValueError):
        generator.validate_row(row)


def test_manifest_matches_actual_payload_tags_and_exact_quarter_distributions(
    tmp_path, encoding
):
    manifests = generator.write_dataset(tmp_path, train_count=80, val_count=40, seed=31)
    all_ids, all_text, full, aliases = set(), set(), {}, {}
    for split, count in (("train", 80), ("validation", 40)):
        path = (
            tmp_path
            / f"nominal_alias_{'train' if split == 'train' else 'validation'}.jsonl"
        )
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_bytes())
        assert manifest == manifests[split]
        assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
        assert (
            manifest["source_sha256"]
            == hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()
        )
        assert (
            manifest["license"] == "CC0-1.0"
            and manifest["code_license"] == "Apache-2.0"
        )
        assert (
            manifest["external_datasets"] == []
            and manifest["independent_final"] is False
        )
        assert manifest["used_for_training"] is (split == "train")
        assert manifest["used_for_checkpoint_selection"] is (split == "validation")
        assert len(rows) == count
        assert (
            manifest["phi_free_notes"]
            == manifest["unanchored_positive_notes"]
            == count // 4
        )
        assert manifest["anchored_contrast_pairs"] == count // 4
        counts, tags, lengths = Counter(), Counter(), []
        full[split], aliases[split] = set(), set()
        for row in rows:
            assert row["id"] not in all_ids and row["text"] not in all_text
            all_ids.add(row["id"])
            all_text.add(row["text"])
            counts[row["variant"]] += 1
            generator.validate_row(row)
            length, row_tags = generator.audit.token_audit(row, encoding)
            lengths.append(length)
            tags.update(row_tags)
            for person in (row["person_identity"], row["other_identity"]):
                if person is not None and person["full"] is not None:
                    full[split].add(generator.canonical(person["full"]))
            aliases[split].update(
                generator.canonical(name)
                for name, mention in zip(_names(row), row["name_mentions"])
                if mention["kind"] != "full"
            )
        assert counts == {
            variant: count // 4
            for variant in ("nominal", "counterfactual", "unanchored", "phi_free")
        }
        assert manifest["bioes_tag_support"] == dict(tags)
        assert manifest["token_length"]["max"] == max(lengths) <= 512
        assert manifest["token_length"]["mean"] == sum(lengths) / len(lengths)
        if split == "validation":
            verified_rows, verified_manifest = load_verified_corpus(path)
            assert verified_rows == rows and verified_manifest == manifest
    assert not full["train"] & full["validation"]
    assert manifests["train"]["cross_split_audit"][
        "shared_reduced_alias_values"
    ] == len(aliases["train"] & aliases["validation"])


def test_small_publication_is_byte_reproducible_and_never_overwrites(tmp_path):
    for directory in (tmp_path / "first", tmp_path / "second"):
        generator.write_dataset(directory, train_count=24, val_count=16, seed=74)
    for filename in (
        "nominal_alias_train.jsonl",
        "nominal_alias_validation.jsonl",
        "nominal_alias_train.jsonl.manifest.json",
        "nominal_alias_validation.jsonl.manifest.json",
    ):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    before = (tmp_path / "first" / "nominal_alias_train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        generator.write_dataset(tmp_path / "first", train_count=24, val_count=16)
    assert (tmp_path / "first" / "nominal_alias_train.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "metadata",
    [
        {"split": "final"},
        {"split": "test"},
        {"independent_final": True},
        {"suite": "independent_qualification"},
    ],
)
def test_final_preflight_rejects_before_any_first_payload_read(
    tmp_path, monkeypatch, metadata
):
    approved, forbidden = tmp_path / "approved.jsonl", tmp_path / "forbidden.jsonl"
    _declared(approved, [generator.generate_pair(1)[0]])
    _declared(forbidden, [], **metadata)
    original = Path.read_bytes
    opened = []

    def guarded(path):
        if path in {approved, forbidden}:
            opened.append(path)
            raise AssertionError(
                "Qualification preflight must precede every payload read"
            )
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Qualification"):
        generator.write_dataset(
            tmp_path / "output",
            train_count=8,
            val_count=8,
            compare_corpora=[approved, forbidden],
        )
    assert not opened


def test_duplicate_approved_note_and_id_are_rejected_before_publication(tmp_path):
    prior = tmp_path / "prior.jsonl"
    _declared(prior, [generator.generate_unanchored_pair(0)[0]])
    with pytest.raises(ValueError, match="overlaps"):
        generator.write_dataset(
            tmp_path / "output", train_count=8, val_count=8, compare_corpora=[prior]
        )
    assert not (tmp_path / "output" / "nominal_alias_train.jsonl").exists()


def test_tampered_comparison_hash_fails_before_publication(tmp_path):
    prior = tmp_path / "prior.jsonl"
    _declared(prior, [generator.generate_pair(1)[0]])
    prior.write_text("{}\n")
    with pytest.raises(ValueError, match="declaration"):
        generator.write_dataset(
            tmp_path / "output", train_count=8, val_count=8, compare_corpora=[prior]
        )


def test_legacy_adjacent_declaration_requires_matching_approved_aggregate(tmp_path):
    path = tmp_path / "context_train.jsonl"
    row = generator.generate_pair(1)[0]
    _declared(path, [row], split=None)
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_bytes())
    with pytest.raises(ValueError, match="undeclared"):
        generator._comparison_records([path])
    aggregate = tmp_path / "context_manifest.json"
    aggregate.write_text(
        json.dumps(
            {
                "datasets": {
                    "train": {
                        "file": path.name,
                        "sha256": manifest["sha256"],
                        "count": 1,
                    }
                }
            }
        )
    )
    records = generator._comparison_records([path])
    assert records[0]["legacy_split_declaration"] == str(aggregate.resolve())
    assert (
        records[0]["legacy_split_declaration_sha256"]
        == hashlib.sha256(aggregate.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("split", ["final", "test"])
def test_legacy_aggregate_final_split_is_rejected_before_payload_read(
    tmp_path, monkeypatch, split
):
    path = tmp_path / "legacy.jsonl"
    path.write_text("Qualification payload must remain unopened")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"datasets": {split: {"file": path.name, "sha256": "unused", "count": 1}}}
        )
    )
    original = Path.read_bytes

    def guarded(candidate):
        if candidate == path:
            raise AssertionError("Legacy final payload must not be opened")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Qualification"):
        generator._comparison_records([path])


@pytest.mark.parametrize("adjacent", [False, True])
@pytest.mark.parametrize(
    "metadata,entry_metadata",
    [
        ({"split": "final"}, {}),
        ({"independent_final": True}, {}),
        ({"suite": "independent_qualification"}, {}),
        ({}, {"split": "test"}),
        ({}, {"independent_final": True}),
        ({}, {"suite": "final"}),
    ],
)
def test_forbidden_aggregate_and_entry_declarations_precede_every_legacy_payload(
    tmp_path, monkeypatch, adjacent, metadata, entry_metadata
):
    path = tmp_path / "legacy.jsonl"
    _declared(path, [generator.generate_pair(1)[0]], split=None)
    declaration = json.loads(path.with_suffix(".jsonl.manifest.json").read_bytes())
    if not adjacent:
        path.with_suffix(".jsonl.manifest.json").unlink()
    aggregate = tmp_path / ("context_manifest.json" if adjacent else "manifest.json")
    aggregate.write_text(
        json.dumps(
            {
                "datasets": {
                    "train": {
                        "file": path.name,
                        "sha256": declaration["sha256"],
                        "count": 1,
                    }
                    | entry_metadata
                }
            }
            | metadata
        )
    )
    original = Path.read_bytes

    def guarded(candidate):
        if candidate == path:
            raise AssertionError(
                "Forbidden legacy declaration must be checked before payload"
            )
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Qualification"):
        generator._comparison_records([path])


def test_legacy_aggregate_approved_training_is_pinned_and_hash_verified(tmp_path):
    path = tmp_path / "legacy.jsonl"
    payload = (json.dumps(generator.generate_pair(1, split="train")[0]) + "\n").encode()
    path.write_bytes(payload)
    aggregate = tmp_path / "manifest.json"
    aggregate.write_text(
        json.dumps(
            {
                "datasets": {
                    "train": {
                        "file": path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "count": 1,
                    }
                }
            }
        )
    )
    records = generator._comparison_records([path])
    assert records[0]["rows"] == 1
    assert (
        records[0]["legacy_split_declaration_sha256"]
        == hashlib.sha256(aggregate.read_bytes()).hexdigest()
    )


def test_declared_development_to_training_conversion_is_accepted_for_overlap_audit(
    tmp_path,
):
    path = tmp_path / "development_training.jsonl"
    _declared(
        path,
        [generator.generate_pair(1)[0]],
        split=None,
        training_conversion={
            "declared": True,
            "source_suite": "development",
            "source_split": "validation",
        },
    )
    assert generator._comparison_records([path])[0]["rows"] == 1


@pytest.mark.parametrize("count", [True, 0, 6, 9, 15])
def test_counts_must_support_exact_quarter_cohorts(tmp_path, count):
    with pytest.raises(ValueError, match="divisible by four"):
        generator.write_dataset(tmp_path, train_count=count, val_count=8)
