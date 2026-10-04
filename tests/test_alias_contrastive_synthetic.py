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

from clinical_deid.data.generators import alias_contrastive as generator
from clinical_deid.evaluation.evaluate import load_verified_corpus
from clinical_deid.types import ENTITY_TYPES


@pytest.fixture(scope="module")
def encoding():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    return tiktoken.get_encoding("o200k_base")


def _canonical(value):
    return unicodedata.normalize("NFC", value).casefold()


def _names(row):
    return [
        row["text"][s["start"] : s["end"]] for s in row["spans"] if s["label"] == "NAME"
    ]


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


def test_pairs_reproduce_exactly_with_owned_alias_and_second_person_offsets():
    rows = [generator.generate_pair(index, seed=59) for index in range(90)]
    assert rows == [generator.generate_pair(index, seed=59) for index in range(90)]
    assert rows != [generator.generate_pair(index, seed=60) for index in range(90)]
    for positive, negative in rows:
        assert positive["pair_id"] == negative["pair_id"]
        assert positive["person_identity"] == negative["person_identity"]
        assert not positive["negative"] and negative["negative"]
        assert negative["spans"] == negative["name_mentions"] == []
        for row in (positive, negative):
            generator.validate_row(row)
        assert _names(positive)[0] == positive["person_identity"]["full"]
        assert [mention["kind"] for mention in positive["name_mentions"][:3]] == [
            "full",
            "first",
            "surname",
        ]
        for value, mention in zip(_names(positive), positive["name_mentions"]):
            person = (
                positive["person_identity"]
                if mention["person"] == "primary"
                else positive["other_identity"]
            )
            assert value == person[mention["kind"]]


@pytest.mark.parametrize("split,count", [("train", 1000), ("validation", 250)])
def test_production_scale_covers_all_contexts_clinical_styles_and_identity_forms(
    split, count
):
    captions, aliases, profiles, references, layouts, modes, collisions = (
        set() for _ in range(7)
    )
    for index in range(count):
        positive, negative = generator.generate_pair(index, split=split)
        person = positive["person_identity"]
        modes.add(person["mode"])
        profiles.add(positive["clinical_profile"])
        references.add(positive["public_reference_context"])
        layouts.add(positive["family"].split(".")[2])
        for context in positive["authored_contexts"]:
            kind, number = context.split("-")
            (captions if kind == "caption" else aliases).add(int(number))
        if person["common_first"]:
            collisions.add(person["first"].casefold())
            assert person["first"] in negative["text"]
        assert negative["spans"] == []
    assert captions == set(range(len(generator.CAPTIONS[split])))
    assert aliases == set(range(len(generator.ALIAS_FRAMES[split])))
    assert profiles == set(range(24))
    assert references == set(range(8))
    assert layouts == set(generator.LAYOUTS)
    assert modes == set(range(10))
    assert collisions == {value.casefold() for value in generator.COMMON_FIRST}
    assert not set(generator.CAPTIONS["train"]) & set(generator.CAPTIONS["validation"])
    assert not set(generator.ALIAS_FRAMES["train"]) & set(
        generator.ALIAS_FRAMES["validation"]
    )


def test_full_private_identities_are_disjoint_but_common_alias_overlap_is_declared():
    full, aliases, ids, texts = {}, {}, {}, {}
    for split, count in (("train", 1000), ("validation", 250)):
        full[split], aliases[split], ids[split], texts[split] = (
            set() for _ in range(4)
        )
        for index in range(count):
            for row in generator.generate_pair(index, split=split):
                ids[split].add(row["id"])
                texts[split].add(row["text"])
                if row["negative"]:
                    continue
                for value, mention in zip(_names(row), row["name_mentions"]):
                    (full if mention["kind"] == "full" else aliases)[split].add(
                        _canonical(value)
                    )
        assert len(ids[split]) == len(texts[split]) == count * 2
    assert not full["train"] & full["validation"]
    assert not ids["train"] & ids["validation"]
    assert not texts["train"] & texts["validation"]
    assert aliases["train"] & aliases["validation"] == {
        value.casefold() for value in generator.COMMON_FIRST
    }


def test_common_word_first_aliases_require_explicit_identity_or_person_role_proof():
    for split in generator.CAPTIONS:
        for index in range(160):
            positive, _ = generator.generate_pair(index, split=split)
            if not positive["person_identity"]["common_first"]:
                continue
            for context, mention in zip(
                positive["authored_contexts"][1:], positive["name_mentions"][1:4]
            ):
                if mention["kind"] == "first":
                    assert int(context.split("-")[1]) in generator.SAFE_COMMON[split]
                    frame = generator.ALIAS_FRAMES[split][int(context.split("-")[1])]
                    assert "{role}" in frame


def test_reduced_alias_captions_do_not_confuse_given_names_and_surnames():
    for split in generator.CAPTIONS:
        for index in range(180):
            positive, _ = generator.generate_pair(index, split=split)
            for context, mention in zip(
                positive["authored_contexts"][1:], positive["name_mentions"][1:4]
            ):
                frame = generator.ALIAS_FRAMES[split][int(context.split("-")[1])]
                if "surname" in frame:
                    assert mention["kind"] == "surname"
                if "given name" in frame or "first-name" in frame:
                    assert mention["kind"] == "first"


def test_distinct_person_and_comma_order_names_remain_complete(encoding):
    hits = Counter()
    for index in range(180):
        positive, _ = generator.generate_pair(index)
        length, tags = generator.components.shared.token_audit(positive, encoding)
        assert length < 400
        person = positive["person_identity"]
        if person["comma_order"]:
            assert ", " in _names(positive)[0]
            hits["comma"] += 1
        if positive["other_identity"] is not None:
            assert person["full"] != positive["other_identity"]["full"]
            assert person["first"] != positive["other_identity"]["first"]
            assert person["surname"] != positive["other_identity"]["surname"]
            assert len(_names(positive)) == 8
            assert [m["person"] for m in positive["name_mentions"][-2:]] == [
                "primary",
                "clinician",
            ]
            assert "\nFor " in positive["text"]
            hits["two-person"] += 1
        else:
            assert len(_names(positive)) == 4
        hits["S-NAME"] += tags["S-NAME"]
        for span in positive["spans"]:
            assert not unicodedata.category(positive["text"][span["start"]]).startswith(
                "M"
            )
            assert span["end"] == len(positive["text"]) or not unicodedata.category(
                positive["text"][span["end"]]
            ).startswith("M")
    assert hits["comma"] == 45 and hits["two-person"] == 60
    assert hits["S-NAME"] > 100


def test_lowercase_nfd_particles_initials_suffixes_and_short_first_names_are_present():
    traits = Counter()
    for index in range(150):
        positive, _ = generator.generate_pair(index)
        identity = positive["person_identity"]
        traits["lower"] += identity["full"].islower()
        traits["nfd"] += identity["full"] != unicodedata.normalize(
            "NFC", identity["full"]
        )
        if identity["mode"] == 4:
            assert len(identity["first"]) <= 3
            traits["short"] += 1
        if identity["mode"] == 6:
            assert unicodedata.category(identity["first"][1]).startswith("M")
        if identity["mode"] == 7:
            assert re.search(r"[A-Z][\u0300-\u036f]+\.", identity["full"], re.I)
            traits["initial"] += 1
        if identity["mode"] == 8:
            assert identity["surname"].split()[0].casefold() in {
                "de",
                "van",
                "al",
                "von",
            }
            traits["particle"] += 1
        if identity["mode"] == 9:
            assert identity["suffix"] in identity["full"]
            traits["suffix"] += 1
    assert all(
        traits[key] >= 15
        for key in ("lower", "nfd", "short", "initial", "particle", "suffix")
    )


def test_clinical_bodies_public_references_and_normal_measurements_are_nonidentifying():
    hits = Counter()
    patterns = (
        r"\d+/\d+ mmHg",
        r"\d+ (?:mg|mcg)",
        r"\d+ mmol/L",
        r"\d+ mg/dL",
        r"\d+/30",
        r"\b5/5\b",
        r"\d+ days",
        r"Romberg testing",
        r"Addison disease",
        r"Graves disease",
    )
    for index in range(100):
        positive, negative = generator.generate_pair(index)
        clinical_clause = generator.CLINICAL_STYLES[positive["clinical_profile"]]
        end_body = positive["text"].index(clinical_clause) + len(clinical_clause)
        assert positive["text"][:end_body] == negative["text"][:end_body]
        assert positive["text"].splitlines()[-1] == negative["text"].splitlines()[-1]
        reference_start = positive["text"].rfind("\n") + 1
        assert all(span["end"] <= reference_start for span in positive["spans"])
        assert positive["publication_year"] < positive["clinical_episode_year"]
        assert negative["spans"] == []
        for pattern in patterns:
            for match in re.finditer(pattern, positive["text"]):
                hits[pattern] += 1
                assert all(
                    match.end() <= span["start"] or match.start() >= span["end"]
                    for span in positive["spans"]
                )
    assert all(hits[pattern] for pattern in patterns)


def test_private_dates_follow_in_order_without_labelling_public_reference_years():
    for index in range(80):
        positive, negative = generator.generate_pair(index)
        dates = [
            positive["text"][span["start"] : span["end"]]
            for span in positive["spans"]
            if span["label"] == "DATE"
        ]
        if index % 5 == 0:
            assert len(dates) == 2
            start, followup = map(date.fromisoformat, dates)
            assert (followup - start).days == 21
            assert start.year == positive["clinical_episode_year"]
            assert dates[0] not in negative["text"] and dates[1] not in negative["text"]
        else:
            assert dates == []


@pytest.mark.parametrize(
    "mutation",
    ["overlap", "out-of-bounds", "label", "alias-owner", "negative", "chronology"],
)
def test_corrupt_offsets_and_alias_ownership_are_rejected(mutation):
    row = copy.deepcopy(generator.generate_pair(0)[0])
    if mutation == "overlap":
        row["spans"][1]["start"] = row["spans"][0]["start"]
    elif mutation == "out-of-bounds":
        row["spans"][0]["end"] = len(row["text"]) + 1
    elif mutation == "label":
        row["spans"][0]["label"] = "DRUG"
    elif mutation == "alias-owner":
        row["name_mentions"][1]["person"] = "clinician"
    elif mutation == "negative":
        row["negative"] = True
    else:
        row["publication_year"] = row["clinical_episode_year"]
    with pytest.raises(ValueError):
        generator.validate_row(row)


def test_manifest_matches_actual_annotation_token_ownership_and_development_role(
    tmp_path, encoding
):
    compare = tmp_path / "approved.jsonl"
    _declared(
        compare,
        [
            {
                "id": "prior-id",
                "text": "Clinical observations outside this authored corpus",
                "split": "validation",
            }
        ],
    )
    manifests = generator.write_dataset(
        tmp_path / "output", 160, 80, compare_corpora=[compare]
    )
    for split in ("train", "validation"):
        path = tmp_path / "output" / f"alias_contrastive_{split}.jsonl"
        rows, manifest = load_verified_corpus(path)
        assert manifests[split] == manifest
        assert (
            manifest["source_sha256"]
            == hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()
        )
        assert (
            manifest["component_source_sha256"]
            == hashlib.sha256(
                Path(generator.components.__file__).read_bytes()
            ).hexdigest()
        )
        assert (
            manifest["license"] == "CC0-1.0"
            and manifest["code_license"] == "Apache-2.0"
        )
        assert manifest["external_datasets"] == []
        assert manifest["used_for_training"] is (split == "train")
        assert manifest["used_for_checkpoint_selection"] is (split == "validation")
        assert manifest["independent_final"] is False
        labels, tags, lengths = Counter(), Counter(), []
        for row in rows:
            generator.validate_row(row)
            length, row_tags = generator.components.shared.token_audit(row, encoding)
            lengths.append(length)
            tags.update(row_tags)
            labels.update(span["label"] for span in row["spans"])
        assert manifest["bioes_tag_support"] == tags
        assert manifest["label_support"] == {
            label: labels[label] for label in ENTITY_TYPES
        }
        assert (
            manifest["phi_free_notes"] == manifest["contrast_pairs"] == len(rows) // 2
        )
        assert manifest["name_span_traits"]["name_spans"] == labels["NAME"]
        assert manifest["token_length"]["max"] == max(lengths) <= 512
        assert manifest["token_length"]["mean"] == sum(lengths) / len(lengths)
        assert manifest["cross_split_audit"]["full_identity_overlap"] == 0
        assert manifest["cross_split_audit"]["shared_reduced_alias_values"] > 0
        assert manifest["comparison_audit"][0]["exact_text_overlap"] == 0
        assert manifest["comparison_audit"][0]["exact_id_overlap"] == 0


def test_immutable_publication_and_seed_reproduction(tmp_path):
    first = generator.write_dataset(tmp_path / "first", 40, 20, seed=74)
    assert first == generator.write_dataset(tmp_path / "second", 40, 20, seed=74)
    path = tmp_path / "first" / "alias_contrastive_train.jsonl"
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        generator.write_dataset(tmp_path / "first", 40, 20, seed=75)
    assert path.read_bytes() == before


@pytest.mark.parametrize("collision", ["text", "id"])
def test_exact_approved_text_or_id_overlap_is_rejected(tmp_path, collision):
    row = generator.generate_pair(0)[0]
    comparison = tmp_path / "prior.jsonl"
    _declared(
        comparison,
        [
            {
                "id": row["id"] if collision == "id" else "prior-id",
                "text": row["text"]
                if collision == "text"
                else "Ordinary unrelated clinical observations",
                "split": "validation",
            }
        ],
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps"):
        generator.write_dataset(output, 4, 2, compare_corpora=[comparison])
    assert not list(output.iterdir())


def test_qualification_declaration_is_rejected_before_any_payload_read(
    tmp_path, monkeypatch
):
    approved = tmp_path / "approved.jsonl"
    _declared(
        approved,
        [
            {
                "id": "approved-id",
                "text": "Ordinary clinical observations",
                "split": "validation",
            }
        ],
    )
    sealed = tmp_path / "no-qualification-payload-created.jsonl"
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
def test_note_counts_require_complete_pairs(tmp_path, count):
    with pytest.raises(ValueError, match="positive even"):
        generator.write_dataset(tmp_path, count, 2)
