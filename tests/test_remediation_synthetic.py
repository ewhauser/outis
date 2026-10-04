import hashlib
import json
import re
from collections import Counter, defaultdict

import pytest

from clinical_deid.data.generators import remediation as generator
from clinical_deid.types import ENTITY_TYPES


@pytest.fixture(scope="module")
def rows():
    return generator.build_corpus(short_count=1000, long_count=0)


def test_replacement_is_deterministic_and_grouped_without_alphabet_split(rows):
    assert rows == generator.build_corpus(short_count=1000, long_count=0)
    assert rows[:30] != generator.build_corpus(short_count=30, long_count=0, seed=17)
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["text"] for row in rows}) == len(rows)
    for row in rows:
        generator.validate_row(row)
        assert row["split"] == "train"
        assert row["suite"] == "training"
        assert row["annotation_policy"] == "clinical-phi-v2"
        assert row["identity_groups"] == [row["identity_group"]]
        assert int(row["identity_group"][:8], 16) % 10 < 8
        assert row["template_family"].startswith("remediation_train.")
    assert generator.identity_group("Lucía García") == generator.identity_group(
        "Luci\u0301a Garci\u0301a"
    )
    assert generator.identity_group("Lucía García") == generator.identity_group(
        "LUCÍA GARCÍA"
    )


def test_corpus_support_and_realistic_vs_unusual_names(rows):
    assert Counter(row["cohort"] for row in rows) == {
        "realistic": 950,
        "unusual_name": 50,
    }
    assert sum(row["negative"] for row in rows) == 180
    support = Counter(span["label"] for row in rows for span in row["spans"])
    assert set(support) == set(ENTITY_TYPES)
    assert min(support.values()) > 100
    assert len({row["template_family"] for row in rows}) >= 60
    for row in rows:
        given = row["full_identity"].split()[0]
        if row["cohort"] == "realistic":
            assert given not in generator.UNUSUAL_NAMES
        else:
            assert given in generator.UNUSUAL_NAMES
            assert "Patient registration name:" in row["text"]


def test_context_pairs_keep_the_same_owner_and_label_eponyms_in_context(rows):
    pairs = defaultdict(list)
    for row in rows:
        if "context_pair" in row["phenomena"]:
            pairs[row["pair_id"]].append(row)
    assert len(pairs) == 180
    collision_pairs = 0
    for positive, negative in pairs.values():
        assert positive["identity_group"] == negative["identity_group"]
        assert positive["spans"] and not negative["spans"]
        if "eponym_collision" in positive["phenomena"]:
            collision_pairs += 1
            surname = positive["full_identity"].split()[-1]
            disease = dict(generator.EPONYMS)[surname]
            assert disease in positive["text"] and disease in negative["text"]
            offset = positive["text"].index(disease)
            assert all(
                span["end"] <= offset or span["start"] >= offset + len(disease)
                for span in positive["spans"]
            )
    assert collision_pairs >= 50


def test_negative_and_positive_prose_share_clinical_distractions_without_annotation_cues(
    rows,
):
    for negative in (False, True):
        cohort = [row for row in rows if row["negative"] == negative]
        combined = "\n".join(row["text"] for row in cohort)
        assert "published in" in combined
        assert "ages 50 to 75" in combined
        assert "He asked" in combined and "She reports" in combined
        assert "over 90 days" in combined
        assert "|" in combined
        assert "thought process" in combined.lower()
        assert "fetal movement" in combined
        assert "fluoride toothpaste" in combined
        assert "variant of uncertain significance" in combined
    for row in rows:
        text = row["text"]
        assert not re.search(
            r"\b(?:PHI|annotation|labeling|non-PHI|hard negative|remains clinical information)\b",
            text,
            re.I,
        )
        if "younger_age_context" in row["phenomena"]:
            patient_age = re.search(r"Patient: (\d+)", text)
            assert "ages 50 to 75" in text or (
                patient_age and int(patient_age.group(1)) <= 89
            )
        if "eponym_negative" in row["phenomena"]:
            assert any(disease in text for _, disease in generator.EPONYMS)
        for phrase in generator.PRONOUNS:
            if phrase in text:
                start = text.index(phrase)
                assert all(
                    s["end"] <= start or s["start"] >= start + len(phrase)
                    for s in row["spans"]
                )


def test_compact_and_spelled_ages_label_values_only_and_young_positive_patients_exist(
    rows,
):
    compact, spelled, young_positive = 0, 0, 0
    for row in rows:
        age_match = re.search(r"Patient: (\d+)", row["text"])
        if age_match and int(age_match.group(1)) <= 89 and not row["negative"]:
            young_positive += 1
            assert not any(s["label"] == "AGE" for s in row["spans"])
        if "prenatal_review" in row["family"]:
            assert age_match and 20 <= int(age_match.group(1)) <= 44
        for span in row["spans"]:
            if span["label"] != "AGE":
                continue
            value = row["text"][span["start"] : span["end"]]
            assert "year" not in value and "yo" not in value
            if value.isdigit():
                assert 90 <= int(value) <= 100
                if row["text"][span["end"] :].startswith(("F", "M", "yo", " y/o")):
                    compact += 1
            else:
                assert value.startswith("ninety") or value == "one hundred"
                spelled += 1
    assert compact > 50 and spelled > 30 and young_positive > 100


def test_reserved_identifiers_cover_varied_formats(rows):
    phones, hosts, ips = set(), set(), set()
    for row in rows:
        for span in row["spans"]:
            value = row["text"][span["start"] : span["end"]]
            if span["label"] == "PHONE":
                assert re.search(r"555[-. ]01\d\d", value)
                phones.add(
                    "international"
                    if value.startswith("+")
                    else "parenthesized"
                    if value.startswith("(")
                    else "extension"
                    if " ext." in value
                    else "dots"
                    if "." in value
                    else "extension"
                )
            if span["label"] == "EMAIL":
                hosts.add(value.split("@")[1])
                assert value.endswith((".test", ".example"))
            if span["label"] == "IP":
                ips.add(value.rsplit(".", 1)[0])
            if span["label"] == "SSN":
                assert value.startswith("000-")
    assert len(phones) == 4 and len(hosts) == 4 and len(ips) == 3


@pytest.mark.parametrize(
    "index,target", [(303, 3000), (305, 4000), (306, 4500), (999, 1500), (1001, 6000)]
)
def test_long_stress_uses_actual_token_ownership_at_every_declared_join(index, target):
    row = generator.generate_long_row(index, target_tokens=target)
    audit = row["token_audit"]
    assert 1500 <= audit["token_count"] <= 6000
    assert row["diagnostic_slice"] == "seam_stress"
    assert audit["token_count"] == len(
        generator.get_encoding().encode(row["text"], disallowed_special=())
    )
    targeted = set(audit["targeted_boundaries"])
    for key in ("inference_joins", "training_seams"):
        assert all(
            item["crossing_spans"] for item in audit[key] if item["token"] in targeted
        )
    assert row["text"].count("Earlier handover:") == 2
    generator.validate_row(row)


@pytest.mark.parametrize("target", [1500, 6000])
def test_natural_long_notes_are_separate_from_join_alignment(target):
    row = generator.generate_long_row(7501, target_tokens=target, mode="long_narrative")
    assert row["diagnostic_slice"] == "long_narrative"
    assert "targeted_boundaries" not in row["token_audit"]
    assert 1500 <= row["token_audit"]["token_count"] <= 6000
    assert row["text"].count("Handover carried forward:") == 2
    assert "Final handover" in row["text"]
    assert len({s["label"] for s in row["spans"]}) >= 4


def test_immutable_writer_is_loader_compatible_and_refuses_existing_artifacts(tmp_path):
    from clinical_deid.evaluation.evaluate import load_verified_corpus

    rows = generator.build_corpus(short_count=30, long_count=0)
    path = tmp_path / "train.jsonl"
    manifest = generator.write_corpus(path, rows, config={"seed": generator.SEED})
    before = path.read_bytes()
    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["replaces_legacy_training_inputs"] is True
    assert manifest["annotation_policy"] == "clinical-phi-v2"
    loaded, verified = load_verified_corpus(path)
    assert loaded == rows and verified == manifest
    with pytest.raises(FileExistsError):
        generator.write_corpus(path, rows)
    assert path.read_bytes() == before
    manifest_only = tmp_path / "existing.jsonl.manifest.json"
    manifest_only.write_text(json.dumps({"keep": True}))
    with pytest.raises(FileExistsError):
        generator.write_corpus(tmp_path / "existing.jsonl", rows)
    assert not (tmp_path / "existing.jsonl").exists()


def test_invalid_counts_fractions_and_long_modes_fail():
    for kwargs in (
        {"short_count": 0},
        {"long_count": -1},
        {"negative_fraction": 0.01},
        {"natural_long_fraction": 2},
    ):
        with pytest.raises(ValueError):
            generator.build_corpus(**kwargs)
    with pytest.raises(ValueError):
        generator.generate_long_row(0, target_tokens=1000)
    with pytest.raises(ValueError):
        generator.generate_long_row(0, mode="unknown")
