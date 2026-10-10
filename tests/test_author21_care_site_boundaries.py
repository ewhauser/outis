"""Tests for the tk21 care-site boundaries training bank author."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from outis.data.annotation_policy import (
    POLICY_FIXTURES,
    annotation_errors,
    identity_split,
)
from outis.data.annotation_policy_ages import AGE_FIXTURES
from outis.data.annotation_policy_care_sites import CARE_SITE_FIXTURES
from outis.data.annotation_policy_places import PLACE_FIXTURES, place_value_pool
from outis.data.generators import place_sources

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author21_care_site_boundaries.py"


def recorded_sha256(script):
    """Hash of the generator as recorded, before the package was renamed to outis."""
    return hashlib.sha256(
        script.read_bytes().replace(b"outis", b"clinical_deid")
    ).hexdigest()


def _load():
    spec = importlib.util.spec_from_file_location(
        "author21_care_site_boundaries", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


author = _load()
needs_sources = pytest.mark.skipif(
    not place_sources.sources_available()
    or not (ROOT / "data" / "places_exclusions.sha256.json").exists(),
    reason="pinned place sources or exclusion list not available",
)


def _fixture(fixtures, key):
    return next(item for item in fixtures if item["id"] == key)


def _same(marked, fixture):
    text, spans = author.parse(marked)
    assert text == fixture["text"]
    assert spans == fixture["spans"]


# ---------------------------------------------------------------------------- policy fixtures


def test_renderers_rebuild_place_and_location_fixtures():
    _same(
        "Treated at " + author.site_in_locality("Mayo Clinic", "Rochester, MN") + ".",
        _fixture(PLACE_FIXTURES, "p5_care_site_then_locality"),
    )
    _same(
        "Lives in "
        + author.locality_with_state("Tacoma", "WA")
        + ", not elsewhere in Washington.",
        _fixture(PLACE_FIXTURES, "state_inside_locality"),
    )
    _same(
        "Lives in "
        + author.locality_with_state("Cedar Haven", "UT")
        + "; transferred to "
        + author.site_in_locality("Silver Finch Medical Center", "North Cedar Haven")
        + ".",
        _fixture(POLICY_FIXTURES, "maximal_location"),
    )


def test_renderers_rebuild_at_rule_fixture():
    _same(
        "Seen at "
        + author.site_at_continuation("Lakeshore Family Medicine", "Ellery Falls")
        + "; moved to "
        + author.unit_at_parent("Larkspur Ward", "Mossmere Health Hub")
        + ", then "
        + author.site_at_qualified("Brackenwheel Health Center", "Brackenford, NM")
        + ".",
        _fixture(POLICY_FIXTURES, "facility_at_continuation"),
    )


def test_renderers_rebuild_article_fixture():
    _same(
        author.article_site("The", "Pellham Drug & Gift")
        + " pharmacist called; she was seen at "
        + author.article_site("the", "Ashgrove Family Health")
        + " office.",
        _fixture(CARE_SITE_FIXTURES, "the_as_article"),
    )


def test_renderers_rebuild_age_fixtures():
    _same(
        f"A {author.age_value(72)}-year-old woman; {author.age_value(89)} years old at the earlier visit; "
        f"{author.age_value(93)}F; a {author.age_value(6, words=True)}-month-old sibling.",
        _fixture(AGE_FIXTURES, "ages_any_value"),
    )
    _same(
        f"Mother died at {author.age_value(72, words=True)}; brother, {author.age_value(58)}, is well.",
        _fixture(AGE_FIXTURES, "ages_of_relatives"),
    )
    _same(
        f"A woman in her {author.M('AGE', author.DECADES[7])}; her brother is in his {author.M('AGE', '40s')}; "
        f"father in his early {author.M('AGE', author.DECADES[6])}; the 1970s were hard.",
        _fixture(AGE_FIXTURES, "decades"),
    )


def test_department_stays_outside():
    text, spans = author.parse(
        "To: " + author.site_with_department("Quenby Valley Hospital", "Nurse Pool")
    )
    assert [text[s["start"] : s["end"]] for s in spans] == ["Quenby Valley Hospital"]


def test_number_words():
    assert author.number_words(72) == "seventy-two"
    assert author.number_words(100) == "one hundred"
    assert author.number_words(103) == "one hundred three"
    assert author.an(80) == "an" and author.an(18) == "an" and author.an(72) == "a"


def test_title_case_cms():
    assert (
        author.title_case_cms("ST MARYS REGIONAL MEDICAL CENTER")
        == "St. Marys Regional Medical Center"
    )
    assert author.title_case_cms("MCPHERSON HOSPITAL") == "McPherson Hospital"
    assert (
        author.title_case_cms("MURRAY-CALLOWAY COUNTY HOSPITAL")
        == "Murray-Calloway County Hospital"
    )
    assert author.title_case_cms("KAISER FOUNDATION HOSPITAL - SAN DIEGO") is None
    assert author.title_case_cms("THE HEART HOSPITAL") is None


# ---------------------------------------------------------------------------- the bank


@pytest.fixture(scope="module")
def built():
    if (
        not place_sources.sources_available()
        or not (ROOT / "data" / "places_exclusions.sha256.json").exists()
    ):
        pytest.skip("pinned place sources or exclusion list not available")
    return author.build(author.SEED)


@pytest.fixture(scope="module")
def rows(built):
    return built["rows"]


@needs_sources
def test_deterministic(built):
    again = author.build(author.SEED)
    assert again["payload"] == built["payload"]
    assert again["manifest"]["sha256"] == built["manifest"]["sha256"]


@needs_sources
def test_written_bank_matches_build(built):
    bank = author.BANK
    if not bank.exists():
        pytest.skip("bank not written")
    payload = bank.read_bytes()
    assert payload == built["payload"]
    manifest = json.loads(bank.with_suffix(bank.suffix + ".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["author_provenance"]["generator_sha256"] == recorded_sha256(SCRIPT)


@needs_sources
def test_manifest_fields(built):
    m = built["manifest"]
    for key in (
        "annotation_policy",
        "split",
        "rows",
        "documents",
        "sha256",
        "label_support",
        "entity_support",
        "cohort_counts",
        "author_provenance",
        "source_pins",
    ):
        assert key in m
    assert (
        m["annotation_policy"] == "clinical-phi-v2"
        and m["split"] == "train"
        and m["independent_final"] is False
    )
    assert (
        m["rows"]
        == m["documents"]
        == len(built["rows"])
        == sum(author.FAMILY_ROWS.values())
    )
    assert m["author_provenance"]["seed"] == author.SEED
    assert set(m["source_pins"]) == set(place_sources.SOURCES)
    for pin in m["source_pins"].values():
        assert pin["verified_sha256"] == pin["sha256"]


@needs_sources
def test_offsets_and_structure(rows):
    for row in rows:
        assert not annotation_errors(row["text"], row["spans"]), row["id"]
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
            assert span["text"] == span["text"].strip()
            assert (
                span["text"][-1] not in ".;:"
                or span["label"] == "FACILITY"
                and span["text"].endswith(".")
            )
        assert row["family"] == row["template_family"]
        assert row["split"] == "train" and row["annotation_policy"] == "clinical-phi-v2"
        assert identity_split(row["identity_group"]) == "train"
        assert all(identity_split(group) == "train" for group in row["identity_groups"])
        assert row["identity_group"] in row["identity_groups"]
    assert len({row["text"] for row in rows}) == len(rows)
    assert len({row["id"] for row in rows}) == len(rows)


@needs_sources
def test_composition(rows):
    families = Counter(row["family"] for row in rows)
    assert families == Counter(author.FAMILY_ROWS)
    layouts = Counter(row["layout"] for row in rows)
    assert len(layouts) >= 25
    assert max(layouts.values()) <= 0.08 * len(rows)
    cases = Counter(row["case"] for row in rows)
    assert abs(cases["normal"] / len(rows) - 0.6) < 0.05
    assert abs(cases["upper"] / len(rows) - 0.2) < 0.05
    assert abs(cases["lower"] / len(rows) - 0.2) < 0.05
    negatives = sum(1 for row in rows if row["role"] in ("negative", "contrast"))
    assert abs(negatives / len(rows) - 1 / 3) < 0.03


@needs_sources
def test_case_is_whole_row(rows):
    for row in rows:
        if row["case"] == "upper":
            assert row["text"] == row["text"].upper()
        elif row["case"] == "lower":
            assert row["text"] == row["text"].lower()


@needs_sources
def test_repetition_limit(rows):
    counts = Counter()
    for row in rows:
        for key in {
            (s["label"], s["text"].casefold())
            for s in row["spans"]
            if s["label"] in ("FACILITY", "LOCATION")
        }:
            counts[key] += 1
    assert max(counts.values()) <= 0.005 * len(rows)


@needs_sources
def test_zero_heldout_values(rows):
    seen = 0
    for row in rows:
        for item in row["real_place_values"]:
            seen += 1
            assert item["pool"] == "train", item
    assert seen > 500
    for row in rows:
        if row["family"] in ("tk21_city_with_state", "tk21_facility_and_locality"):
            for span in row["spans"]:
                if span["label"] == "LOCATION" and row["real_place_values"]:
                    values = {
                        item["value"].casefold() for item in row["real_place_values"]
                    }
                    if span["text"].casefold() in values:
                        assert place_value_pool(span["text"]) == "train"


@needs_sources
def test_exclusion_filter_applied(rows):
    from outis.data.generators.places import excluded_sequences

    assert all(excluded_sequences(row["text"]) == 0 for row in rows)


@needs_sources
def test_matched_negatives_and_pairs(rows):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
        if row["role"] == "negative":
            assert not any(
                s["label"] in ("FACILITY", "LOCATION") for s in row["spans"]
            ), row["id"]
            assert "matched_negative" in row["phenomena"]
    for members in by_pair.values():
        assert len(members) in (1, 2)
        if len(members) == 2:
            a, b = members
            assert a["identity_group"] == b["identity_group"]
            assert (
                a["layout"] == b["layout"]
                and a["case"] == b["case"]
                and a["family"] == b["family"]
            )
            assert {a["role"], b["role"]} in (
                {"positive", "negative"},
                {"positive", "contrast"},
            )


@needs_sources
def test_family_conventions(rows):
    departments = {d.casefold() for d in author.DEPARTMENTS}
    states = {s.casefold() for s in author.STATES} | {
        s.casefold() for s in author.STATES.values()
    }
    for row in rows:
        text, fam, role = row["text"], row["family"], row["role"]
        fac = [s for s in row["spans"] if s["label"] == "FACILITY"]
        loc = [s for s in row["spans"] if s["label"] == "LOCATION"]
        if role == "positive" and fam == "tk21_names_as_addresses":
            assert len(fac) == 1 and not loc
            after = text[fac[0]["end"] :].casefold()
            assert after.startswith(" ") and any(
                after[1:].startswith(d) for d in departments
            ), row["id"]
            assert not any(d in fac[0]["text"].casefold() for d in departments)
        if role == "positive" and fam == "tk21_city_with_state":
            assert len(loc) == 1 and not fac
            assert re.split(r"[ ,]+", loc[0]["text"].casefold())[-1] in states or any(
                loc[0]["text"].casefold().endswith(s) for s in states
            ), row["id"]
        if role == "positive" and fam == "tk21_care_type_tails":
            assert len(fac) == 1
            if "campus_in_name" in row["phenomena"]:
                assert (
                    fac[0]["text"].casefold().endswith("campus")
                    and " - " in fac[0]["text"]
                )
        if role == "positive" and fam == "tk21_long_connector_names":
            assert len(fac) == 1
            assert re.search(r"\b(?:and|of|for)\b", fac[0]["text"], re.I)
            assert not fac[0]["text"].casefold().startswith("the ")
        if role == "positive" and fam == "tk21_real_hospital_abbreviations":
            assert len(fac) == 1
            if "care_site_short_form" in row["phenomena"]:
                assert re.search(r"\b(?:Hosp|Med Ctr)\b", fac[0]["text"], re.I)
            assert (
                not fac[0]["text"].casefold().endswith((" ed", " emergency department"))
            )
        if fam == "tk21_facility_and_locality":
            if "p5a_at_continuation" in row["phenomena"]:
                assert len(fac) == 1 and not loc and " at " in fac[0]["text"].casefold()
            elif "p5a_qualified_locality_split" in row["phenomena"]:
                assert len(fac) == 1 and len(loc) == 1
                assert text[fac[0]["end"] : loc[0]["start"]].casefold() == " at "
                assert any(loc[0]["text"].casefold().endswith(s) for s in states)
            else:
                assert len(fac) == 1 and len(loc) == 1
                assert text[fac[0]["end"] : loc[0]["start"]].casefold() in (
                    " in ",
                    ", ",
                )


@needs_sources
def test_every_age_annotated(rows):
    pattern = re.compile(
        r"\b(\d{1,3})(?=-year-old| y/o| yo |[mf]\b| years\b| y\b)|(?<=aged )(\d{1,3})\b|(?<=age: )(\d{1,3})\b",
        re.I,
    )
    for row in rows:
        ages = {(s["start"], s["end"]) for s in row["spans"] if s["label"] == "AGE"}
        for match in pattern.finditer(row["text"]):
            group = next(g for g in range(1, 4) if match.group(g))
            assert (match.start(group), match.end(group)) in ages, (
                row["id"],
                match.group(0),
            )
