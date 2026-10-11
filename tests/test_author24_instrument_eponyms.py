"""Tests for the tk24 named-instrument and eponym training bank author."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from outis.data.annotation_policy import annotation_errors, identity_split
from outis.data.annotation_policy_eponyms import EPONYM_FIXTURES
from outis.data.annotation_policy_places import place_value_pool
from outis.types import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author24_instrument_eponyms.py"
BANK = ROOT / "data" / "tk24_train_instrument_eponyms.jsonl"


def recorded_sha256(script):
    """Hash of the generator as recorded, before the package was renamed to outis."""
    return hashlib.sha256(
        script.read_bytes().replace(b"outis", b"clinical_deid")
    ).hexdigest()


def _load():
    spec = importlib.util.spec_from_file_location("author24_instrument_eponyms", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


author24 = _load()


@pytest.fixture(scope="module")
def author():
    a = author24.Author()
    a.rows_out = a.generate()
    return a


@pytest.fixture(scope="module")
def rows(author):
    return author.rows_out


def _payload(rows):
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    ).encode("utf-8")


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_deterministic_for_seed(rows):
    again = author24.Author().generate()
    assert _payload(again) == _payload(rows)
    other = author24.Author(seed=author24.SEED + 1).generate()
    assert _payload(other) != _payload(rows)


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_written_bank_matches_generator(rows):
    if not BANK.exists():
        pytest.skip("bank not written")
    payload = BANK.read_bytes()
    assert payload == _payload(rows)
    manifest = json.loads(BANK.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["rows"] == manifest["documents"] == len(rows)
    assert manifest["author_provenance"]["seed"] == author24.SEED
    assert manifest["author_provenance"]["generator_sha256"] == recorded_sha256(SCRIPT)


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_offsets_and_structure(rows):
    assert len(rows) == author24.ROWS
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        assert row["split"] == "train" and row["family"] == row["template_family"]
        for span in row["spans"]:
            assert span["label"] in ENTITY_TYPES
            assert row["text"][span["start"] : span["end"]] == span["text"]
        for mention in row["instrument_mentions"]:
            assert row["text"][mention["start"] : mention["end"]].strip()
        for group in row["identity_groups"]:
            assert identity_split(group) == "train"
    assert len({r["text"] for r in rows}) == len(rows)


def test_e1_fixtures_rebuilt():
    P, Seg = author24.P, author24.Seg
    built = {
        "eponymous_instruments_shorthand": [
            Seg("wells", instr="wells"),
            P(" 4, ddimer neg. "),
            Seg("bishop", instr="bishop"),
            P(" 7 -> induce. "),
            Seg("apgar", instr="apgar"),
            P(" 8/9. "),
            Seg("murphy sign", instr="murphy"),
            P(" neg. "),
            Seg("romberg", instr="romberg"),
            P(" neg."),
        ],
        "eponymous_instruments_fields": [
            Seg("Braden", instr="braden"),
            P(": 14\n"),
            Seg("Mallampati", instr="mallampati"),
            P(": II\n"),
            Seg("Child-Pugh", instr="childpugh"),
            P(": B (8)\n"),
            Seg("Glasgow Coma Scale", instr="gcs"),
            P(": 15\n"),
            Seg("Ottawa ankle rules", instr="ottawa_ankle"),
            P(": negative"),
        ],
        "same_surname_two_roles": [
            P("Dr. "),
            Seg("Corin Wells", "NAME"),
            P(" reviewed the "),
            Seg("Wells score", instr="wells"),
            P("; pt "),
            Seg("Ana Murphy", "NAME"),
            P(", "),
            Seg("murphy sign", instr="murphy"),
            P(" neg."),
        ],
    }
    keys = {entry[0] for entry in author24.CATALOGUE}
    for fixture in EPONYM_FIXTURES:
        text, spans, mentions = author24.build_from_segments(built[fixture["id"]])
        assert text == fixture["text"]
        assert [{k: s[k] for k in ("start", "end", "label")} for s in spans] == fixture[
            "spans"
        ]
        assert {m["key"] for m in mentions} <= keys
        for mention in mentions:
            assert not any(
                s["start"] < mention["end"] and mention["start"] < s["end"]
                for s in fixture["spans"]
            )


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_instruments_never_annotated(rows, author):
    for row in rows:
        assert row["instrument_mentions"], row["id"]
        for mention in row["instrument_mentions"]:
            for span in row["spans"]:
                assert not (
                    span["start"] < mention["end"] and mention["start"] < span["end"]
                ), row["id"]
    # Every instrument surface is unannotated wherever it occurs in a PHI-free row.
    assert all(
        not r["spans"]
        for r in rows
        if r["family"] in ("tk24_instruments_shorthand", "tk24_instruments_fields")
    )


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_same_surname_contrast(rows, author):
    ss = [r for r in rows if r["family"] == "tk24_same_surname"]
    assert len(ss) == 200
    for row in ss:
        names = [s["text"].casefold() for s in row["spans"] if s["label"] == "NAME"]
        found = False
        for mention in row["instrument_mentions"]:
            eponym = author.by_key[mention["key"]].eponym
            surface = row["text"][mention["start"] : mention["end"]].casefold()
            if (
                eponym
                and eponym.casefold() in surface
                and any(eponym.casefold() in n for n in names)
            ):
                found = True
        assert found, row["id"]


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_place_instruments_train_pool_only(rows, author):
    used = {m["key"] for r in rows for m in r["instrument_mentions"]}
    for key in used:
        inst = author.by_key[key]
        if inst.origin == "place":
            assert place_value_pool(inst.eponym) == "train"


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_composition(rows):
    from collections import Counter

    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 15 and max(layouts.values()) <= 0.10 * len(rows)
    styles = Counter(r["layout_style"] for r in rows)
    assert styles["shorthand"] + styles["structured"] >= 0.6 * len(rows)
    per_inst = Counter(
        k for r in rows for k in {m["key"] for m in r["instrument_mentions"]}
    )
    assert len(per_inst) >= 120 and max(per_inst.values()) <= 0.02 * len(rows)
    assert all(r["text"] for r in rows)


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_exclusion_filter_rejects_values_and_sequences():
    lists = author24.exclusion_lists()
    probe = "ddimer neg"
    injected = {"v1": lists["v1"], "v2": lists["v2"] | {author24.exclusion_hash(probe)}}
    assert author24.exclusion_hits([], f"wells 4, {probe}.", injected) == {"v2"}
    assert author24.exclusion_hits(["Ana Ddimer neg"], "x", injected) == {"v2"}
    assert author24.exclusion_hits(["Ana Murphy"], "murphy sign neg", injected) == set()
    a = author24.Author(rows=150, lists=injected)
    out = a.generate()
    assert out and all(probe not in r["text"].casefold() for r in out)
    assert a.rejected.get("rows_hitting_v2", 0) > 0


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_bank_clear_of_both_lists(rows):
    lists = author24.exclusion_lists()
    for row in rows:
        values = [s["text"] for s in row["spans"]] + [
            row["text"][m["start"] : m["end"]] for m in row["instrument_mentions"]
        ]
        assert author24.exclusion_hits(values, row["text"], lists) == set()


@pytest.mark.requires_data("places_exclusions.sha256.json", "places_exclusions_v2.sha256.json")
def test_people_invented_in_train_partition(rows):
    for row in rows:
        if row["spans"]:
            assert row["identity_group"] in row["identity_groups"]
        else:
            assert row["identity_groups"] == []
