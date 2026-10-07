"""Tests for the tk23 age-contexts training bank generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_deid.data.annotation_policy import (  # noqa: E402
    annotation_errors,
    identity_split,
)
from clinical_deid.data.annotation_policy_ages import AGE_FIXTURES  # noqa: E402
from clinical_deid.data.generators.places import excluded_sequences  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "author23", ROOT / "scripts/takeover/author23_age_contexts.py"
)
author = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(author)

FIXTURES = {fixture["id"]: fixture for fixture in AGE_FIXTURES}


@pytest.fixture(scope="module")
def built():
    rows, pools, stats = author.build(author.SEED)
    return rows, pools, stats


def _plain(spans):
    return [(s["start"], s["end"], s["label"]) for s in spans]


def _same_as_fixture(key, marked):
    text, spans = author.parse_marked(marked)
    fixture = FIXTURES[key]
    assert text == fixture["text"]
    assert _plain(spans) == _plain(fixture["spans"])


def test_deterministic_for_seed(built):
    rows, _, _ = built
    again, _, _ = author.build(author.SEED)
    assert again == rows
    other, _, _ = author.build(author.SEED + 1)
    assert [r["text"] for r in other] != [r["text"] for r in rows]


def test_written_bank_matches_generator(built, tmp_path):
    rows, pools, stats = built
    manifest = author.write_bank(
        rows, pools, stats, author.SEED, tmp_path / "bank.jsonl"
    )
    bank = ROOT / "data" / "tk23_train_age_contexts.jsonl"
    if not bank.exists():
        pytest.skip("bank not generated in this checkout")
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == manifest["sha256"]
    recorded = json.loads(bank.with_suffix(".jsonl.manifest.json").read_text())
    assert recorded["sha256"] == manifest["sha256"]
    assert recorded["author_provenance"]["seed"] == author.SEED


def test_offsets_and_structure(built):
    rows, _, _ = built
    assert len(rows) == 1500
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
        assert row["split"] == "train" and row["annotation_policy"] == "clinical-phi-v2"
        assert "[[" not in row["text"] and "]]" not in row["text"]


def test_composition(built):
    rows, _, _ = built
    decoys = sum(r["contrast"] == "decoy" for r in rows)
    assert 0.35 <= decoys / len(rows) <= 0.45
    layouts = {}
    for row in rows:
        layouts[row["layout"]] = layouts.get(row["layout"], 0) + 1
    assert len(layouts) >= 20
    assert max(layouts.values()) <= 0.08 * len(rows)
    cases = {
        s: sum(r["case_style"] == s for r in rows) / len(rows)
        for s in ("normal", "upper", "lower")
    }
    assert (
        0.5 < cases["normal"] < 0.7
        and 0.12 < cases["upper"] < 0.28
        and 0.12 < cases["lower"] < 0.28
    )


def test_decoys_carry_no_age_and_positives_do(built):
    rows, _, _ = built
    for row in rows:
        ages = [s for s in row["spans"] if s["label"] == "AGE"]
        if row["contrast"] == "decoy":
            assert not ages, row["id"]
        else:
            assert ages, row["id"]
    assert all(
        r["contrast"] == "decoy"
        for r in rows
        if r["family"] == "tk23_hypothetical_guideline_ages"
    )


def test_matched_decoys_share_pair_layout_and_identity(built):
    rows, _, _ = built
    positives = {r["pair_id"]: r for r in rows if r["contrast"] == "positive"}
    for row in rows:
        if row["contrast"] == "decoy" and row["pair_id"] in positives:
            mate = positives[row["pair_id"]]
            assert (mate["layout"], mate["identity_group"], mate["case_style"]) == (
                row["layout"],
                row["identity_group"],
                row["case_style"],
            )


def test_identities_in_train_partition(built):
    rows, _, _ = built
    for row in rows:
        assert row["identity_group"] in row["identity_groups"]
        for group in row["identity_groups"]:
            assert identity_split(group) == "train"


def test_d1_hypothetical_fixture_rebuilt_and_honoured(built):
    _same_as_fixture(
        "hypothetical_ages",
        "A comparison exercise described a person aged 46; for example, a 50-year-old would qualify; "
        "if a patient were 70, the dose would change.",
    )
    _same_as_fixture(
        "public_ages_and_durations",
        "Screen from age 45 per guideline; 5-year survival; 10-year history of asthma.",
    )
    hyp, guide, dur = author.d1_sentences(random.Random(0))
    assert any(
        s.startswith("A comparison exercise described a person aged") for s in hyp
    )
    assert all("[[" not in s for s in hyp + guide + dur)
    rows, _, _ = built
    cue = re.compile(
        r"(person aged|if a patient were|for example, a|hypothetical|sample entry uses a|"
        r"screen from age|adults over|year survival|year history)\s*(\d+)",
        re.I,
    )
    seen = 0
    for row in rows:
        ages = {(s["start"], s["end"]) for s in row["spans"] if s["label"] == "AGE"}
        for match in cue.finditer(row["text"]):
            seen += 1
            assert (match.start(2), match.end(2)) not in ages, row["id"]
    assert seen > 250


def test_d2_decade_fixture_rebuilt_and_honoured(built):
    _same_as_fixture(
        "decades",
        f"A woman {author.decade_phrase('her', 'seventies')}; her brother is {author.decade_phrase('his', '40s')}; "
        f"father {author.decade_phrase('his', 'sixties', 'early')}; the 1970s were hard.",
    )
    rows, _, _ = built
    decade_spans = 0
    for row in rows:
        for span in row["spans"]:
            if span["label"] == "AGE" and span["text"].lower() in author.DECADE_VALUES:
                decade_spans += 1
                assert not re.search(r"early|late|in ", span["text"], re.I)
        # eras and measurement ranges never carry AGE
        ages = {(s["start"], s["end"]) for s in row["spans"] if s["label"] == "AGE"}
        for match in re.finditer(
            r"\b(?:the|from the) (1\d{3}s|\d{2,3}s)\b", row["text"], re.I
        ):
            assert (match.start(1), match.end(1)) not in ages, row["id"]
    assert decade_spans >= 130


def test_all_ages_fixtures_rebuilt():
    _same_as_fixture(
        "ages_any_value",
        f"A {author.A(72)}-year-old woman; {author.A(89)} years old at the earlier visit; {author.A(93)}F; "
        f"a {author.A(author.words(6))}-month-old sibling.",
    )
    _same_as_fixture(
        "ages_of_relatives",
        f"Mother died at {author.A(author.words(72))}; brother, {author.A(58)}, is well.",
    )


def test_age_values_parse_and_cover_range(built):
    rows, _, _ = built
    values = []
    for row in rows:
        for span in row["spans"]:
            if span["label"] == "AGE":
                value, unit = author.age_span_value(row["text"], span)
                assert value is not None, (row["id"], span)
                if unit == "years":
                    values.append(value)
    assert min(values) <= 2 and max(values) >= 100
    assert sum(v > 89 for v in values) >= 50


def test_exclusion_filter_applied(built):
    rows, pools, stats = built
    assert all(excluded_sequences(row["text"]) == 0 for row in rows)
