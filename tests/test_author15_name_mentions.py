import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from clinical_deid.data.annotation_policy import (
    POLICY_FIXTURES,
    POLICY_VERSION,
    _example,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from clinical_deid.data.annotation_policy_transcripts import TRANSCRIPT_FIXTURES
from clinical_deid.data.generators.places import excluded_sequences, excluded_value

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "author15", ROOT / "scripts/takeover/author15_name_mentions.py"
)
author15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author15)

HONORIFICS = re.compile(r"^(dr|mr|mrs|ms)\.?\s", re.I)
CREDENTIALS = re.compile(r"\b(md|rn)$", re.I)


@pytest.fixture(scope="module")
def bank():
    rows, counters, _ = author15.build_bank()
    return rows, counters


def names(row):
    return [s for s in row["spans"] if s["label"] == "NAME"]


def test_determinism(bank):
    rows, _ = bank
    again, _, _ = author15.build_bank()
    assert [json.dumps(r, sort_keys=True) for r in rows] == [
        json.dumps(r, sort_keys=True) for r in again
    ]


def test_committed_corpus_matches_generator(bank):
    rows, _ = bank
    path = ROOT / "data/tk15_train_name_mentions.jsonl"
    if not path.exists():
        pytest.skip("data file not present in this checkout")
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert written == rows


def test_shape_and_composition(bank):
    rows, _ = bank
    assert len(rows) == 1800
    fam = Counter(r["family"] for r in rows)
    assert fam == {
        "tk15_staff_first_name": 500,
        "tk15_family_first_name": 500,
        "tk15_alias_after_full_name": 300,
        "tk15_last_first_forms": 300,
        "tk15_name_like_words": 200,
    }
    assert sum(r["role"] == "negative" for r in rows) == 600
    assert all(r["family"] == r["template_family"] for r in rows)
    assert all(
        r["split"] == "train" and r["annotation_policy"] == POLICY_VERSION for r in rows
    )
    case = Counter(r["case"] for r in rows)
    assert (
        abs(case["upper"] / 1800 - 0.4) < 0.02
        and abs(case["lower"] / 1800 - 0.2) < 0.02
    )
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20 and max(layouts.values()) / 1800 <= 0.08
    assert all(not names(r) for r in rows if r["role"] == "negative")
    assert all(names(r) for r in rows if r["role"] == "positive")
    assert all(not names(r) for r in rows if r["family"] == "tk15_name_like_words")


def test_offsets_and_structure(bank):
    rows, _ = bank
    for r in rows:
        assert not annotation_errors(r["text"], r["spans"])
        for s in r["spans"]:
            assert r["text"][s["start"] : s["end"]] == s["text"]
        assert "[[" not in r["text"] and "]]" not in r["text"]


def test_identity_metadata(bank):
    rows, _ = bank
    for r in rows:
        assert r["identity_group"] in r["identity_groups"]
        assert len(set(r["identity_groups"])) == len(r["identity_groups"])
        assert all(identity_split(g) == "train" for g in r["identity_groups"])
    pairs = Counter(r["pair_id"] for r in rows)
    assert sum(1 for c in pairs.values() if c == 2) == 360


def test_conventions(bank):
    rows, _ = bank
    honorific_rows = suffix_rows = initial_rows = alias_rows = credential_rows = 0
    for r in rows:
        for s in names(r):
            assert not HONORIFICS.match(s["text"]), "honorific inside span"
            assert not CREDENTIALS.search(s["text"]), "credential inside span"
            before = r["text"][max(0, s["start"] - 5) : s["start"]]
            if re.search(r"(dr|mr|mrs|ms)\.\s$", before, re.I):
                honorific_rows += 1
            if re.search(r"\b(jr\.|sr\.|ii|iii|iv)\b", s["text"], re.I):
                suffix_rows += 1
            if re.search(r", \S+ (\S+ )?[A-Za-z]$", s["text"]) and "," in s["text"]:
                initial_rows += 1
            # possessive 's stays outside; spans never end in an apostrophe
            assert not re.search(r"['’]s?$", s["text"], re.I)
            assert not s["text"].endswith(('"', ")"))
            after = r["text"][s["end"] : s["end"] + 3]
            if after.upper().startswith((" MD", " RN")):
                credential_rows += 1
        if "alias_after_full_name" in r["phenomena"] and r["role"] == "positive":
            alias_rows += 1
            texts = [s["text"] for s in names(r)]
            assert len(texts) >= 3  # full name, alias, later alias mention
            alias = (
                texts[0]
                if " " not in texts[0]
                else next(t for t in texts if " " not in t)
            )
            assert texts.count(alias) >= 2 or any(
                t.casefold() == alias.casefold() for t in texts[1:]
            )
        if "quoted_nickname" in r["phenomena"]:
            assert all('"' not in s["text"] for s in names(r))
    assert (
        honorific_rows > 50
        and suffix_rows > 30
        and initial_rows > 100
        and alias_rows == 220
        and credential_rows > 5
    )


def test_last_first_forms(bank):
    rows, _ = bank
    lf = [
        s["text"]
        for r in rows
        if r["family"] == "tk15_last_first_forms"
        for s in names(r)
    ]
    assert all("," in t for t in lf)
    assert any(re.search(r" (Jr\.|Sr\.|II|III|IV)( [A-Z])?$", t, re.I) for t in lf)
    assert any(re.search(r", \S+ [A-Z]$", t) for t in lf)


def test_name_value_repetition_cap(bank):
    rows, _ = bank
    counts = author15.name_value_rows(rows)
    assert max(counts.values()) <= 9  # 0.5% of 1,800 rows


def test_zero_exclusion_matches(bank):
    rows, counters = bank
    for r in rows:
        assert excluded_sequences(r["text"]) == 0
        assert not any(excluded_value(s["text"]) for s in r["spans"])
    assert counters["redraws"] >= 0


def test_markup_helpers_match_policy_fixtures():
    for fixture in POLICY_FIXTURES + TRANSCRIPT_FIXTURES:
        pieces, previous = [], 0
        for s in fixture["spans"]:
            pieces += [
                fixture["text"][previous : s["start"]],
                f"[[{s['label']}|{fixture['text'][s['start'] : s['end']]}]]",
            ]
            previous = s["end"]
        marked = "".join(pieces) + fixture["text"][previous:]
        text, spans = author15.parse_marked(marked)
        assert text == fixture["text"]
        assert [(s["start"], s["end"], s["label"]) for s in spans] == [
            (s["start"], s["end"], s["label"]) for s in fixture["spans"]
        ]
        assert author15.apply_case(marked, "normal") == marked
        assert _example(fixture["id"], marked, ["x"])["spans"] == fixture["spans"]


def test_case_application_keeps_offsets_valid():
    marked = "Dr. [[NAME|Zoë O’Rían-Soto]] and [[NAME|Q, Ana]] met on [[DATE|3/4/2025]]"
    for mode in ("upper", "lower", "normal"):
        text, spans = author15.parse_marked(author15.apply_case(marked, mode))
        assert not annotation_errors(text, spans)
        assert [s["label"] for s in spans] == ["NAME", "NAME", "DATE"]


def test_canonical_identity_hash_is_stable():
    assert identity_split(canonical_identity_group("Test Person")) in (
        "train",
        "validation",
    )
