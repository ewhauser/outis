import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from clinical_deid.data.annotation_policy import (
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from clinical_deid.data.annotation_policy_targeted import TARGETED_FIXTURES

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "takeover"
    / "author14_targeted_gaps.py"
)
spec = importlib.util.spec_from_file_location("author14_targeted_gaps", SCRIPT)
gen = importlib.util.module_from_spec(spec)
sys.modules["author14_targeted_gaps"] = gen
spec.loader.exec_module(gen)


@pytest.fixture(scope="module")
def bank():
    rows, metas = gen.generate()
    return rows, metas


def test_generation_is_deterministic(bank):
    rows, _ = bank
    again, _ = gen.generate()
    assert gen.serialise(rows) == gen.serialise(again)


def test_row_counts_and_composition(bank):
    rows, _ = bank
    assert len(rows) == 2000
    assert Counter(r["target_family"] for r in rows) == gen.FAMILY_ROWS
    negatives = sum(r["polarity"] == "negative" for r in rows)
    assert 0.30 <= negatives / len(rows) <= 0.36
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) / len(rows) <= 0.08
    cases = Counter(r["case_mode"] for r in rows)
    assert cases["normal"] == 1200 and cases["upper"] == 400 and cases["lower"] == 400
    assert Counter(r["cohort"] for r in rows)["realistic"] > 0.85 * len(rows)


def test_offsets_and_metadata(bank):
    rows, _ = bank
    ids = set()
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
            assert span["text"] == span["text"].strip()
        assert row["split"] == "train"
        assert row["annotation_policy"] == POLICY_VERSION
        assert row["family"] == row["template_family"]
        assert row["id"] not in ids
        ids.add(row["id"])
        assert row["identity_group"] in row["identity_groups"]
        for group in row["identity_groups"]:
            assert int(group[:8], 16) % 10 < 8
        assert identity_split(row["identity_group"]) == "train"
        assert "[[" not in row["text"] and "]]" not in row["text"]
    assert len({r["text"] for r in rows}) == len(rows)


def test_policy_conventions(bank):
    rows, _ = bank
    for row in rows:
        text = row["text"]
        for span in row["spans"]:
            value, label = span["text"], span["label"]
            after = text[span["end"] :]
            if label == "NAME":
                assert not re.match(r"(?i)(dr|mr|mrs|ms)\.?\s", value)
                assert not re.search(r"(?i)[,\s]\s*(md|rn|np|np-c|pa-c|do)$", value)
            if label == "AGE":
                assert re.fullmatch(r"(?i)(9\d|ninety(-\w+)?|nineties)", value)
            if label == "MRN":
                assert not re.match(r"(?i)mrn|mr#|medical", value)
            if label == "PHONE":
                assert re.search(r"555[ .-]0?1\d\d", value) or re.search(
                    r"555[ .-]01\d\d", value
                )
                # an extension written after a phone belongs inside the span
                assert not re.match(r"(?i)[ ,]*(ext\.?|extension)\s*[:#]?\s*\d", after)
                assert not re.match(r"(?i)[ ,]*x\s*\d", after)
            if label == "DATE":
                assert not re.match(r"(?i)(from|between|on)\b", value)
        positives = [s for s in row["spans"] if s["label"] == "FACILITY"]
        if row["target_family"] == "phone_extension" and row["polarity"] == "negative":
            assert not any(
                re.search(r"(?i)\b(ext\.?|extension|x)\s*[:#]?\s*\d", s["text"])
                for s in row["spans"]
                if s["label"] == "PHONE"
            )
        if row["target_family"] == "funeral_homes" and row["polarity"] == "negative":
            assert not any(
                re.search(r"(?i)funeral|mortuary|cremation|memorial", s["text"])
                for s in positives
            )
        if row["target_family"] == "funeral_homes" and row["polarity"] == "positive":
            assert any(
                re.search(r"(?i)funeral|mortuary|cremation|memorial", s["text"])
                for s in positives
            )
        if row["target_family"] == "pharmacy_zip" and row["polarity"] == "positive":
            zips = [
                s
                for s in row["spans"]
                if s["label"] == "LOCATION" and re.fullmatch(r"\d{5}", s["text"])
            ]
            assert zips
            for z in zips:
                before = [
                    s
                    for s in row["spans"]
                    if s["end"] <= z["start"] and s["label"] == "FACILITY"
                ]
                assert before and z["start"] - before[-1]["end"] < 25
    for row, meta in zip(*bank):
        if row["target_family"] == "history_years" and row["polarity"] == "negative":
            assert meta["years"] == []
            for match in re.finditer(
                r"(?<![\w/#-])(19|20)\d\d(?![\w/%-])", row["text"]
            ):
                inside = any(
                    s["start"] <= match.start() < s["end"] for s in row["spans"]
                )
                if not inside:
                    assert re.search(r"(?i)\bthe $", row["text"][: match.start()]), row[
                        "text"
                    ][max(0, match.start() - 30) : match.end() + 10]


def test_year_spread_and_repetition(bank):
    rows, metas = bank
    years = Counter(y for m in metas for y in m["years"])
    assert min(years) >= 1950 and max(years) <= 2026
    assert max(years.values()) / sum(years.values()) <= 0.03
    values: dict = {}
    for row in rows:
        for label in ("FACILITY", "LOCATION"):
            for value in {
                s["text"].casefold() for s in row["spans"] if s["label"] == label
            }:
                values.setdefault((label, value), 0)
                values[(label, value)] += 1
    assert max(values.values()) <= 0.005 * len(rows)


def test_labels_supported(bank):
    rows, _ = bank
    support = Counter(s["label"] for r in rows for s in r["spans"])
    for label in (
        "NAME",
        "DATE",
        "PHONE",
        "FACILITY",
        "LOCATION",
        "AGE",
        "MRN",
        "ADDRESS",
        "EMAIL",
        "URL",
        "ID",
        "ACCOUNT",
        "SSN",
    ):
        assert support[label] > 0


@pytest.mark.parametrize("fixture", TARGETED_FIXTURES, ids=lambda f: f["id"])
def test_span_logic_rebuilds_every_targeted_fixture(fixture):
    """Rebuild each fixture through the generator's own RowBuilder and compare exact offsets."""
    builder = gen.RowBuilder()
    text, position = fixture["text"], 0
    for span in fixture["spans"]:
        builder.add(
            text[position : span["start"]],
            (span["label"], text[span["start"] : span["end"]]),
        )
        position = span["end"]
    builder.add(text[position:])
    rebuilt_text, rebuilt_spans, _ = builder.build("normal")
    assert rebuilt_text == text
    assert [(s["start"], s["end"], s["label"]) for s in rebuilt_spans] == [
        (s["start"], s["end"], s["label"]) for s in fixture["spans"]
    ]
    # the case transform keeps gold on the authored roles
    for case, fold in (("upper", str.upper), ("lower", str.lower)):
        folded_text, folded_spans, _ = builder.build(case)
        assert folded_text == fold(text)
        assert [(s["label"], s["text"]) for s in folded_spans] == [
            (s["label"], fold(text[s["start"] : s["end"]])) for s in fixture["spans"]
        ]
        assert annotation_errors(folded_text, folded_spans) == []


def test_fixture_shapes_are_covered_by_the_generator(bank):
    rows, _ = bank
    joined = [(r["target_family"], r["polarity"], s) for r in rows for s in r["spans"]]
    # year alone, month-year, phone with written extension, pharmacy then ZIP, funeral home
    assert any(
        f == "history_years" and re.fullmatch(r"\d{4}", s["text"])
        for f, _, s in joined
        if s["label"] == "DATE"
    )
    assert any(
        f == "history_years" and re.fullmatch(r"(?i)[a-z]{3,9} \d{4}", s["text"])
        for f, _, s in joined
        if s["label"] == "DATE"
    )
    assert any(
        s["label"] == "PHONE" and re.search(r"(?i)extension", s["text"])
        for _, _, s in joined
    )
    assert any(
        s["label"] == "PHONE" and re.search(r"EXT\.", s["text"]) for _, _, s in joined
    )
    assert any(
        s["label"] == "FACILITY" and re.search(r"& Sons", s["text"])
        for _, _, s in joined
    )
    assert canonical_identity_group("x")
