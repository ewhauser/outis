import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from outis.data.annotation_policy import (
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from outis.data.annotation_policy_targeted import (
    TARGETED_FIXTURES,
    TARGETED_FOLLOWUP_FIXTURES,
)
from outis.data.annotation_policy_transcripts import TRANSCRIPT_FIXTURES
from outis.data.generators.places import excluded_sequences, excluded_value

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


V3_TRANSCRIPT_FIXTURES = tuple(
    f
    for f in TRANSCRIPT_FIXTURES
    if f["id"] in ("session_and_access_codes", "timestamp_with_date")
)


@pytest.mark.parametrize(
    "fixture",
    (*TARGETED_FIXTURES, *TARGETED_FOLLOWUP_FIXTURES, *V3_TRANSCRIPT_FIXTURES),
    ids=lambda f: f["id"],
)
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


# ---------------------------------------------------------------------------
# tk14 v2
# ---------------------------------------------------------------------------
V1_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "tk14_train_targeted_gaps.jsonl"
)


@pytest.fixture(scope="module")
def bank_v2():
    return gen.generate("v2")


@pytest.mark.skipif(not V1_DATA.exists(), reason="v1 data file not present")
def test_generator_still_reproduces_v1_byte_for_byte(bank):
    rows, _ = bank
    assert gen.serialise(rows) == V1_DATA.read_bytes()


def test_v2_is_deterministic_and_differs_from_v1(bank, bank_v2):
    rows, _ = bank_v2
    again, _ = gen.generate("v2")
    assert gen.serialise(rows) == gen.serialise(again)
    assert gen.serialise(rows) != gen.serialise(bank[0])


def test_v2_composition_and_limits(bank_v2):
    rows, metas = bank_v2
    assert 2200 <= len(rows) <= 2400
    assert all(
        r["id"].endswith("_v2") and r["suite"] == "tk14_targeted_gaps_v2" for r in rows
    )
    assert all(r["version"] == "tk14-v2" and r["split"] == "train" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20 and max(layouts.values()) / len(rows) <= 0.08
    negatives = sum(r["polarity"] == "negative" for r in rows)
    assert 0.30 <= negatives / len(rows) <= 0.37
    assert Counter(r["cohort"] for r in rows)["realistic"] > 0.85 * len(rows)
    cases = Counter(r["case_mode"] for r in rows)
    assert 0.55 <= cases["normal"] / len(rows) <= 0.65
    years = Counter(y for m in metas for y in m["years"])
    assert min(years) >= 1950 and max(years) <= 2026
    assert max(years.values()) / sum(years.values()) <= 0.03
    values: Counter = Counter()
    for row in rows:
        for label in ("FACILITY", "LOCATION"):
            for value in {
                s["text"].casefold() for s in row["spans"] if s["label"] == label
            }:
                values[(label, value)] += 1
    assert max(values.values()) <= 0.005 * len(rows)
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        assert identity_split(row["identity_group"]) == "train"
        assert all(int(g[:8], 16) % 10 < 8 for g in row["identity_groups"])


def test_v2_every_new_case_has_all_case_modes_and_matched_negatives(bank_v2):
    rows, _ = bank_v2
    for case, (positives, negatives, _family, _layouts) in gen.NEW_CASES.items():
        group = [r for r in rows if r["new_case"] == case]
        assert sum(r["polarity"] == "positive" for r in group) == positives
        assert sum(r["polarity"] == "negative" for r in group) == negatives
        assert {r["case_mode"] for r in group} == {"normal", "upper", "lower"}


def spans_of(row, label):
    return [s for s in row["spans"] if s["label"] == label]


def test_v2_new_case_conventions(bank_v2):
    rows, metas = bank_v2
    for row, meta in zip(rows, metas):
        case, text = row["new_case"], row["text"]
        positive = row["polarity"] == "positive"
        if case == "year_range" and positive:
            ranges = [
                s
                for s in spans_of(row, "DATE")
                if re.fullmatch(r"(\d{4})( ?- ?| to )(\d{4})", s["text"], re.I)
            ]
            assert ranges
            for s in ranges:
                a, b = (int(x) for x in re.findall(r"\d{4}", s["text"]))
                assert 1950 <= a < b <= 2026
        if case == "year_range" and not positive:
            assert meta["years"] == []
        if case == "future_year" and positive:
            assert meta["future_years"]
        if case == "future_year" and not positive:
            assert meta["future_years"] == []
        if case == "onset_year" and positive:
            assert meta["years"]
        if case == "zip_plus_four":
            zips = [
                s
                for s in spans_of(row, "LOCATION")
                if re.fullmatch(r"\d{5}-\d{4}", s["text"])
            ]
            if positive:
                assert zips
            else:
                assert not zips
                assert re.search(r"\b\d{5}-\d{3,4}\b", text)
        if case == "rx_number":
            ids = [
                s
                for s in spans_of(row, "ID")
                if re.fullmatch(r"(RX-)?\d{5,8}", s["text"], re.I)
            ]
            if positive:
                assert ids
                for s in ids:
                    assert re.search(
                        r"(rx\s?(#|no\.|number:?|id)\s?|prescription number )$",
                        text[: s["start"]],
                        re.I,
                    )
            else:
                assert not ids
        if case == "pager":
            pagers = [t for t in meta["tags"] if t[0] == "pager"]
            if positive:
                assert pagers
                for s in spans_of(row, "PHONE"):
                    if any(s["text"] == v for _, v in pagers):
                        assert re.search(
                            r"pager|beeper|paged via",
                            text[max(0, s["start"] - 30) : s["start"]],
                            re.I,
                        )
            else:
                assert not pagers
        if case == "bare_extension":
            assert not positive
            assert re.search(r"\b(ext\.?|extension|x)\s*#?\s*\d{3,5}", text, re.I)
            for s in spans_of(row, "PHONE"):
                assert not re.search(r"(ext|extension|\bx)\s*#?\s*\d", s["text"], re.I)
                # a bare extension is never attached to the end of a phone span
                assert not re.match(
                    r"[ ,]*(ext\.?|extension|x)\s*#?\s*\d", text[s["end"] :], re.I
                )


# ---------------------------------------------------------------------------
# tk14 v3
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
V2_DATA = DATA_DIR / "tk14_train_targeted_gaps__v2.jsonl"
V3_DATA = DATA_DIR / "tk14_train_targeted_gaps__v3.jsonl"
WEEKDAY_START = re.compile(r"(?i)(mon|tue|wed|thu|fri|sat|sun)[a-z]*[.,]? ")


@pytest.fixture(scope="module")
def bank_v3():
    return gen.generate("v3")


@pytest.mark.skipif(not V2_DATA.exists(), reason="v2 data file not present")
def test_generator_still_reproduces_v2_byte_for_byte(bank_v2):
    rows, _ = bank_v2
    assert gen.serialise(rows) == V2_DATA.read_bytes()


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_is_deterministic_and_differs_from_v2(bank_v2, bank_v3):
    rows, _ = bank_v3
    again, _ = gen.generate("v3")
    assert gen.serialise(rows) == gen.serialise(again)
    assert gen.serialise(rows) != gen.serialise(bank_v2[0])


@pytest.mark.skipif(not V3_DATA.exists(), reason="v3 data file not present")
def test_v3_file_matches_the_generator(bank_v3):
    assert gen.serialise(bank_v3[0]) == V3_DATA.read_bytes()


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_composition_and_limits(bank_v3):
    rows, metas = bank_v3
    assert len(rows) == 2950
    assert all(r["id"].endswith("_v3") and r["suite"] == gen.SUITE3 for r in rows)
    assert all(r["version"] == "tk14-v3" and r["split"] == "train" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    families = Counter(r["target_family"] for r in rows)
    assert families["signature_dates"] == 150 and families["rare_identifiers"] == 150
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20 and max(layouts.values()) / len(rows) <= 0.08
    negatives = sum(r["polarity"] == "negative" for r in rows)
    assert 0.30 <= negatives / len(rows) <= 0.37
    assert Counter(r["cohort"] for r in rows)["realistic"] > 0.85 * len(rows)
    years = Counter(y for m in metas for y in m["years"])
    assert min(years) >= 1950 and max(years) <= 2026
    assert max(years.values()) / sum(years.values()) <= 0.03
    values: Counter = Counter()
    for row in rows:
        for label in ("FACILITY", "LOCATION"):
            for value in {
                s["text"].casefold() for s in row["spans"] if s["label"] == label
            }:
                values[(label, value)] += 1
    assert max(values.values()) <= 0.005 * len(rows)
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        assert identity_split(row["identity_group"]) == "train"
        assert all(int(g[:8], 16) % 10 < 8 for g in row["identity_groups"])


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_additions_have_counts_cases_and_matched_negatives(bank_v3):
    rows, _ = bank_v3
    expected = {"hist_v3": 300, "sig_weekday": 150, "rare_id": 150}
    for case, count in expected.items():
        group = [r for r in rows if r["new_case"] == case]
        assert len(group) == count
        negatives = sum(r["polarity"] == "negative" for r in group)
        assert 0.30 <= negatives / count <= 0.37
        assert {r["case_mode"] for r in group} == {"normal", "upper", "lower"}
        assert {r["cohort"] for r in group} == {"realistic"}
    hist = [r for r in rows if r["new_case"] == "hist_v3"]
    upper_core = [
        r
        for r in hist
        if r["case_mode"] == "upper" and r["layout"] in ("awv_note", "soap_note")
    ]
    assert len(upper_core) >= 150
    # every v2 case is still present
    for case in gen.NEW_CASES:
        assert any(r["new_case"] == case for r in rows)


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_weekday_signature_lines(bank_v3):
    rows, metas = bank_v3
    time_pattern = re.compile(r"\d{1,2}:\d{2}|\b\d{4}\b")
    for row, meta in zip(rows, metas):
        if row["new_case"] != "sig_weekday":
            continue
        sig_dates = [v for tag, v in meta["tags"] if tag == "sigdate"]
        if row["polarity"] == "negative":
            assert not sig_dates
            continue
        assert sig_dates
        for span in row["spans"]:
            if span["label"] == "DATE" and span["text"] in sig_dates:
                # the weekday is inside the span, the clock time outside it
                assert WEEKDAY_START.match(span["text"]), span["text"]
                assert not re.search(r"\d{1,2}:\d{2}", span["text"])
                line_end = row["text"].find("\n", span["end"])
                tail = row["text"][span["end"] : line_end if line_end != -1 else None]
                start = row["text"].rfind("\n", 0, span["start"]) + 1
                before = row["text"][start : span["start"]]
                assert time_pattern.search(tail) or time_pattern.search(before)
    modes = {
        r["case_mode"]
        for r in rows
        if r["new_case"] == "sig_weekday" and r["polarity"] == "positive"
    }
    assert modes == {"normal", "upper", "lower"}


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_rare_identifier_rows(bank_v3):
    rows, metas = bank_v3
    version = re.compile(
        r"(?i)\b(app version|firmware v|software build)\s?\d+\.\d+\.\d+"
    )
    seen_kinds = set()
    for row, meta in zip(rows, metas):
        if row["new_case"] != "rare_id":
            continue
        ids = [s for s in row["spans"] if s["label"] == "ID"]
        tagged = [(t, v) for t, v in meta["tags"] if t in gen.RARE_TAGS]
        if row["polarity"] == "negative":
            assert not ids and not tagged
        else:
            assert tagged and len(ids) >= len(tagged)
            seen_kinds |= {t for t, _ in tagged}
        for span in ids:
            # captions stay outside; software versions are never annotated
            assert not re.match(
                r"(?i)(serial|s/n|session|call |code|visit)", span["text"]
            )
            assert not re.fullmatch(r"\d+\.\d+\.\d+", span["text"])
        for match in version.finditer(row["text"]):
            assert not any(
                s["start"] < match.end() and match.start() < s["end"]
                for s in row["spans"]
            )
        for span in row["spans"]:
            if span["label"] == "URL":
                assert span["text"].lower().startswith("https://")
    assert seen_kinds == set(gen.RARE_TAGS)


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_history_additions_have_years_only_in_positives(bank_v3):
    rows, metas = bank_v3
    for row, meta in zip(rows, metas):
        if row["new_case"] != "hist_v3":
            continue
        assert bool(meta["years"]) == (row["polarity"] == "positive")


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_v3_has_zero_exclusion_matches(bank_v3):
    """Counts only: never print which strings matched."""
    rows, _ = bank_v3
    assert sum(excluded_sequences(r["text"]) > 0 for r in rows) == 0
    assert sum(excluded_value(s["text"]) for r in rows for s in r["spans"]) == 0


@pytest.mark.skipif(not V3_DATA.exists(), reason="v3 data file not present")
def test_v3_data_file_has_zero_exclusion_matches():
    import json

    lines = V3_DATA.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 2950
    assert sum(excluded_sequences(r["text"]) > 0 for r in rows) == 0
    assert sum(excluded_value(s["text"]) for r in rows for s in r["spans"]) == 0
