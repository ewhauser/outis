"""Tests for the tk18 radiology bank generator (scripts/takeover/author18_radiology.py)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from clinical_deid.data.annotation_policy import (
    POLICY_VERSION,
    annotation_errors,
    identity_split,
)
from clinical_deid.data.annotation_policy_reports import REPORT_FIXTURES
from clinical_deid.data.generators.places import excluded_sequences, excluded_value

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author18_radiology.py"
spec = importlib.util.spec_from_file_location("author18_radiology", SCRIPT)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

FIXTURES = {f["id"]: f for f in REPORT_FIXTURES}


@pytest.fixture(scope="module")
def built():
    return gen.build()


@pytest.fixture(scope="module")
def rows(built):
    return built[0]


def spans_of(row, label=None):
    return [s for s in row["spans"] if label is None or s["label"] == label]


# ---- determinism and file identity ----------------------------------------
def test_build_is_deterministic():
    a, ra = gen.build(300)
    b, rb = gen.build(300)
    assert a == b and ra == rb


def test_written_corpus_matches_generator(rows):
    path = ROOT / "data" / "tk18_train_radiology.jsonl"
    if not path.exists():
        pytest.skip("corpus not present in this checkout")
    expected = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    assert path.read_bytes() == expected
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(expected).hexdigest()
    assert manifest["rows"] == len(rows) == 1500
    assert manifest["independent_final"] is False


# ---- offsets and structure -------------------------------------------------
def test_offsets_and_structure(rows):
    seen_ids, seen_texts = set(), set()
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == []
        for s in row["spans"]:
            assert row["text"][s["start"]:s["end"]] == s["text"]
        assert "[[" not in row["text"] and "]]" not in row["text"]
        assert row["annotation_policy"] == POLICY_VERSION and row["split"] == "train"
        assert row["family"] == row["template_family"]
        assert row["id"] not in seen_ids and row["text"] not in seen_texts
        seen_ids.add(row["id"])
        seen_texts.add(row["text"])


def test_composition(rows):
    n = len(rows)
    assert n == 1500
    fam = Counter(r["report_family"] for r in rows)
    assert fam == {"ct_mri": 375, "plain_us": 300, "mammo": 225, "interventional": 150, "tele_critical": 225, "dicom_worklist": 225}
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) / n <= 0.08
    free = sum(r["phi_free"] for r in rows)
    assert 0.30 <= free / n <= 0.37
    assert all(bool(r["spans"]) != r["phi_free"] for r in rows)
    case = Counter(r["case_style"] for r in rows)
    assert abs(case["normal"] / n - 0.55) < 0.02
    assert abs(case["caps"] / n - 0.30) < 0.02
    assert abs(case["lower"] / n - 0.15) < 0.02


def test_every_row_carries_decoys(rows):
    without = [r["id"] for r in rows if not gen.decoy_flags(r["text"])]
    # a handful of very short PHI-free texts may carry none; the vast majority do
    assert len(without) / len(rows) < 0.01


def test_value_cap(rows):
    n = len(rows)
    count = Counter()
    for r in rows:
        for v in {s["text"].casefold() for s in r["spans"]}:
            count[v] += 1
    assert max(count.values()) <= gen.VALUE_CAP
    assert max(count.values()) / n <= 0.005
    tokens = Counter()
    for r in rows:
        for t in {t for s in r["spans"] if s["label"] == "NAME" for t in gen.name_tokens(s["text"])}:
            tokens[t] += 1
    assert max(tokens.values()) <= gen.VALUE_CAP


def test_identity_partition(rows):
    for r in rows:
        if r["phi_free"]:
            assert r["identity_groups"] == [] and r["identity_group"].startswith("phi-free:")
        else:
            assert r["identity_group"] in r["identity_groups"]
            assert len(set(r["identity_groups"])) == len(r["identity_groups"])
            assert all(identity_split(g) == "train" for g in r["identity_groups"])


# ---- exclusion filter -------------------------------------------------------
def test_zero_exclusion_matches(rows, built):
    assert sum(excluded_sequences(r["text"]) for r in rows) == 0
    assert not any(excluded_value(s["text"]) for r in rows for s in r["spans"])
    assert set(built[1]) <= {"exclusion", "value_cap", "duplicate_text", "structural"}


# ---- conventions -------------------------------------------------------------
HONORIFIC = re.compile(r"^(?:dr|mr|ms|mrs|prof)\b\.?\s", re.I)
CREDENTIAL = re.compile(r"\b(?:md|do|mbbs|phd|mph|rn|bsn|rdms)\b", re.I)
TIME = re.compile(r"\d{1,2}:\d{2}")
CAPTION = re.compile(r"^(?:mrn|acc|accession|npi|acct|enc)\b", re.I)
NON_PHI = re.compile(
    r"\b(?:cpt \d{5}|series \d+|image \d+|(?:bi|li|lung|ti|pi|o)-rads[ a-z0-9-]*|lot \d{4}[a-k]|slice position [-\d.]+ mm)\b", re.I
)


def overlaps(span, start, end):
    return span["start"] < end and start < span["end"]


def test_names_exclude_honorifics_and_credentials(rows):
    for r in rows:
        for s in spans_of(r, "NAME"):
            assert not HONORIFIC.match(s["text"]), s
            assert not CREDENTIAL.search(s["text"]), s
            assert not s["text"].lower().endswith("'s") and not s["text"].lower().endswith("’s")


def test_captions_outside_identifier_spans(rows):
    for r in rows:
        for s in spans_of(r, "MRN") + spans_of(r, "ID") + spans_of(r, "ACCOUNT"):
            assert not CAPTION.match(s["text"]), s
            assert not s["text"].lower().startswith(("npi", "mrn"))


def test_dates_have_no_clock_time_inside(rows):
    for r in rows:
        for s in spans_of(r, "DATE"):
            assert not TIME.search(s["text"]), s
            assert not re.search(r"\b(?:am|pm)\b", s["text"], re.I), s


def test_ages_only_over_89_and_only_the_number(rows):
    ages = [s for r in rows for s in spans_of(r, "AGE")]
    assert ages
    for s in ages:
        assert s["text"].isdigit() and int(s["text"]) >= 90, s


def test_non_phi_codes_never_annotated(rows):
    for r in rows:
        for m in NON_PHI.finditer(r["text"]):
            assert not any(overlaps(s, m.start(), m.end()) for s in r["spans"]), (r["id"], m.group())
        low = r["text"].lower()
        for model in gen.SCANNERS:
            for m in re.finditer(re.escape(model.lower()), low):
                assert not any(overlaps(s, m.start(), m.end()) for s in r["spans"]), (r["id"], model)
        for ep in gen.EPONYMS:
            for m in re.finditer(re.escape(ep.lower()), low):
                assert not any(overlaps(s, m.start(), m.end()) for s in r["spans"] if s["label"] in ("NAME", "LOCATION", "FACILITY")), (r["id"], ep)


def test_reserved_identifier_schemes(rows):
    for r in rows:
        for s in spans_of(r, "PHONE"):
            assert re.search(r"555[-. ]0(1\d\d)\b", s["text"]), s
        for s in spans_of(r, "EMAIL"):
            assert re.search(r"\.(?:example|test)$|@[\w.-]*example\.(?:org|net|com)$", s["text"], re.I), s
        for s in spans_of(r, "URL"):
            assert re.search(r"(?:\.example|\.test|example\.(?:org|net|com))(?:/|$)", s["text"], re.I), s
        for s in spans_of(r, "IP"):
            assert s["text"].startswith(("192.0.2.", "198.51.100.", "203.0.113.")), s


def test_initials_only_person_is_name(rows):
    hits = 0
    for r in rows:
        for s in spans_of(r, "NAME"):
            if re.fullmatch(r"(?:[A-Za-z]\.){2,3}|[A-Za-z]{2,3}", s["text"]):
                hits += 1
    assert hits > 300
    assert any(re.search(r"(?:tech|technologist|transcribed by|logged by)[a-z ]*[:\s]+\[?[a-z.]{2,6}", r["text"], re.I) for r in rows)


def test_datetime_pattern_in_corpus(rows):
    pat = re.compile(r"(?:\d{2}[/-]\d{2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9} \d{1,2}, \d{4}|\d{1,2} [A-Za-z]{3} \d{4}) (?:at )?\d{1,2}:?\d{2}", re.I)
    found = 0
    for r in rows:
        for m in pat.finditer(r["text"]):
            date_part = re.match(r"(.+?)(?: at)? \d{1,2}:?\d{2}$", m.group(), re.I).group(1)
            assert any(s["label"] == "DATE" and s["text"].endswith(date_part) and s["end"] <= m.end() for s in r["spans"]), (r["id"], m.group())
            found += 1
    assert found > 500


def test_weekday_inside_date_span(rows):
    days = re.compile(r"^(?:mon|tue|wed|thu|fri|sat|sun)", re.I)
    assert any(days.match(s["text"]) for r in rows for s in spans_of(r, "DATE"))


# ---- fixtures ---------------------------------------------------------------
def rebuild(marked: str):
    text, spans = gen.render(marked, "normal")
    return text, [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]


def test_rebuild_initials_only_person():
    fx = FIXTURES["initials_only_person"]
    marked = f"Tech: {gen.M('NAME', 'JKL')}. Verified by: {gen.M('NAME', 'MKT')}. Read by Dr. {gen.M('NAME', 'Quennell')}."
    text, spans = rebuild(marked)
    assert text == fx["text"] and spans == fx["spans"]


def test_rebuild_report_datetimes():
    fx = FIXTURES["report_datetimes"]
    marked = (f"Accession {gen.M('ID', 'RA-7720931')}. Collected {gen.M('DATE', '03/14/2026')} 07:42; "
              f"reported {gen.M('DATE', 'Mar 15, 2026')} 13:05.")
    text, spans = rebuild(marked)
    assert text == fx["text"] and spans == fx["spans"]


def test_rebuild_report_codes_not_phi():
    fx = FIXTURES["report_codes_not_phi"]
    text, spans = rebuild(fx["text"])
    assert text == fx["text"] and spans == fx["spans"] == []


def test_generator_context_produces_fixture_conventions():
    """A context built by the generator renders the fixture conventions with the same offsets."""
    import random

    ctx = gen.Ctx(random.Random("fixture"), Counter(), True, "dicom_worklist", "report_audit_trail")
    initials = ctx.initials()
    assert re.fullmatch(r"[A-Z]{2,3}|(?:[A-Z]\.){2,3}", initials)
    d = ctx.exam_date
    marked = f"Collected {ctx.dtm(d, ' ')}"
    text, spans = gen.render(marked, "normal")
    assert [s["label"] for s in spans] == ["DATE"]
    assert re.fullmatch(r"\d{1,2}:?\d{2}( [AP]M)?", text[spans[0]["end"]:].strip())


def test_case_transform_preserves_offsets():
    marked = f"Read by Dr. {gen.M('NAME', 'Zoë Quennell')}, MD on {gen.M('DATE', 'Mar 15, 2026')} at 13:05."
    for case in ("normal", "caps", "lower"):
        text, spans = gen.render(marked, case)
        assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
        assert len(text) == len(gen.render(marked, "normal")[0])
