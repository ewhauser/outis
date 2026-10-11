"""Tests for the tk17 laboratory and pathology training bank generator."""

from __future__ import annotations

import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from outis.data.annotation_policy import annotation_errors, identity_split
from outis.data.annotation_policy_reports import REPORT_FIXTURES
from outis.data.generators.places import excluded_sequences
from outis.types import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author17_lab_pathology.py"
CORPUS = ROOT / "data" / "tk17_train_lab_pathology.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("author17_lab_pathology", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["author17_lab_pathology"] = module
    spec.loader.exec_module(module)
    return module


gen = _load()


@pytest.fixture(scope="module")
def built():
    return gen.build_rows()


@pytest.fixture(scope="module")
def rows(built):
    return built[0]


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_deterministic(built, rows):
    again, rejects = gen.build_rows()
    assert again == rows
    assert rejects == built[1]
    if CORPUS.exists():
        stored = [
            json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
        ]
        assert stored == json.loads(json.dumps(rows, ensure_ascii=False))


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_shape_and_shares(rows):
    assert len(rows) == 1800
    fam = collections.Counter(r["family"] for r in rows)
    assert fam == {
        "lab_chem_heme_coag": 540,
        "lab_microbiology": 270,
        "lab_surgical_cytology": 360,
        "lab_poc_sendout": 270,
        "lab_portal_fax_copies": 360,
    }
    negatives = [r for r in rows if r["role"] == "matched_negative"]
    assert len(negatives) == 600
    assert all(r["spans"] == [] for r in negatives)
    assert all(r["spans"] for r in rows if r["role"] != "matched_negative")
    layouts = collections.Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) / len(rows) <= 0.08
    case = collections.Counter(r["case_style"] for r in rows)
    assert 0.55 <= case["normal"] / len(rows) <= 0.65
    assert 0.20 <= case["caps"] / len(rows) <= 0.30
    assert 0.10 <= case["lower"] / len(rows) <= 0.20
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_offsets_and_metadata(rows):
    for r in rows:
        assert annotation_errors(r["text"], r["spans"]) == []
        for s in r["spans"]:
            assert r["text"][s["start"] : s["end"]] == s["text"]
            assert s["label"] in ENTITY_TYPES
        assert r["split"] == "train" and r["annotation_policy"] == "clinical-phi-v2"
        assert r["family"] == r["template_family"]
        assert r["identity_group"] in r["identity_groups"]
        assert len(set(r["identity_groups"])) == len(r["identity_groups"])
        assert all(identity_split(g) == "train" for g in r["identity_groups"])
        assert r["phenomena"] and len(set(r["phenomena"])) == len(r["phenomena"])
        assert "[[" not in r["text"]


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_pairs_share_identity(rows):
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    paired = [v for v in by_pair.values() if len(v) == 2]
    assert len(paired) == 600
    for a, b in paired:
        assert a["identity_group"] == b["identity_group"] and a["layout"] == b["layout"]


def _spans_of(row, label):
    return [s["text"] for s in row["spans"] if s["label"] == label]


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_lab_identifiers_are_id(rows):
    seen = 0
    for r in rows:
        ids = {s["text"] for s in r["spans"] if s["label"] == "ID"}
        for m in re.finditer(r"CLIA\s*#?:?\s*(\w+)", r["text"], re.I):
            assert m[1] in ids, r["id"]
            seen += 1
        for m in re.finditer(r"\bNPI\s*:?\s*(\d+)", r["text"], re.I):
            assert m[1] in ids
        for m in re.finditer(r"State licence:\s*(\S+)", r["text"], re.I):
            assert m[1] in ids
    assert seen > 500


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_initials_only_people_are_names(rows):
    seen = 0
    for r in rows:
        names = set(_spans_of(r, "NAME"))
        for m in re.finditer(
            r"(?:Tech|Verified by|Verified|Operator|Performed by|Screened by|Drawn by|TECH):\s*(\w+)",
            r["text"],
            re.I,
        ):
            assert m[1] in names, r["id"]
            seen += 1
    assert seen > 300


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_names_exclude_honorifics_and_credentials(rows):
    cred = re.compile(r"\b(MD|DO|NP|PA-C|DNP|FACP|MBBS|PHD)\b", re.I)
    for r in rows:
        for t in _spans_of(r, "NAME"):
            assert not re.match(r"(dr|mr|ms|mx)\.?\s", t, re.I)
            assert not cred.search(t) and "," not in t


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_mrn_caption_outside_and_age_rules(rows):
    for r in rows:
        for s in r["spans"]:
            if s["label"] == "MRN":
                assert not re.match(r"(mrn|medical)", s["text"], re.I)
                assert " " not in s["text"]
            if s["label"] == "AGE":
                assert s["text"].isdigit() and int(s["text"]) > 89
        # ages at or below 89 are never annotated
        for m in re.finditer(
            r"(\d+)(?=Y [FM]\b| y/o| years / |y\b|-year-old)", r["text"], re.I
        ):
            if int(m[1]) <= 89:
                assert not any(
                    s["label"] == "AGE" and s["start"] <= m.start(1) < s["end"]
                    for s in r["spans"]
                )


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_datetime_time_outside_and_date_only_in_date(rows):
    for r in rows:
        for s in r["spans"]:
            if s["label"] == "DATE":
                assert not re.search(
                    r"\d:\d\d|\b\d{4}\b.*\b\d{4}\b", s["text"]
                ) or re.fullmatch(r"\d{4}-\d\d-\d\d", s["text"])
                assert not re.search(r"\b(am|pm)\b", s["text"], re.I)


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_codes_and_units_not_annotated(rows):
    pats = [
        r"LOINC \d+-\d",
        r"CPT \d{5}",
        r"\blot [\w-]+",
        r"Analyzer: [\w .-]+",
        r"\(ref [^)]*\)",
        r"\[[<>]?[\d.]+-?[\d.]*\]",
    ]
    for r in rows:
        for p in pats:
            for m in re.finditer(p, r["text"], re.I):
                assert not any(
                    s["start"] < m.end() and m.start() < s["end"] for s in r["spans"]
                ), (r["id"], m[0])


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_reserved_identifier_families(rows):
    for r in rows:
        for s in r["spans"]:
            t = s["text"].lower()
            if s["label"] == "PHONE":
                digits = re.sub(r"\D", "", re.split(r"\s*(?:ext|x|,)", t)[0])
                assert re.search(r"5550(1\d\d)$", digits), t
            if s["label"] == "EMAIL":
                assert re.search(r"\.(example|test|example\.(org|com|net))$", t), t
            if s["label"] == "URL":
                assert re.search(r"\.(example|test|example\.(org|com|net))/", t), t
            if s["label"] == "IP":
                assert t.startswith(("192.0.2.", "198.51.100.", "203.0.113.")), t
            if s["label"] == "SSN":
                assert t.startswith("9"), t


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_no_value_above_half_percent(rows):
    usage = collections.Counter()
    for r in rows:
        usage.update(gen.value_keys(r["spans"]))
    assert max(usage.values()) <= 9


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_zero_exclusion_matches(rows):
    assert sum(excluded_sequences(r["text"]) for r in rows) == 0


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_label_support_floor(rows):
    support = collections.Counter(s["label"] for r in rows for s in r["spans"])
    for label in (
        "NAME",
        "DATE",
        "ADDRESS",
        "PHONE",
        "EMAIL",
        "MRN",
        "ACCOUNT",
        "ID",
        "URL",
        "AGE",
        "LOCATION",
        "FACILITY",
    ):
        assert support[label] >= 80, label


def test_report_fixtures_rebuild_from_generator_builder():
    for fx in REPORT_FIXTURES:
        marked, prev = [], 0
        for s in sorted(fx["spans"], key=lambda s: s["start"]):
            marked.append(fx["text"][prev : s["start"]])
            marked.append(f"[[{s['label']}|{fx['text'][s['start'] : s['end']]}]]")
            prev = s["end"]
        marked.append(fx["text"][prev:])
        doc = gen.Doc.from_marked("".join(marked))
        text, spans = doc.render("normal")
        assert text == fx["text"], fx["id"]
        assert [(s["start"], s["end"], s["label"]) for s in spans] == [
            (s["start"], s["end"], s["label"]) for s in fx["spans"]
        ]
        for style in ("caps", "lower"):
            t2, s2 = doc.render(style)
            assert len(t2) == len(text) and [(s["start"], s["end"]) for s in s2] == [
                (s["start"], s["end"]) for s in spans
            ]


def test_generator_conventions_match_fixtures():
    """The fixture patterns, rendered by the generator's own emitters, give the fixture labels."""
    case = gen.build_case("fixture-check", "chem")
    ctx = gen.Ctx(case, True, "fixture-check")
    ctx.t("Performed at ")
    ctx.emit("lab")
    ctx.t(", CLIA ")
    ctx.emit("clia")
    ctx.t(". Specimen ")
    ctx.emit("spec")
    ctx.t(". Tech: ")
    ctx.emit("tech")
    ctx.t(". Collected ")
    ctx.emit("collected")
    text, spans = ctx.d.render("normal")
    assert [s["label"] for s in spans] == ["FACILITY", "ID", "ID", "NAME", "DATE"]
    assert not re.search(r"\d:\d\d", spans[-1]["text"]) and re.search(
        r"\d\d:\d\d|\d:\d\d [AP]M|\d{4}$", text
    )
