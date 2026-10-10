"""Tests for the tk20 FHIR training bank generator (scripts/takeover/author20_fhir.py)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

from outis.data.annotation_policy import (
    POLICY_VERSION,
    annotation_errors,
    identity_split,
)
from outis.data.annotation_policy_fhir import FHIR_FIXTURES
from outis.data.generators.places import excluded_sequences
from outis.types import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("author20_fhir", ROOT / "scripts" / "takeover" / "author20_fhir.py")
gen = importlib.util.module_from_spec(SPEC)
sys.modules["author20_fhir"] = gen
SPEC.loader.exec_module(gen)

DATA = ROOT / "data" / "tk20_train_fhir.jsonl"


@pytest.fixture(scope="module")
def built():
    rows, stats, used = gen.build_rows()
    return rows, stats, used


@pytest.fixture(scope="module")
def rows(built):
    return built[0]


def test_determinism(built):
    rows_a = built[0]
    rows_b, stats_b, _ = gen.build_rows()
    assert [json.dumps(r, sort_keys=True) for r in rows_a] == [json.dumps(r, sort_keys=True) for r in rows_b]
    assert dict(built[1]) == dict(stats_b)


def test_committed_bank_matches_generator(rows):
    if not DATA.exists():
        pytest.skip("bank file not present in this checkout")
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    assert DATA.read_bytes() == payload
    manifest = json.loads(Path(str(DATA) + ".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()


def test_row_schema_and_offsets(rows):
    assert len(rows) == gen.TARGET_ROWS
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    for r in rows:
        assert annotation_errors(r["text"], r["spans"]) == [], r["id"]
        assert r["split"] == "train" and r["annotation_policy"] == POLICY_VERSION
        assert r["family"] == r["template_family"]
        assert r["identity_groups"] == list(dict.fromkeys(r["identity_groups"]))
        assert r["phenomena"] and len(set(r["phenomena"])) == len(r["phenomena"])
        for s in r["spans"]:
            assert r["text"][s["start"]:s["end"]] == s["text"]
            assert s["label"] in ENTITY_TYPES
            assert s["text"] == s["text"].strip()
        assert bool(r["spans"]) == r["has_phi"]


def test_every_row_parses_in_its_format(rows):
    for r in rows:
        text = r["text"]
        if r["format"] == "json":
            json.loads(text)
        elif r["format"] == "ndjson":
            lines = text.split("\n")
            assert len(lines) >= 2
            for line in lines:
                assert isinstance(json.loads(line), dict)
        elif r["format"] == "xhtml":
            ET.fromstring(text)
        elif r["format"] == "key_value":
            for line in text.split("\n"):
                assert not line.strip() or gen.KV_LINE.match(line), (r["id"], line)
        else:
            pytest.fail(f"unknown format {r['format']}")


def json_string_ranges(text: str):
    """Content ranges of every JSON string and whether the string is an object key."""
    ranges, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            k = j + 1
            while k < n and text[k] in " \n\t":
                k += 1
            ranges.append((i + 1, j, k < n and text[k] == ":"))
            i = j + 1
        else:
            i += 1
    return ranges


def test_json_spans_are_whole_string_values_never_keys(rows):
    for r in rows:
        if r["format"] not in ("json", "ndjson"):
            continue
        ranges = json_string_ranges(r["text"])
        for s in r["spans"]:
            hits = [(a, b, key) for a, b, key in ranges if a <= s["start"] and s["end"] <= b]
            assert hits and not hits[0][2], (r["id"], s)
            assert '"' not in s["text"] and "\\" not in s["text"]


def test_xhtml_tags_stay_outside_spans(rows):
    for r in rows:
        if r["layout"].startswith("narrative") or r["format"] == "xhtml":
            tags = [(m.start(), m.end()) for m in re.finditer(r"<[^<>]*>", r["text"]) if r["format"] == "xhtml"
                    or m.group(0).startswith(("<div", "<p", "<table", "<h", "<ul", "<li", "<t", "</"))]
            for s in r["spans"]:
                assert "<" not in s["text"] and ">" not in s["text"]
                assert not any(a < s["end"] and s["start"] < b for a, b in tags), (r["id"], s)


def covered(r, start: int, end: int) -> bool:
    return any(s["start"] < end and start < s["end"] for s in r["spans"])


def test_unannotated_fhir_values_stay_unannotated(rows):
    string_value = r'"{key}"\s*:\s*"([^"\\]*)"'
    for r in rows:
        if r["format"] not in ("json", "ndjson"):
            continue
        for key in ("state", "country", "versionId", "system", "resourceType", "status", "gender", "lastUpdated", "code", "unit"):
            for m in re.finditer(string_value.format(key=key), r["text"]):
                assert not covered(r, m.start(1), m.end(1)), (r["id"], key)


def test_ids_references_urls_and_dates(rows):
    seen = Counter()
    for r in rows:
        text = r["text"]
        for s in r["spans"]:
            before = text[max(0, s["start"] - 12):s["start"]]
            if s["label"] == "URL" and r["format"] in ("json", "ndjson"):
                assert re.match(r"https?://", s["text"]) and "example." in s["text"]
                seen["url"] += 1
            if s["label"] == "ID" and before.endswith("Patient/"):
                assert "/" not in s["text"]
                seen["reference_id"] += 1
            if s["label"] == "DATE" and r["format"] in ("json", "ndjson") and r["case"] == "normal" and text[s["start"] - 1] == '"':
                assert re.fullmatch(r"\d{4}(-\d\d(-\d\d)?)?", s["text"]), s
                after = text[s["end"]:s["end"] + 4]
                assert after.startswith('"') or re.match(r"T\d\d:", after), (r["id"], after)
                seen["date"] += 1
    assert seen["url"] > 100 and seen["reference_id"] > 200 and seen["date"] > 500


def test_name_conventions(rows):
    for r in rows:
        for s in r["spans"]:
            if s["label"] != "NAME":
                continue
            assert not re.match(r"(Dr|Mr|Mrs|Ms|DR|dr)\.?\s", s["text"]), s
            assert not re.search(r",\s*(MD|DO|NP|RN|PA-C|PharmD|PhD)\b", s["text"], re.I), s
    # generational suffixes sit inside the full-name spans and stand alone as their own NAME value
    suffixes = [s["text"] for r in rows for s in r["spans"] if s["text"] in ("Jr.", "Sr.", "II", "III")]
    assert suffixes
    assert any(re.search(r"\b(Jr\.|Sr\.|II|III)$", s["text"]) for r in rows for s in r["spans"] if s["label"] == "NAME" and " " in s["text"])


def test_reserved_schemes(rows):
    for r in rows:
        for s in r["spans"]:
            t = s["text"]
            if s["label"] == "PHONE":
                digits = re.sub(r"\D", "", re.sub(r"x\d+$", "", t, flags=re.I))
                assert re.fullmatch(r"1?\d{3}5550(1\d\d)", digits), t
            elif s["label"] == "SSN" and r["case"] == "normal":
                assert 900 <= int(re.sub(r"\D", "", t)[:3]) <= 999, t
            elif s["label"] == "IP":
                assert re.match(r"(192\.0\.2|198\.51\.100|203\.0\.113)\.\d+$", t), t
            elif s["label"] == "EMAIL":
                assert re.search(r"@(?:[\w-]+\.)*(example\.(com|org|net|test))$", t, re.I), t
            elif s["label"] == "URL":
                assert re.match(r"https?://(?:[\w-]+\.)*example\.(com|org|net|test)/", t, re.I), t
            elif s["label"] == "AGE":
                assert (t.isdigit() and int(t) > 89) or t.lower().startswith("ninety"), t


def test_no_explanation_wording(rows):
    pattern = re.compile(r"\b(?:gold spans?|PHI|annotation policy|hard negative|training data|tokenizer|this is not phi)\b", re.I)
    for r in rows:
        assert not pattern.search(r["text"]), r["id"]
        assert "[[" not in r["text"] and "]]" not in r["text"], r["id"]


def test_composition_layouts_and_negatives(rows):
    n = len(rows)
    share = {fam: sum(1 for r in rows if r["composition"] == fam) / n for fam in ("single", "bundle", "narrative", "kv", "decoy")}
    target = {"single": 0.30, "bundle": 0.25, "narrative": 0.20, "kv": 0.15, "decoy": 0.10}
    for fam, t in target.items():
        assert abs(share[fam] - t) < 0.02, (fam, share[fam])
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) / n <= 0.08
    negatives = sum(1 for r in rows if not r["spans"])
    assert 0.30 <= negatives / n <= 0.36
    assert all(not r["spans"] for r in rows if r["composition"] == "decoy")
    # negatives are matched: every non-decoy negative shares its pair with a positive of the same layout
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for group in by_pair.values():
        assert len({g["layout"] for g in group}) == 1
    cased = [r for r in rows if r["layout"].startswith(("narrative", "kv"))]
    share_cased = sum(1 for r in cased if r["case"] != "normal") / len(cased)
    assert 0.10 <= share_cased <= 0.20


def test_label_support_and_value_cap(rows, built):
    support = Counter(s["label"] for r in rows for s in r["spans"])
    for label in ENTITY_TYPES:
        assert support[label] >= 80, (label, support[label])
    per_value = Counter()
    for r in rows:
        for v in {s["text"] for s in r["spans"]}:
            per_value[v] += 1
    assert max(per_value.values()) <= gen.FINAL_CAP
    assert gen.FINAL_CAP / len(rows) <= 0.005 + 1e-9 + 0.5 / len(rows)


def test_identities_are_in_the_training_partition(rows):
    for r in rows:
        for g in r["identity_groups"]:
            assert identity_split(g) == "train", r["id"]
        if r["spans"]:
            assert r["identity_groups"] and r["identity_group"] == r["identity_groups"][0]
        else:
            assert r["identity_group"]


def test_zero_exclusion_matches(rows):
    for r in rows:
        assert excluded_sequences(r["text"]) == 0, r["id"]
        assert excluded_sequences(gen.DEPUNCT.sub(" ", r["text"])) == 0, r["id"]


def test_no_fixture_value_is_reused(rows):
    fixture_values = gen.FIXTURE_TEXTS
    assert fixture_values
    for r in rows:
        assert not any(s["text"] in fixture_values for s in r["spans"]), r["id"]


def test_metadata_avoids_banned_words_and_keys(rows):
    for r in rows:
        blob = json.dumps({k: v for k, v in r.items() if k not in ("text", "spans")}).lower()
        for word in gen.BANNED_METADATA:
            assert word not in blob, (r["id"], word)
    if DATA.exists():
        manifest = json.loads(Path(str(DATA) + ".manifest.json").read_text())
        assert manifest["role"] == "training bank"
        assert manifest["split"] == "train" and manifest["independent_final"] is False
        for key in ("provenance", "source", "data"):
            assert key not in manifest
        blob = json.dumps(manifest).lower()
        for word in gen.BANNED_METADATA:
            assert word not in blob, word


# ---- rebuild every FHIR_FIXTURES gold span with this generator's writer ----------------

def _rebuild_fixtures():
    P, Rich, Style = gen.P, gen.Rich, gen.Style
    flat = Style("flat")
    built = {}
    built["fhir_ids_and_systems"] = {
        "fullUrl": P("URL", "https://fhir.example.test/Patient/pt-77"),
        "resource": {"resourceType": "Patient", "id": P("ID", "pt-77"), "meta": {"versionId": "3"},
                     "identifier": [{"system": "urn:oid:2.16.840.1.113883.19.5", "value": P("MRN", "MRN-7720931")}]},
    }
    built["fhir_reference"] = {
        "subject": {"reference": Rich(["Patient/", P("ID", "pt-77")])},
        "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7"}]},
    }
    built["fhir_datetime"] = {"effectiveDateTime": Rich([P("DATE", "2026-03-14"), "T07:42:00-05:00"]),
                              "birthDate": P("DATE", "1941-03-03")}
    built["fhir_name_address_parts"] = {
        "name": [{"family": P("NAME", "Brannock"), "given": [P("NAME", "Odalys"), P("NAME", "Q")]}],
        "address": [{"line": [P("ADDRESS", "412 Wenlow Lane"), P("ADDRESS", "Apt 2B")], "city": P("LOCATION", "Corriston"),
                     "state": "WI", "postalCode": P("LOCATION", "53728"), "country": "US"}],
        "gender": "female",
    }
    built["fhir_narrative"] = {"text": {"status": "generated", "div": Rich([
        '<div xmlns="http://www.w3.org/1999/xhtml"><p>', P("NAME", "Odalys Brannock"), ", born ", P("DATE", "3 March 1941"),
        "</p></div>"])}}
    return {key: gen.render_json(value, flat) for key, value in built.items()}


def test_rebuilds_every_fhir_fixture_gold_span():
    rebuilt = _rebuild_fixtures()
    assert {fx["id"] for fx in FHIR_FIXTURES} == set(rebuilt), "a fixture was added or renamed; extend _rebuild_fixtures"
    for fx in FHIR_FIXTURES:
        out = rebuilt[fx["id"]]
        assert out.text == fx["text"], fx["id"]
        assert [(s["start"], s["end"], s["label"]) for s in out.spans] == [(s["start"], s["end"], s["label"]) for s in fx["spans"]], fx["id"]

