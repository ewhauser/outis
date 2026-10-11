"""Tests for the tk19 spoken and spelled PHI bank generator and its committed bank."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from outis.data.annotation_policy import (
    POLICY_VERSION,
    annotation_errors,
    identity_split,
)
from outis.data.annotation_policy_spoken import SPOKEN_FIXTURES
from outis.data.generators.places import (
    excluded_sequences,
    excluded_value,
    exclusion_hashes,
)
from outis.types import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author19_spoken_phi.py"
BANK = ROOT / "data" / "tk19_train_spoken_phi.jsonl"
MANIFEST = ROOT / "data" / "tk19_train_spoken_phi.jsonl.manifest.json"

_spec = importlib.util.spec_from_file_location("author19_spoken_phi", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

WORD_NUMBERS = {w: i for i, w in enumerate(gen.ONES)}
WORD_NUMBERS.update({w: 10 * i for i, w in enumerate(gen.TENS) if w})


def recorded_sha256(script):
    """Hash of the generator as recorded, before the package was renamed to outis."""
    return hashlib.sha256(
        script.read_bytes().replace(b"outis", b"clinical_deid")
    ).hexdigest()


def parse_age(value: str) -> int:
    total = 0
    for word in re.split(r"[ -]", value.lower()):
        if word == "hundred":
            total *= 100
        else:
            total += WORD_NUMBERS[word]
    return total


def load_bank():
    return [
        json.loads(line)
        for line in BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def rows():
    if not BANK.exists():
        pytest.skip("bank not present (corpora live in the private data repository)")
    return load_bank()


@pytest.mark.requires_data("places_exclusions.sha256.json")
def test_generation_is_deterministic():
    first, stats_first = gen.generate(limit=150)
    second, stats_second = gen.generate(limit=150)
    assert first == second
    assert stats_first["rejected_draws"] == stats_second["rejected_draws"]


def test_committed_bank_is_reproduced_byte_for_byte(rows):
    regenerated, _ = gen.generate()
    payload = (
        "\n".join(json.dumps(r, ensure_ascii=False) for r in regenerated) + "\n"
    ).encode("utf-8")
    assert payload == BANK.read_bytes()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["generator_sha256"] == recorded_sha256(SCRIPT)


def test_offsets_and_row_schema(rows):
    assert len(rows) == gen.N_ROWS
    ids = set()
    for row in rows:
        ids.add(row["id"])
        assert annotation_errors(row["text"], row["spans"]) == []
        assert row["text"] == row["text"].strip()
        assert len(row["span_forms"]) == len(row["spans"])
        previous = -1
        for span, form in zip(row["spans"], row["span_forms"]):
            assert row["text"][span["start"] : span["end"]] == span["text"]
            assert span["label"] in ENTITY_TYPES
            assert span["start"] >= previous
            previous = span["end"]
            assert form in ("s", "w")
        assert row["split"] == "train" and row["annotation_policy"] == POLICY_VERSION
        assert row["family"] == row["template_family"]
        assert "[[" not in row["text"] and "]]" not in row["text"]
        assert (not row["spans"]) == row["negative"]
        assert row["negative"] == (row["identity_groups"] == [])
        for group in row["identity_groups"]:
            assert identity_split(group) == "train"
    assert len(ids) == len(rows)


def test_quotas_layouts_case_and_value_cap(rows):
    negatives = sum(r["negative"] for r in rows)
    assert abs(negatives / len(rows) - 1 / 3) < 0.01
    per_family = Counter(r["bank_family"] for r in rows)
    assert per_family == Counter(dict(gen.FAMILY_ROWS))
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) / len(rows) <= 0.08
    cases = Counter(r["case"] for r in rows)
    assert cases == Counter(
        {"normal": 300, "caps": 225, "lower": 975}
    )  # 150 STT rows are all lowercase
    value_rows = Counter()
    for row in rows:
        for value in {gen.norm_value(s["text"]) for s in row["spans"]}:
            value_rows[value] += 1
    assert max(value_rows.values()) <= 0.005 * len(rows)
    verify = per_family["verify"]
    assert verify == len(rows) // 2


def test_every_type_appears_and_spoken_and_written_both_occur(rows):
    forms = {label: Counter() for label in ENTITY_TYPES}
    for row in rows:
        for span, form in zip(row["spans"], row["span_forms"]):
            forms[span["label"]][form] += 1
    for label in ENTITY_TYPES:
        assert sum(forms[label].values()) > 0, label
    for label in (
        "NAME",
        "DATE",
        "ADDRESS",
        "PHONE",
        "EMAIL",
        "MRN",
        "SSN",
        "ACCOUNT",
        "ID",
        "URL",
        "IP",
        "AGE",
        "LOCATION",
    ):
        assert forms[label]["s"] > 0, label
    for label in (
        "NAME",
        "DATE",
        "ADDRESS",
        "PHONE",
        "EMAIL",
        "MRN",
        "SSN",
        "ACCOUNT",
        "ID",
        "URL",
        "IP",
        "AGE",
        "LOCATION",
        "FACILITY",
    ):
        assert forms[label]["w"] > 0, label


HONORIFICS = {"dr", "dr.", "mr", "mrs", "ms", "mr.", "mrs.", "ms.", "nurse", "doctor"}
CAPTIONS = re.compile(
    r"^(?:mrn|rx|ssn|social|account|member|policy|group|record|chart|number|id)(?=[ :#]|$)",
    re.I,
)
FILLER = re.compile(r"(?:^|[ ,])(?:uh|um)[ ,]*$|^(?:uh|um)[ ,]", re.I)
RELATIVE = re.compile(
    r"\b(?:ago|yesterday|today|last|next|this morning|week|weeks|month|months)\b", re.I
)


def test_conventions(rows):
    for row in rows:
        text = row["text"]
        for span in row["spans"]:
            value, label = span["text"], span["label"]
            # spans never cross a speaker label or a line break
            assert "\n" not in value
            if label != "URL":
                assert ":" not in value, (row["id"], value)
            # no honorific or caption inside a span, no punctuation edge, no filler edge
            assert value.split(" ")[0].lower() not in HONORIFICS or label != "NAME", (
                row["id"],
                value,
            )
            assert value[-1] not in ",.;?!" and value[0] not in ",.;?!", (
                row["id"],
                value,
            )
            assert not FILLER.search(value), (row["id"], value)
            if label in ("MRN", "ID", "ACCOUNT", "SSN"):
                assert not CAPTIONS.match(value), (row["id"], value)
            # a time-only stamp is never annotated
            assert not re.fullmatch(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", value)
            # relative dates and public years are not DATE spans
            if label == "DATE":
                assert not RELATIVE.search(value), (row["id"], value)
            # a ZIP said alone is LOCATION, never ADDRESS
            if label == "ADDRESS":
                tokens = re.split(r"[ ,]+", value.lower())
                assert not (
                    len(tokens) <= 7
                    and all(
                        t in gen.ONES or t in ("oh", "double", "triple") for t in tokens
                    )
                    and len(tokens) == 5
                ), (row["id"], value)
            # AGE only above 89, value only (no unit words)
            if label == "AGE":
                if value.lower() == "nineties":
                    continue
                digits = re.fullmatch(r"\d+", value)
                if digits:
                    assert int(value) > 89
                else:
                    assert parse_age(value) > 89, (row["id"], value)
        # speaker-label names: NAME span followed by ')' or ':' on a label line, honorific outside
        if row["layout"] in ("named_role", "dr_label", "chat_name"):
            names = [s for s in row["spans"] if s["label"] == "NAME"]
            assert names
            # a label name sits right before the closing bracket or colon of its speaker label
            assert any(
                text[s["end"] : s["end"] + 1] in (":", ")")
                or text[s["end"] : s["end"] + 2] == " ("
                for s in names
            )


def test_phone_ssn_domain_and_ip_schemes(rows):
    for row in rows:
        for span in row["spans"]:
            value, label = span["text"], span["label"]
            if (
                label == "PHONE"
                and re.search(r"\d{3}\D{0,3}\d{4}$", value)
                and re.search(r"[(\-.]|\+1", value)
            ):
                assert re.search(r"555\D{0,2}01\d\d", value), (row["id"], value)
            if label == "SSN" and re.fullmatch(r"\d{3}[- ]?\d{2}[- ]?\d{4}", value):
                assert 900 <= int(value[:3]) <= 999
            if label == "SSN" and not re.search(r"\d", value):
                assert value.lower().startswith("nine"), (row["id"], value)
            if label == "EMAIL" and "@" in value:
                assert re.search(
                    r"@(?:example\.(?:com|org|net)|mail\.example|webmail\.example|(?:inbox|clinic|portal)\.test)$",
                    value,
                    re.I,
                )
            if label == "EMAIL" and "@" not in value:
                assert re.search(
                    r" at (?:example dot (?:com|org|net)|mail dot example|webmail dot example|(?:inbox|clinic|portal) dot test)$",
                    value,
                    re.I,
                ), value
            if label == "IP" and re.fullmatch(r"[\d.]+", value):
                assert value.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
            if label == "URL" and "//" in value and not value.startswith("h t t p"):
                assert (
                    re.search(
                        r"\.(?:test|example\.(?:com|org|net)|example\.test)/", value
                    )
                    or ".test/" in value
                    or "example." in value
                )


def test_correction_split_and_partial_scenarios(rows):
    saw = Counter()
    for row in rows:
        spans = row["spans"]
        text = row["text"]
        if row["scenario"] == "sc_v_phone_correction" and not row["long_visit"]:
            lines = {}
            for s in spans:
                if s["label"] == "PHONE":
                    lines.setdefault(text.count("\n", 0, s["start"]), []).append(s)
            assert any(len(v) >= 2 for v in lines.values()), row["id"]
            saw["phone_correction"] += 1
        if row["scenario"] == "sc_v_dob_split" and not row["long_visit"]:
            dates = [s for s in spans if s["label"] == "DATE"]
            assert len(dates) >= 3, row["id"]
            saw["dob_split"] += 1
        if row["scenario"] == "sc_v_phone_area_apart":
            phones = [s for s in spans if s["label"] == "PHONE"]
            assert len(phones) >= 2, row["id"]
            saw["area_apart"] += 1
        if row["scenario"] == "sc_v_addr_partial" and not row["long_visit"]:
            assert all(
                s["label"] in ("ADDRESS", "LOCATION", "NAME", "FACILITY", "ID", "DATE")
                for s in spans
            )
            assert any(s["label"] == "ADDRESS" for s in spans)
            saw["addr_partial"] += 1
    assert all(count >= 5 for count in saw.values()) and len(saw) == 4


def test_spoken_fixtures_rebuild_from_generator_primitives():
    markup = gen.fixture_markup()
    assert set(markup) == {f["id"] for f in SPOKEN_FIXTURES}
    for fixture in SPOKEN_FIXTURES:
        text, spans, _ = gen.build_marked(markup[fixture["id"]])
        assert text == fixture["text"], fixture["id"]
        assert [(s["start"], s["end"], s["label"]) for s in spans] == [
            (s["start"], s["end"], s["label"]) for s in fixture["spans"]
        ], fixture["id"]
        # the marked-fixture round trip through the generator's builder is also exact
        again, spans_again, _ = gen.build_marked(
            gen.mark_fixture(fixture["text"], fixture["spans"])
        )
        assert again == fixture["text"] and len(spans_again) == len(fixture["spans"])


def test_case_transform_keeps_offsets():
    marked = (
        "Agent: Zip is "
        + gen.S("LOCATION", "five three seven")
        + ". Name "
        + gen.W("NAME", "Halvorne")
        + "."
    )
    for case in ("normal", "caps", "lower"):
        for strip in (False, True):
            text, spans, _ = gen.build_marked(marked, case, strip)
            for s in spans:
                assert text[s["start"] : s["end"]] == s["text"]
    text, spans, _ = gen.build_marked(marked, "caps", False)
    assert spans[0]["text"] == "FIVE THREE SEVEN"


def test_zero_exclusion_matches(rows):
    assert len(exclusion_hashes()) > 1000
    matched_rows = sum(excluded_sequences(r["text"]) > 0 for r in rows)
    matched_values = sum(excluded_value(s["text"]) for r in rows for s in r["spans"])
    assert matched_rows == 0 and matched_values == 0


def test_manifest_matches_bank(rows):
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["role"] == "training bank"
    assert manifest["rows"] == manifest["documents"] == len(rows)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    for key in ("label_support", "entity_support", "label_counts"):
        assert manifest[key] == {label: support[label] for label in ENTITY_TYPES}
    assert manifest["independent_final"] is False
    assert manifest["exclusion_filter"]["rows_matching_after_filter"] == 0
