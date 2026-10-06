"""Tests for the tk16 phone and telehealth transcript bank generator and its committed bank."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from clinical_deid.data.annotation_policy import (
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from clinical_deid.data.annotation_policy_transcripts import TRANSCRIPT_FIXTURES
from clinical_deid.data.generators.places import excluded_sequences, excluded_value

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author16_transcripts.py"
BANK_PATH = ROOT / "data" / "tk16_train_transcripts.jsonl"

_spec = importlib.util.spec_from_file_location("author16_transcripts", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
sys.modules["author16_transcripts"] = gen
_spec.loader.exec_module(gen)

HONORIFIC = re.compile(r"^(dr|mr|mrs|ms|mx|nurse|rn|md|do|np|pa)\.?$", re.I)
ROLE_LABEL = re.compile(
    r"^(?:[\[(]?[0-9T:\- ]+[\])]?\s+)?(?:patient|nurse|clinician|provider|doctor|physician|interpreter|caregiver|family member|neighbor|scheduler|staff|"
    r"case manager|care coordinator|social worker|agency rep|intake coordinator|representative|rn|triage nurse|on-call clinician|speaker \d+)\s*(?::|$)",
    re.I,
)
PHONE_RE = re.compile(r"555[-. ]01\d\d")
EXT_RE = re.compile(r"555[-. ]01\d\d(?:,\s*| )(?:extension|ext\.?|x)\s?\d+|555[-. ]01\d\d x\d+", re.I)
VERSION_RE = re.compile(r"\bversion \d+\.\d+\.\d+")
SPELLED_AGE = re.compile(r"^(?:ninety|one hundred)(?:-[a-z]+| and [a-z-]+)?$")


@pytest.fixture(scope="module")
def small_bank():
    rows, stats = gen.build(120, 16, cap=7)
    return rows, stats


@pytest.fixture(scope="module")
def committed_bank():
    if not BANK_PATH.exists():
        pytest.skip("committed bank not present")
    return [json.loads(line) for line in BANK_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generator_is_deterministic():
    a, _ = gen.build(40, 16, cap=7)
    b, _ = gen.build(40, 16, cap=7)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
    c, _ = gen.build(40, 17, cap=7)
    assert [r["text"] for r in a] != [r["text"] for r in c]


def test_committed_bank_matches_its_manifest(committed_bank):
    manifest = json.loads(Path(str(BANK_PATH) + ".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()
    assert manifest["rows"] == len(committed_bank) == 1500
    assert manifest["independent_final"] is False
    assert manifest["annotation_policy"] == "clinical-phi-v2"


def test_rows_offsets_and_metadata(small_bank):
    rows, _ = small_bank
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert annotation_errors(r["text"], r["spans"]) == []
        for sp in r["spans"]:
            assert r["text"][sp["start"]:sp["end"]] == sp["text"]
        assert r["split"] == "train" and r["annotation_policy"] == "clinical-phi-v2"
        assert r["family"] == r["template_family"]
        assert r["cohort"] in ("realistic", "long_document")
        assert r["phenomena"] and len(set(r["phenomena"])) == len(r["phenomena"])
        assert "[[" not in r["text"] and "]]" not in r["text"]
        if not r["phi_free"]:
            assert r["identity_group"] in r["identity_groups"]
            assert len(set(r["identity_groups"])) == len(r["identity_groups"])
            for h in r["identity_groups"]:
                assert identity_split(h) == "train"
        else:
            assert r["spans"] == [] and r["identity_groups"] == []
            assert r["identity_group"]


def test_phi_free_rows_have_a_positive_twin(small_bank):
    rows, _ = small_bank
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    twins = [v for v in by_pair.values() if any(x["phi_free"] for x in v)]
    assert twins
    for v in twins:
        assert any(not x["phi_free"] for x in v)
        assert len({x["family"] for x in v}) == 1 and len({x["case"] for x in v}) == 1


def _check_conventions(rows):
    for r in rows:
        text, spans = r["text"], r["spans"]
        for sp in spans:
            val = sp["text"]
            if sp["label"] == "NAME":
                assert not any(HONORIFIC.match(tok) for tok in val.split()), val
                assert not re.search(r"[:()\[\],]", val), val
            if sp["label"] == "DATE":
                assert not re.search(r"\b\d{1,2}:\d{2}\b", val), val
            if sp["label"] == "PHONE":
                assert PHONE_RE.search(val), val
            if sp["label"] == "EMAIL":
                assert re.search(r"@(?:mail\.example|inbox\.example\.com|post\.example\.org|webmail\.example\.net|home\.example|mail\.test)$", val, re.I), val
            if sp["label"] == "URL":
                assert re.match(r"https?://[a-z.]+\.(?:test|example\.org|example\.com|example\.net)/", val, re.I), val
            if sp["label"] == "IP":
                assert val.startswith(("192.0.2.", "198.51.100.", "203.0.113.")), val
            if sp["label"] == "AGE":
                assert val.lower() == "nineties" or (val.isdigit() and int(val) >= 90) or SPELLED_AGE.match(val.lower()), val
        # role-only labels, timestamps and versions stay unannotated
        pos = 0
        for line in text.split("\n"):
            m = ROLE_LABEL.match(line)
            if m:
                label_end = pos + m.end()
                bad = [sp for sp in spans if sp["start"] < label_end and sp["end"] > pos and sp["label"] != "DATE"]
                assert not bad, (r["id"], line[:80])
            pos += len(line) + 1
        for m in re.finditer(r"\[\d{1,2}:\d{2}(?::\d{2})?\]|\(\d{2}:\d{2}:\d{2}\)", text):
            assert not any(sp["start"] < m.end() and sp["end"] > m.start() for sp in spans), m.group(0)
        for m in VERSION_RE.finditer(text):
            assert not any(sp["start"] < m.end() and sp["end"] > m.start() for sp in spans), m.group(0)
        for m in EXT_RE.finditer(text):
            assert any(sp["label"] == "PHONE" and sp["start"] <= m.start() and sp["end"] >= m.end() for sp in spans), m.group(0)
        for m in re.finditer(r"https?://[A-Za-z0-9./_-]+", text):
            end = m.start() + len(m.group(0).rstrip("."))
            assert any(sp["label"] == "URL" and sp["start"] <= m.start() and sp["end"] >= end for sp in spans), m.group(0)
        # dated timestamps: the date is DATE, the clock time stays outside
        for m in re.finditer(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\]", text):
            date_start = m.start(1)
            assert any(sp["label"] == "DATE" and sp["start"] == date_start and sp["end"] == m.end(1) for sp in spans), m.group(0)
        # a person's name never appears unannotated: all-caps and lowercase text follow the authored role
        names = {tok.casefold() for sp in spans if sp["label"] == "NAME" for tok in sp["text"].split() if len(tok) >= 4}
        for tok in names:
            for m in re.finditer(r"(?<![\w'-])" + re.escape(tok) + r"(?![\w-])", text.casefold()):
                assert any(sp["start"] <= m.start() and m.end() <= sp["end"] for sp in spans), (r["id"], tok)


def test_conventions_on_fresh_build(small_bank):
    rows, _ = small_bank
    _check_conventions(rows)


def test_conventions_on_committed_bank(committed_bank):
    _check_conventions(committed_bank)


def test_rebuilds_every_transcript_fixture():
    rebuilt = gen.rebuild_transcript_fixtures()
    assert set(rebuilt) == {fx["id"] for fx in TRANSCRIPT_FIXTURES}
    for fx in TRANSCRIPT_FIXTURES:
        text, spans = rebuilt[fx["id"]]
        assert text == fx["text"], fx["id"]
        assert spans == [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in fx["spans"]], fx["id"]


def test_render_marked_follows_case_and_offsets():
    marked = "[00:08:22] dr [[NAME|Marwick]]: take [[PHONE|(612) 555-0193, extension 2276]] on [[DATE|Mon, 18 Dec 2023]] at 10:43."
    for case in ("normal", "lower", "upper"):
        text, spans = gen.render_marked(marked, case)
        assert len(spans) == 3
        for sp in spans:
            value = text[sp["start"]:sp["end"]]
            assert value == value.strip() and value
        assert text[spans[0]["start"]:spans[0]["end"]] == {"normal": "Marwick", "lower": "marwick", "upper": "MARWICK"}[case]


def test_speech_to_text_rows_have_no_punctuation_but_keep_values(committed_bank):
    stt = [r for r in committed_bank if r["family"].endswith("stt")]
    assert len(stt) == 225
    for r in stt:
        assert not re.search(r"[?!;]", r["text"])
        for sp in r["spans"]:
            if sp["label"] in ("ADDRESS", "DATE", "LOCATION", "PHONE"):
                assert "," not in sp["text"]


def test_exclusion_filter_leaves_no_matches(small_bank, committed_bank):
    for rows in (small_bank[0], committed_bank):
        for r in rows:
            assert excluded_sequences(r["text"]) == 0, r["id"]
            for sp in r["spans"]:
                assert not excluded_value(sp["text"]), r["id"]
    manifest = json.loads(Path(str(BANK_PATH) + ".manifest.json").read_text())
    assert manifest["exclusion_filter"]["rows_matching_after_filter"] == 0
    assert manifest["exclusion_filter"]["counts_only"] is True


def test_no_value_above_half_a_percent_of_rows(committed_bank):
    cap = int(0.005 * len(committed_bank))
    assert cap == 7
    values, tokens = Counter(), Counter()
    for r in committed_bank:
        for key in {(sp["label"], sp["text"].casefold()) for sp in r["spans"]}:
            values[key] += 1
        for tok in {t.casefold() for sp in r["spans"] if sp["label"] == "NAME" for t in sp["text"].split()}:
            tokens[tok] += 1
    assert max(values.values()) <= cap
    assert max(tokens.values()) <= cap


def test_composition_of_committed_bank(committed_bank):
    n = len(committed_bank)
    fam = Counter(r["family"].replace("tk16_transcript_", "") for r in committed_bank)
    assert fam == {"telehealth": 525, "phone": 375, "interpreter": 150, "coordination": 225, "stt": 225}
    free = sum(1 for r in committed_bank if r["phi_free"])
    assert 0.19 <= free / n <= 0.21
    combos = Counter((r["label_format"], r["case"]) for r in committed_bank if not r["phi_free"])
    for fmt in gen.FMTS:
        for case in gen.CASES:
            assert combos[(fmt, case)] >= 30, (fmt, case)
    assert {"none", "intro", "switch"} <= {r["label_change"] for r in committed_bank}
    long_rows = sum(1 for r in committed_bank if r["tokens"] > 512)
    assert long_rows / n > 0.9
    labels = Counter(sp["label"] for r in committed_bank for sp in r["spans"])
    for lab in ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "ACCOUNT", "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY"):
        assert labels[lab] > 0, lab


def test_partition_of_identities_in_committed_bank(committed_bank):
    for r in committed_bank:
        for h in r["identity_groups"]:
            assert identity_split(h) == "train"
    assert canonical_identity_group("A B") == canonical_identity_group("a b")
