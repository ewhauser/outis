"""Tests for the tk22 word-surnames bank author (docs/word-surnames-bank-brief.md)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from outis.data.annotation_policy import (
    POLICY_FIXTURES,
    annotation_errors,
    identity_split,
)
from outis.data.annotation_policy_ages import AGE_FIXTURES
from outis.data.annotation_policy_care_sites import (
    CARE_SITE_FIXTURES,
    CARE_SITE_FOLLOWUP_FIXTURES,
)
from outis.data.annotation_policy_places import PLACE_FIXTURES
from outis.data.annotation_policy_targeted import (
    TARGETED_FIXTURES,
    TARGETED_FOLLOWUP_FIXTURES,
)
from outis.data.annotation_policy_transcripts import TRANSCRIPT_FIXTURES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "takeover" / "author22_word_surnames.py"
BANK = ROOT / "data" / "tk22_train_word_surnames.jsonl"


def recorded_sha256(script):
    """Hash of the generator as recorded, before the package was renamed to outis."""
    return hashlib.sha256(
        script.read_bytes().replace(b"outis", b"clinical_deid")
    ).hexdigest()


def _load():
    spec = importlib.util.spec_from_file_location("author22_word_surnames", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


author = _load()


@pytest.fixture(scope="module")
def generated():
    rows, stats = author.generate(author.SEED)
    return rows, stats


@pytest.fixture(scope="module")
def rows(generated):
    return generated[0]


def _fixture(collection, key):
    return next(item for item in collection if item["id"] == key)


def _remark(item) -> str:
    """Rebuild a fixture's marked source from its text and spans."""
    text, out, previous = item["text"], [], 0
    for span in sorted(item["spans"], key=lambda s: s["start"]):
        out.append(text[previous : span["start"]])
        out.append(author.M(span["label"], text[span["start"] : span["end"]]))
        previous = span["end"]
    out.append(text[previous:])
    return "".join(out)


# --- determinism ----------------------------------------------------------------


def test_deterministic_for_seed(rows):
    again, _ = author.generate(author.SEED)
    assert json.dumps(rows, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_other_seed_differs(rows):
    other, _ = author.generate(author.SEED + 1)
    assert [r["text"] for r in other] != [r["text"] for r in rows]


# --- structure and offsets -----------------------------------------------------------


def test_offsets_and_structure(rows):
    ids = Counter(r["id"] for r in rows)
    assert max(ids.values()) == 1
    assert len({r["text"] for r in rows}) == len(rows)
    for row in rows:
        assert annotation_errors(row["text"], row["spans"]) == [], row["id"]
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
            assert "\n" not in span["text"], "spans never cross a line or speaker label"
        assert "[[" not in row["text"] and "]]" not in row["text"]
        assert row["split"] == "train" and row["annotation_policy"] == "clinical-phi-v2"
        assert row["family"] == row["template_family"]


def test_plan_counts_layouts_and_case(rows):
    assert len(rows) == 1500
    family = Counter((r["family"], r["role"]) for r in rows)
    for name, (positives, negatives) in author.PLAN.items():
        assert family[(f"tk22_{name}", "positive")] == positives
        assert family[(f"tk22_{name}", "matched_negative")] == negatives
    layouts = Counter(r["layout"] for r in rows)
    assert len(layouts) >= 20
    assert max(layouts.values()) <= 0.08 * len(rows)
    cases = Counter(r["case"] for r in rows)
    assert abs(cases["lower"] / len(rows) - 0.40) <= 0.03
    assert abs(cases["upper"] / len(rows) - 0.20) <= 0.03
    assert abs(cases["normal"] / len(rows) - 0.40) <= 0.03
    for row in rows:
        if row["case"] == "lower":
            assert row["text"] == row["text"].lower()
        elif row["case"] == "upper":
            assert row["text"] == row["text"].upper()


def test_cohorts_follow_word_names(rows):
    for row in rows:
        expected = "unusual_name" if row["role"] == "positive" else "realistic"
        assert row["cohort"] == expected, row["id"]


# --- the brief's conventions -----------------------------------------------------------

LABEL = re.compile(r"^(?:\[(\d\d:\d\d:\d\d)\] )?([^:\n]{1,60}?): ", re.M)
TITLE_WORDS = {
    "dr",
    "dr.",
    "doctor",
    "nurse",
    "therapist",
    "coordinator",
    "np",
    "clinician",
    "interpreter",
    "daughter",
    "son",
    "spouse",
    "caregiver",
    "wife",
    "husband",
    "rn",
    "pa",
}


def _transcript_rows(rows):
    return [
        r
        for r in rows
        if r["layout"].endswith("transcript") or "transcript" in r["layout"]
    ]


def test_speaker_label_boundaries(rows):
    """T1: NAME on the name only; honorific, role words and punctuation outside."""
    checked = 0
    for row in _transcript_rows(rows):
        text = row["text"]
        for match in LABEL.finditer(text):
            start, end = match.start(2), match.end(2)
            inside = [
                s for s in row["spans"] if s["start"] < end + 1 and s["end"] > start
            ]
            for span in inside:
                assert span["label"] == "NAME", row["id"]
                assert span["end"] <= end, "the colon stays outside the span"
                value = span["text"]
                assert not re.search(r"[:()]", value), (row["id"], value)
                if (
                    len(value.split()) > 1
                ):  # a one-word name may itself be a word such as `Son`
                    assert value.split()[0].casefold().rstrip(".") not in TITLE_WORDS, (
                        row["id"],
                        value,
                    )
                assert text[span["end"]] in ":),", (row["id"], value)
                outside = (
                    text[start : span["start"]] + " " + text[span["end"] : end]
                ).casefold()
                assert set(re.findall(r"[a-z.]+", outside)) <= TITLE_WORDS | {
                    "dr.",
                    "interpreter",
                }, (row["id"], outside)
                checked += 1
    assert checked > 1000


def test_role_only_labels_not_annotated(rows):
    roles = set(author.ROLE_ONLY_LABELS) | {"porter"}
    count = 0
    for row in _transcript_rows(rows):
        for match in LABEL.finditer(row["text"]):
            label = match.group(2).strip().casefold()
            if label in roles:
                start, end = match.start(2), match.end(2)
                assert not any(
                    s["start"] < end and s["end"] > start for s in row["spans"]
                ), row["id"]
                count += 1
    assert count > 1500


def test_time_only_stamps_not_annotated(rows):
    for row in rows:
        for span in row["spans"]:
            assert not re.fullmatch(r"\d\d:\d\d(:\d\d)?", span["text"])


def test_matched_negatives_keep_the_word_unannotated(rows):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    negatives = 0
    for pair in by_pair.values():
        positive = next(r for r in pair if r["role"] == "positive")
        for neg in (r for r in pair if r["role"] == "matched_negative"):
            negatives += 1
            word = neg["word_name"].casefold()
            assert (
                neg["layout"] == positive["layout"]
                and neg["family"] == positive["family"]
            )
            assert neg["identity_group"] == positive["identity_group"]
            names = {
                t
                for s in neg["spans"]
                if s["label"] == "NAME"
                for t in author.name_tokens(s["text"])
            }
            assert word not in names, neg["id"]
            if neg["family"] != "tk22_short_names":
                assert re.search(rf"\b{re.escape(word)}\b", neg["text"].casefold()), (
                    neg["id"]
                )
    assert negatives == sum(n for _, n in author.PLAN.values())


def test_positive_rows_annotate_the_word_name(rows):
    for row in (r for r in rows if r["role"] == "positive"):
        tokens = {
            t
            for s in row["spans"]
            if s["label"] == "NAME"
            for t in author.name_tokens(s["text"])
        }
        assert row["word_name"].casefold() in tokens, row["id"]


def test_ordinary_uses_are_not_annotated(rows):
    """A phrase from the ordinary-use tables is never inside a gold span."""
    phrases = {
        p.casefold()
        for table in (author.SPEECH, author.PROSE)
        for values in table.values()
        for p in values
    }
    found = 0
    for row in rows:
        low = row["text"].casefold()
        for phrase in phrases:
            index = low.find(phrase)
            if index >= 0:
                found += 1
                assert not any(
                    s["start"] < index + len(phrase) and s["end"] > index
                    for s in row["spans"]
                ), row["id"]
    assert found > 500


def test_repetition_cap(rows):
    counts = Counter(
        t
        for r in rows
        for t in {
            t
            for s in r["spans"]
            if s["label"] == "NAME"
            for t in author.name_tokens(s["text"])
        }
    )
    assert max(counts.values()) <= 0.01 * len(rows)


def test_identity_groups_in_train_partition(rows):
    for row in rows:
        assert row["identity_group"] in row["identity_groups"]
        assert len(set(row["identity_groups"])) == len(row["identity_groups"])
        for group in row["identity_groups"]:
            assert identity_split(group) == "train"


AGE_PATTERNS = re.compile(
    r"(\b\d{1,3})(?=-year-old| y/o| yo\b|[FM]\b)|(?<=\bage: )(\d{1,3})", re.I
)


def test_every_age_annotated(rows):
    for row in rows:
        for match in AGE_PATTERNS.finditer(row["text"]):
            start = match.start(1) if match.group(1) else match.start(2)
            assert any(
                s["start"] == start and s["label"] == "AGE" for s in row["spans"]
            ), (row["id"], match.group(0))


def test_exclusion_filter_applied(generated):
    rows, stats = generated
    assert stats["final_rows_with_exclusion_hits"] == 0
    assert all(author.exclusion_hits(r) == 0 for r in rows[:200])


# --- policy fixtures rebuilt with the author's builders ------------------------------------


def test_rebuilds_transcript_speaker_label_fixture():
    fixture = _fixture(TRANSCRIPT_FIXTURES, "speaker_label_names")
    marked = "\n".join(
        (
            author.turn(
                author.speaker_label("title", "Marwick", role="dr"),
                "any palpitations?",
                "00:08:22",
            ),
            author.turn(author.speaker_label("role", role="Patient"), "no."),
            author.turn(
                author.speaker_label("role_paren", "Brannock", role="CLINICIAN"),
                "Take it with food.",
            ),
            author.turn(
                author.speaker_label("role", role="Interpreter"), "She agrees."
            ),
        )
    )
    text, spans = author.parse_marked(marked)
    assert text == fixture["text"]
    assert spans == fixture["spans"]


def test_rebuilds_unusual_name_fixtures():
    for key, marked in (
        (
            "explicit_unusual_name",
            f"Patient name: {author.N('He')}. He answered the nurse.",
        ),
        (
            "heading_and_name",
            f"Care history reviewed for the patient named {author.N('Care')}.",
        ),
        (
            "eponym_and_name",
            f"{author.N('Mira Wilson')} was assessed for Wilson disease.",
        ),
    ):
        fixture = _fixture(POLICY_FIXTURES, key)
        assert author.parse_marked(marked) == (fixture["text"], fixture["spans"])


def test_rebuilds_age_fixtures():
    M = author.M
    fixture = _fixture(AGE_FIXTURES, "ages_of_relatives")
    marked = f"Mother died at {M('AGE', author.age_words(72))}; brother, {M('AGE', '58')}, is well."
    assert author.parse_marked(marked) == (fixture["text"], fixture["spans"])


@pytest.mark.parametrize(
    "collection",
    [
        POLICY_FIXTURES,
        PLACE_FIXTURES,
        CARE_SITE_FIXTURES,
        CARE_SITE_FOLLOWUP_FIXTURES,
        TARGETED_FIXTURES,
        TARGETED_FOLLOWUP_FIXTURES,
        TRANSCRIPT_FIXTURES,
        AGE_FIXTURES,
    ],
)
def test_parser_round_trips_every_policy_fixture(collection):
    for item in collection:
        assert author.parse_marked(_remark(item)) == (item["text"], item["spans"]), (
            item["id"]
        )


def test_case_change_keeps_offsets():
    text, spans = author.parse_marked(f"dr {author.N('Do')}: do you have pain?")
    for case in ("lower", "upper", "normal"):
        changed = author.apply_case(text, case)
        assert changed[spans[0]["start"] : spans[0]["end"]].casefold() == "do"


# --- the written bank -----------------------------------------------------------------------


@pytest.mark.skipif(not BANK.exists(), reason="bank not written")
def test_written_bank_matches_generator(rows):
    payload = BANK.read_bytes()
    expected = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    ).encode("utf-8")
    assert payload == expected
    manifest = json.loads(BANK.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["rows"] == manifest["documents"] == len(rows)
    assert (
        manifest["split"] == "train"
        and manifest["annotation_policy"] == "clinical-phi-v2"
    )
    provenance = manifest["author_provenance"]
    assert provenance["seed"] == author.SEED
    assert provenance["generator_sha256"] == recorded_sha256(SCRIPT)
