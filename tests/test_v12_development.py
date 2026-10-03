"""Acceptance checks for the independently authored development bank only."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import socket
from urllib.parse import urlsplit

import pytest

from clinical_deid.annotation_policy import annotation_errors, identity_split
from clinical_deid.corpus_quality import audit_corpora
from clinical_deid.types import ENTITY_TYPES
from clinical_deid.v12_development import (
    CACHE_FILE, CACHE_SHA256, LONG_PASSAGES, Case, build_rows,
    identity_pool, local_tokenizer, mark, render, token_span_positions, write_bank,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def encoding():
    return local_tokenizer(ROOT / ".cache/tiktoken")


@pytest.fixture(scope="module")
def rows(encoding):
    return build_rows(encoding)


def test_predeclared_counts_and_per_label_support(rows):
    declaration = json.loads((ROOT / "data/v12_validation.predeclaration.json").read_text())
    assert len(rows) == declaration["total_documents"] == 1280
    assert Counter(row["cohort"] for row in rows) == {"realistic": 1156, "unusual_name": 64, "long_document": 60}
    assert sum(not row["spans"] for row in rows) == declaration["minimum_phi_free_documents"] == 288
    support = Counter(span["label"] for row in rows for span in row["spans"])
    assert set(support) == set(ENTITY_TYPES)
    assert min(support.values()) >= declaration["minimum_spans_per_label"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["text"] for row in rows}) == len(rows)
    assert all(not annotation_errors(row["text"], row["spans"]) for row in rows)


def test_all_people_and_paired_negatives_share_validation_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        assert row["identity_group"] in row["identity_groups"]
        assert len(set(row["identity_groups"])) == len(row["identity_groups"])
        assert all(identity_split(identity) == "validation" for identity in row["identity_groups"])
        groups[row["pair_id"]].append(row)
    paired = [members for members in groups.values() if len(members) == 2]
    assert len(paired) == 288
    for first, second in paired:
        assert bool(first["spans"]) != bool(second["spans"])
        assert first["identity_group"] == second["identity_group"]
        assert first["identity_groups"] == second["identity_groups"]
        assert first["template_family"] == second["template_family"]


def test_unicode_offsets_names_aliases_and_ordinary_occurrences(rows):
    text, spans = render("Dr. " + mark("NAME", "Noe\u0301 van Aarden") + " reviewed " + mark("NAME", "L. D’Avel") + ".")
    assert text[spans[0]["start"]:spans[0]["end"]] == "Noe\u0301 van Aarden"
    assert spans[0]["start"] == 4
    assert not annotation_errors(text, spans)
    names = [row["text"][s["start"]:s["end"]] for row in rows for s in row["spans"] if s["label"] == "NAME"]
    assert any(not name.isascii() for name in names)
    assert any(" van " in name or " de la " in name or " al-" in name for name in names)
    assert any(name.islower() for name in names)
    assert any(name.endswith(".") and len(name) < 8 for name in names)
    unusual = next(row for row in rows if row["id"] == "v12-d-unusual-positive-00")
    spans = [s for s in unusual["spans"] if s["label"] == "NAME"]
    assert unusual["text"][spans[0]["start"]:spans[0]["end"]] == "He"
    ordinary_offset = unusual["text"].index("He listened")
    assert all(not s["start"] <= ordinary_offset < s["end"] for s in spans)
    negative = next(row for row in rows if row["id"] == "v12-d-unusual-negative-00")
    assert negative["text"].startswith("He listened") and not negative["spans"]


def test_complete_address_locality_dates_and_age_boundaries(rows):
    compact_ages = 0
    for row in rows:
        text = row["text"]
        for span in row["spans"]:
            value = text[span["start"]:span["end"]]
            if span["label"] == "ADDRESS":
                assert value[0].isdigit() and ", " in value and value[-5:].isdigit()
                assert not any(other["label"] == "LOCATION" and span["start"] <= other["start"] < span["end"] for other in row["spans"])
            elif span["label"] == "LOCATION":
                assert ", " in value and len(value.rsplit(", ", 1)[1]) == 2
            elif span["label"] == "AGE":
                assert (value.isdigit() and int(value) > 89) or value.startswith("ninety-")
                assert not value.endswith(("yo", "F", "years old"))
                compact_ages += text[span["end"]:].startswith(("F", "yo"))
            elif span["label"] == "DATE":
                assert not value.endswith((".", ";"))
        for phrase in ("A published ", "a published "):
            if phrase in text:
                start = text.index(phrase) + len(phrase)
                if text[start:start + 4].isdigit():
                    assert not any(s["start"] <= start < s["end"] for s in row["spans"])
    assert compact_ages > 50


def test_reserved_network_phone_and_invalid_ssn_provenance(rows):
    for row in rows:
        for span in row["spans"]:
            value = row["text"][span["start"]:span["end"]]
            if span["label"] in {"EMAIL", "URL"}:
                host = value.split("@", 1)[1] if span["label"] == "EMAIL" else urlsplit(value).hostname
                assert host.endswith((".example", ".test")) or host in {"example.com", "example.net", "example.org"}
            elif span["label"] == "IP":
                assert value.rsplit(".", 1)[0] in {"192.0.2", "198.51.100", "203.0.113"}
            elif span["label"] == "PHONE":
                digits = "".join(c for c in value if c.isdigit())
                assert digits[-7:-4] == "555" and 100 <= int(digits[-4:]) <= 199
            elif span["label"] == "SSN":
                assert value.startswith("000-")


def test_long_natural_and_seam_stress_are_separate_with_measured_ownership(rows, encoding):
    natural = [row for row in rows if row.get("subcohort") == "natural_narrative"]
    seams = [row for row in rows if row.get("subcohort") == "repeated_progress_seam_stress"]
    assert len(natural) == 40 and len(seams) == 20
    for row in natural + seams:
        assert row["o200k_tokens"] == len(encoding.encode(row["text"], disallowed_special=())) >= 4200
        assert row["span_token_positions"] == token_span_positions(row["text"], row["spans"], encoding)
    for row in natural:
        assert row["authored_paragraphs"] == len(LONG_PASSAGES)
        assert "repeated_progress_seam_stress" not in row["phenomena"]
        for title, body in LONG_PASSAGES:
            assert row["text"].count(body) == 1
    targeted_labels = set()
    for row in seams:
        target = row["target_ownership_join"]
        assert target >= 384 and (target - 384) % 256 == 0
        label = row["target_ownership_label"]
        targeted_labels.add(label)
        assert any(position["label"] == label and target in position["ownership_joins_crossed"] for position in row["span_token_positions"])
    assert targeted_labels == set(ENTITY_TYPES)


def test_local_tokenizer_refuses_missing_or_corrupt_cache_without_network(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Tokenizer must not open a network connection")
    monkeypatch.setattr(socket, "create_connection", denied)
    with pytest.raises(FileNotFoundError):
        local_tokenizer(tmp_path)
    (tmp_path / CACHE_FILE).write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="hash mismatch"):
        local_tokenizer(tmp_path)
    assert hashlib.sha256((ROOT / ".cache/tiktoken" / CACHE_FILE).read_bytes()).hexdigest() == CACHE_SHA256


def test_writer_refuses_to_overwrite_before_reading_or_generating(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    existing = data / "v12_validation.jsonl"
    existing.write_text("retain me\n")
    with pytest.raises(FileExistsError):
        write_bank(tmp_path, tmp_path / "missing-cache")
    assert existing.read_text() == "retain me\n"


def test_strict_audit_accepts_original_rows_with_pinned_manifest(rows, encoding, tmp_path):
    corpus = tmp_path / "v12_development.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    corpus.write_bytes(payload)
    manifest = {"split": "validation", "annotation_policy": "clinical-phi-v2", "sha256": hashlib.sha256(payload).hexdigest()}
    Path(str(corpus) + ".manifest.json").write_text(json.dumps(manifest))
    report = audit_corpora({"validation": [corpus]}, strict_policy=True,
                          token_counter=lambda text: len(encoding.encode(text, disallowed_special=())), tokenizer_name="o200k_base")
    assert report["passed"], report["errors"]
    assert report["valid_structural_rows"] == 1280
    # The generic teaching-word regex also matches reserved .example domains.
    # Keep that diagnostic visible rather than modifying prose to suppress it.
    assert all(warning["code"] == "context_cue_imbalance" and warning["cue"] == "teaching_wording"
               for warning in report["warnings"]), report["warnings"]


def test_secondary_identity_pool_is_partitioned():
    pool = identity_pool()
    case = Case(12, pool[12], pool[1012], pool[1112])
    assert len(case.identities) == len(set(case.identities)) == 3
    assert all(identity_split(group) == "validation" for group in case.identities)
