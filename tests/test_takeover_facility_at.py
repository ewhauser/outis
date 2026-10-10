"""Synthetic-only contract tests for training FACILITY 'at <place>' derivatives."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from outis.data.annotation_policy import POLICY_FIXTURES

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/takeover/normalize_facility_at.py"
)
SPEC = importlib.util.spec_from_file_location("takeover_facility_at", SCRIPT)
facat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = facat
SPEC.loader.exec_module(facat)


def row(*parts, row_id="r1"):
    """Build a row from (text, label|None) parts."""
    text, spans = "", []
    for value, label in parts:
        if label:
            spans.append(
                {"start": len(text), "end": len(text) + len(value), "label": label}
            )
        text += value
    return {"id": row_id, "text": text, "spans": spans, "split": "train"}


def labeled(r):
    return [(s["label"], r["text"][s["start"] : s["end"]]) for s in r["spans"]]


def test_bare_locality_after_at_is_merged_into_facility():
    source = row(
        ("Seen at ", None),
        ("Lakeshore Family Medicine", "FACILITY"),
        (" at ", None),
        ("Ellery Falls", "LOCATION"),
        (" today.", None),
    )
    out, changes, exclusions = facat.normalize_row(source)
    assert labeled(out) == [("FACILITY", "Lakeshore Family Medicine at Ellery Falls")]
    assert len(changes) == 1 and not exclusions
    assert out["text"] == source["text"]


@pytest.mark.parametrize(
    "parts",
    [
        (
            ("Brackenwheel Health Center", "FACILITY"),
            (" at ", None),
            ("Brackenford, NM", "LOCATION"),
        ),
        (
            ("Brackenwheel Health Center", "FACILITY"),
            (" at ", None),
            ("Brackenford", "LOCATION"),
            (", NM", None),
        ),
        (
            ("Brackenwheel Health Center", "FACILITY"),
            (" at ", None),
            ("Brackenford", "LOCATION"),
            (", ", None),
            ("New Mexico", "LOCATION"),
        ),
        (
            ("Brackenwheel Health Center", "FACILITY"),
            (" at ", None),
            ("Brackenford", "LOCATION"),
            (", 87001", None),
        ),
    ],
)
def test_qualified_geography_stays_separate(parts):
    source = row(*parts)
    out, changes, exclusions = facat.normalize_row(source)
    assert out["spans"] == source["spans"] and not changes
    assert [e["reason"] for e in exclusions] == ["qualified_geography"]


def test_named_unit_and_address_are_left_unchanged():
    source = row(
        ("Larkspur Ward", "FACILITY"),
        (" at ", None),
        ("Mossmere Health Hub", "FACILITY"),
        (". Then ", None),
        ("Larkfield Hospice", "FACILITY"),
        (" at ", None),
        ("446 Tamarack Street", "ADDRESS"),
    )
    out, changes, exclusions = facat.normalize_row(source)
    assert out["spans"] == source["spans"] and not changes
    assert sorted(e["reason"] for e in exclusions) == [
        "address_after_at",
        "named_unit_at_facility",
    ]


def test_uppercase_at_is_handled_like_lowercase():
    source = row(
        ("ELLERY FALLS NEUROLOGY CLINIC", "FACILITY"),
        (" AT ", None),
        ("THIMBLE CREEK", "LOCATION"),
        ("\n", None),
        ("WILLOW INTENSIVE CARE UNIT", "FACILITY"),
        (" AT ", None),
        ("NORTHWIND PSYCHIATRIC INSTITUTE", "FACILITY"),
    )
    out, changes, exclusions = facat.normalize_row(source)
    assert labeled(out)[0] == (
        "FACILITY",
        "ELLERY FALLS NEUROLOGY CLINIC AT THIMBLE CREEK",
    )
    assert len(changes) == 1
    assert [e["reason"] for e in exclusions] == ["named_unit_at_facility"]


def test_line_break_or_other_gap_is_not_merged():
    for gap in ("\nat ", " at\n", " in ", " at the ", ", "):
        source = row(
            ("Lakeshore Family Medicine", "FACILITY"),
            (gap, None),
            ("Ellery Falls", "LOCATION"),
        )
        out, changes, _ = facat.normalize_row(source)
        assert out["spans"] == source["spans"] and not changes, gap


def test_span_text_field_is_kept_consistent():
    source = row(
        ("Copper Mesa Urgent Care", "FACILITY"), (" at ", None), ("Ivel", "LOCATION")
    )
    for s in source["spans"]:
        s["text"] = source["text"][s["start"] : s["end"]]
    out, _, _ = facat.normalize_row(source)
    assert out["spans"] == [
        {
            "start": 0,
            "end": len(source["text"]),
            "label": "FACILITY",
            "text": source["text"],
        }
    ]


def test_policy_fixture_is_a_fixed_point():
    fixture = {r["id"]: r for r in POLICY_FIXTURES}["facility_at_continuation"]
    out, changes, exclusions = facat.normalize_row(dict(fixture, split="train"))
    assert out["spans"] == fixture["spans"] and not changes
    assert sorted(e["reason"] for e in exclusions) == [
        "named_unit_at_facility",
        "qualified_geography",
    ]


def test_refuses_non_training_rows_and_paths(tmp_path):
    with pytest.raises(ValueError):
        facat.normalize_row(dict(row(("X Clinic", "FACILITY")), split="validation"))
    with pytest.raises(ValueError):
        facat.check_path(tmp_path / "v12_validation.jsonl")


def _corpus(tmp_path, rows):
    path = tmp_path / "tk_train_admin.jsonl"
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(payload)
    support = {}
    for r in rows:
        for s in r["spans"]:
            support[s["label"]] = support.get(s["label"], 0) + 1
    manifest = {
        "split": "train",
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_counts": {label: support.get(label, 0) for label in facat.LABELS},
    }
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    return path, hashlib.sha256(payload).hexdigest()


def test_apply_writes_new_files_only_and_preserves_unchanged_lines(tmp_path):
    changed = row(
        ("Ellery Falls Family Clinic", "FACILITY"),
        (" at ", None),
        ("South Ashgrove", "LOCATION"),
        row_id="a",
    )
    unchanged = row(("Seen in ", None), ("Varden", "LOCATION"), row_id="b")
    path, sha = _corpus(tmp_path, [changed, unchanged])
    original = path.read_bytes()
    out_dir = tmp_path / "out"
    plan, prepared = facat.plan_sources(
        [{"path": str(path), "sha256": sha}], output_dir=out_dir
    )
    assert plan["summary"]["transformations"] == 1 and not out_dir.exists()
    facat.apply_prepared(prepared)
    assert path.read_bytes() == original
    output = out_dir / "tk_train_admin__facat1.jsonl"
    lines = output.read_bytes().splitlines(keepends=True)
    assert lines[1] == original.splitlines(keepends=True)[1]
    assert labeled(json.loads(lines[0])) == [
        ("FACILITY", "Ellery Falls Family Clinic at South Ashgrove")
    ]
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        manifest["label_counts"]["LOCATION"] == 1
        and manifest["label_counts"]["FACILITY"] == 1
    )
    assert manifest["facility_at_normalization"]["source"]["sha256"] == sha
    with pytest.raises(FileExistsError):
        facat.plan_sources([{"path": str(path), "sha256": sha}], output_dir=out_dir)


def test_rejects_hash_mismatch(tmp_path):
    path, _ = _corpus(tmp_path, [row(("X Clinic", "FACILITY"))])
    with pytest.raises(ValueError):
        facat.plan_sources(
            [{"path": str(path), "sha256": "0" * 64}], output_dir=tmp_path / "out"
        )
