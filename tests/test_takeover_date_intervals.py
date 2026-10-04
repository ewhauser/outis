"""Synthetic-only contract tests for training DATE interval derivatives."""

from collections import Counter
from copy import deepcopy
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/takeover/normalize_date_intervals.py"
SPEC = importlib.util.spec_from_file_location("takeover_date_intervals", SCRIPT)
intervals = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = intervals
SPEC.loader.exec_module(intervals)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sidecar(path):
    return path.with_suffix(path.suffix + ".manifest.json")


def date_row(connector=" to ", *, prefix="Treatment from ", dates=None, row_id="interval"):
    dates = dates or ("March 3, 2025", "March 5, 2025")
    text = prefix + connector.join(dates) + "."
    spans = []
    position = len(prefix)
    for value in dates:
        spans.append({"start": position, "end": position + len(value), "label": "DATE"})
        position += len(value) + len(connector)
    return {"id": row_id, "text": text, "spans": spans, "split": "train"}


def bank(tmp_path, name="synthetic_training", *, rows=None, manifest_updates=None, payload=None):
    source = tmp_path / f"{name}.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else [date_row()]
    if payload is None:
        payload = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
    source.write_bytes(payload)
    counts = Counter(span["label"] for row in rows for span in row["spans"])
    support = {label: counts[label] for label in intervals.LABELS}
    manifest = {
        "split": "train", "corpus": name, "sha256": digest(payload),
        "rows": len(rows), "documents": len(rows), "count": len(rows),
        "label_counts": dict(support), "label_support": dict(support),
        "entity_support": dict(support), "annotation_policy": "synthetic-fixture-only",
        "source_provenance": {"generator": "unit-test", "version": 7},
        "cohort_counts": {"synthetic": len(rows)},
        "notes": "Preserve this metadata verbatim in the derivative.",
    }
    manifest.update(manifest_updates or {})
    sidecar(source).write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest, {"path": str(source), "sha256": digest(payload), "rows": len(rows)}


def assert_no_new_files(tmp_path, before):
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def paths_in(tmp_path):
    return {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}


def forbid_reads(monkeypatch, paths):
    """Trip on file reads through either builtins.open or pathlib's io.open."""
    forbidden = {str(Path(path).absolute()) for path in paths}
    originals = [(builtins, builtins.open), (io, io.open)]
    for module, original in originals:
        def guarded(file, mode="r", *args, _original=original, **kwargs):
            if not isinstance(file, int):
                requested = str(Path(file).absolute())
                if requested in forbidden and ("r" in mode or "+" in mode):
                    raise AssertionError(f"Corpus or sidecar read before batch preflight: {requested}")
            return _original(file, mode, *args, **kwargs)
        monkeypatch.setattr(module, "open", guarded)


@pytest.mark.parametrize("connector", [" - ", "–", " — ", " − ", " to ", " THROUGH ", " thru ", " until "])
def test_literal_intervals_merge_with_exact_offsets_and_audit(connector):
    row = date_row(connector)
    original = deepcopy(row)
    normalized, changes, exclusions = intervals.normalize_row(row)
    expected = {"start": row["spans"][0]["start"], "end": row["spans"][1]["end"], "label": "DATE"}
    assert row == original
    assert normalized == {**row, "spans": [expected]}
    assert len(changes) == 1
    assert changes[0]["row_id"] == row["id"]
    assert [(span["start"], span["end"], span["label"]) for span in changes[0]["old_spans"]] == [
        (span["start"], span["end"], span["label"]) for span in row["spans"]
    ]
    assert {key: changes[0]["new_span"][key] for key in expected} == expected
    assert changes[0]["connector"].casefold() == connector.strip().casefold()
    assert not exclusions


def test_between_range_preserves_mixed_format_weekday_and_unicode():
    row = date_row(" and ", prefix="Zoë was admitted between ",
                   dates=("MONDAY, SEPTEMBER 28, 2026", "10/03/2026"))
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert len(changes) == 1
    assert not exclusions
    assert normalized["text"] == row["text"]
    span = normalized["spans"][0]
    assert normalized["text"][span["start"]:span["end"]] == "MONDAY, SEPTEMBER 28, 2026 and 10/03/2026"


@pytest.mark.parametrize("dates", [
    ("2025-03-03", "2025-03-05"), ("03-MAR-2025", "05-MAR-2025"),
])
def test_calendar_format_hyphens_are_not_mistaken_for_existing_intervals(dates):
    row = date_row(dates=dates)
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert len(changes) == 1
    assert len(normalized["spans"]) == 1
    assert not exclusions


@pytest.mark.parametrize(("connector", "prefix"), [
    (" and ", "Dates of service: "), (" or ", "Appointment on "),
    (", ", "Seen on "), ("; ", "Seen on "),
    ("\n- ", "Timeline: "), ("\r\n to ", "Admission from "),
    ("\n\n- ", "Date of birth: "), ("\tthrough\n", "Treatment from "),
    (" to ", "Rescheduled from "), (" to ", "Appointment moved from "),
    (" to ", "Date corrected from "), (" to ", "Appointment changed from "),
    (" to ", "Comparison of "), (" through ", "Compared with "),
    (" followed by ", "Visits: "),
])
def test_lists_alternatives_newlines_revisions_and_comparisons_are_unchanged(connector, prefix):
    row = date_row(connector, prefix=prefix)
    normalized, changes, _ = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []


@pytest.mark.parametrize("line_break", ["\n", "\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\u2029"])
@pytest.mark.parametrize("placement", ["gap", "endpoint"])
def test_line_breaks_exclude_intervals_even_inside_annotated_endpoints(line_break, placement):
    row = (
        date_row(f" to{line_break} ") if placement == "gap"
        else date_row(dates=(f"March{line_break}3, 2025", "March 5, 2025"))
    )
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []
    assert any(item["reason"] == "newline_boundary" for item in exclusions)


def test_does_not_bridge_intervening_annotation():
    row = date_row(" to ")
    start = row["spans"][0]["end"] + 1
    row["spans"].insert(1, {"start": start, "end": start + 2, "label": "ID"})
    normalized, changes, _ = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []


def test_date_chain_is_wholly_excluded_and_unrelated_interval_can_merge():
    chain = date_row(" to ", dates=("March 3, 2025", "March 5, 2025", "March 7, 2025"))
    row = deepcopy(chain)
    other = date_row(" until ", prefix=" New admission from ")
    offset = len(row["text"])
    row["text"] += other["text"]
    row["spans"] += [{**span, "start": span["start"] + offset, "end": span["end"] + offset}
                     for span in other["spans"]]
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert normalized["spans"][:3] == chain["spans"]
    assert len(normalized["spans"]) == 4
    assert len(changes) == 1
    assert changes[0]["new_span"]["start"] == row["spans"][3]["start"]
    assert exclusions


@pytest.mark.parametrize("last_connector", [" and ", " or "])
def test_between_with_a_third_date_is_not_treated_as_a_two_endpoint_interval(last_connector):
    row = date_row(" and ", prefix="Admissions between ")
    third = "March 7, 2025"
    start = len(row["text"]) - 1 + len(last_connector)
    row["text"] = row["text"][:-1] + last_connector + third + "."
    row["spans"].append({"start": start, "end": start + len(third), "label": "DATE"})
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []
    assert exclusions[0]["reason"] == "chained_interval"
    assert exclusions[0]["old_spans"][0]["start"] == row["spans"][0]["start"]


@pytest.mark.parametrize("dates", [
    ("March 3–5, 2025", "March 7, 2025"),
    ("March 3, 2025", "March 5, 2025 to March 7, 2025"),
    ("03/03/2025 through 03/05/2025", "03/07/2025"),
])
def test_already_compound_date_endpoints_are_excluded(dates):
    row = date_row(dates=dates)
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []
    assert exclusions


def test_compatible_date_metadata_and_all_non_date_content_are_preserved_idempotently():
    row = date_row(prefix="Zoë: treatment from ")
    row["spans"].insert(0, {"start": 0, "end": 3, "label": "NAME", "source": "synthetic"})
    metadata = {"confidence": 1.0, "annotation": {"reviewed": True, "tags": ["manual"]}}
    for span in row["spans"][1:]:
        span.update(deepcopy(metadata))
    row["metadata"] = {"patient": "synthetic", "nested": [1, {"keep": "exact"}]}
    row["cohort"] = "unusual"
    original = deepcopy(row)
    normalized, changes, _ = intervals.normalize_row(row)
    assert row == original
    assert {key: value for key, value in normalized.items() if key != "spans"} == {
        key: value for key, value in row.items() if key != "spans"
    }
    assert normalized["spans"][0] == row["spans"][0]
    assert all(normalized["spans"][1][key] == value for key, value in metadata.items())
    assert len(changes) == 1
    again, repeated, _ = intervals.normalize_row(normalized)
    assert again == normalized
    assert repeated == []


@pytest.mark.parametrize("second_metadata", [{"confidence": 0.9}, {}, {"confidence": 1.0, "extra": True}])
def test_incompatible_span_metadata_is_not_silently_lost(second_metadata):
    row = date_row()
    row["spans"][0]["confidence"] = 1.0
    row["spans"][1].update(second_metadata)
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert normalized == row
    assert changes == []
    assert exclusions


@pytest.mark.parametrize("mismatch_index", [0, 2])
@pytest.mark.parametrize("connector", [" to ", " - "])
def test_partial_chain_with_metadata_mismatch_remains_idempotent(mismatch_index, connector):
    row = date_row(connector, dates=("March 3, 2025", "March 5, 2025", "March 7, 2025"))
    row["spans"][mismatch_index]["reviewer"] = "separate-review"
    original = deepcopy(row)
    normalized, changes, exclusions = intervals.normalize_row(row)
    assert row == original
    assert len(changes) == 1
    assert len(normalized["spans"]) == 2
    assert row["spans"][mismatch_index] in normalized["spans"]
    assert any(item["reason"] == "incompatible_DATE_metadata" for item in exclusions)
    repeated, second_changes, _ = intervals.normalize_row(normalized)
    assert repeated == normalized
    assert second_changes == []


@pytest.mark.parametrize("defect", ["negative", "empty", "past_end", "noninteger", "overlap"])
def test_invalid_offsets_or_overlaps_fail_without_mutating_input(defect):
    row = date_row()
    if defect == "negative":
        row["spans"][0]["start"] = -1
    elif defect == "empty":
        row["spans"][0]["end"] = row["spans"][0]["start"]
    elif defect == "past_end":
        row["spans"][1]["end"] = len(row["text"]) + 1
    elif defect == "noninteger":
        row["spans"][0]["start"] = 1.5
    else:
        row["spans"][1]["start"] = row["spans"][0]["end"] - 1
    original = deepcopy(row)
    with pytest.raises(ValueError):
        intervals.normalize_row(row)
    assert row == original


def test_plan_is_dry_run_and_synthetic_apply_preserves_bytes_text_and_complete_lineage(tmp_path):
    unchanged = {"id": "untouched", "text": "Zoë arrived.", "spans": [{"start": 0, "end": 3, "label": "NAME"}]}
    changed = date_row()
    changed["metadata"] = {"nested": ["keep", 3]}
    last = {"id": "last", "text": "No fever.", "spans": []}
    first_line = json.dumps(unchanged, ensure_ascii=False, indent=None, separators=(" , ", " : ")).encode() + b"\r\n"
    changed_line = json.dumps(changed, ensure_ascii=False).encode() + b"\n"
    last_line = json.dumps(last, separators=(",", ":")).encode()
    payload = first_line + b"\r\n" + changed_line + b"\n" + last_line
    source, original_manifest, entry = bank(tmp_path, rows=[unchanged, changed, last], payload=payload)
    original_sidecar = sidecar(source).read_bytes()
    before = paths_in(tmp_path)
    plan, prepared = intervals.plan_sources([entry])
    assert_no_new_files(tmp_path, before)
    assert len(plan["corpora"]) == 1
    corpus = plan["corpora"][0]
    assert corpus["source_path"] == str(source)
    assert corpus["source_sha256"] == digest(payload)
    assert corpus["rows"] == 3
    assert len(corpus["transformations"]) == 1
    output = Path(corpus["output_path"])
    assert output != source
    assert "dateinterval1" in output.name
    report = tmp_path / "synthetic_plan.json"
    intervals.write_plan(report, plan)
    assert json.loads(report.read_text()) == plan
    intervals.apply_prepared(prepared)
    result = output.read_bytes()
    assert result.startswith(first_line + b"\r\n")
    assert result.endswith(b"\n" + last_line)
    result_rows = [json.loads(line) for line in result.splitlines() if line.strip()]
    assert result_rows[0] == unchanged
    assert result_rows[-1] == last
    assert result_rows[1] == intervals.normalize_row(changed)[0]
    assert source.read_bytes() == payload
    assert sidecar(source).read_bytes() == original_sidecar
    manifest = json.loads(sidecar(output).read_text())
    assert manifest["sha256"] == digest(result)
    assert all(manifest[key] == 3 for key in ("rows", "documents", "count"))
    for key in ("label_counts", "label_support", "entity_support"):
        assert manifest[key]["DATE"] == 1
        assert manifest[key]["NAME"] == 1
        assert sum(manifest[key].values()) == 2
    for key in ("split", "annotation_policy", "source_provenance", "cohort_counts", "notes"):
        assert manifest[key] == original_manifest[key]
    lineage = manifest["date_interval_normalization"]
    assert lineage["version"] == 1
    assert "user-adopted training convention" in lineage["authority"].casefold()
    assert lineage["source"] == {"path": str(source), "sha256": digest(payload),
                                 "manifest_path": str(sidecar(source)), "manifest_sha256": digest(original_sidecar)}
    assert lineage["source_manifest"] == original_manifest
    assert lineage["script_sha256"] == digest(SCRIPT.read_bytes())
    assert lineage["transformation_count"] == 1
    assert lineage["changed_rows"] == 1


@pytest.mark.parametrize("role", [None, "dev", "development", "final", "proxy", "independent", "retired", "test"])
def test_manifest_role_batch_preflight_precedes_all_corpus_reads(tmp_path, monkeypatch, role):
    first, _, valid = bank(tmp_path, "first_training")
    second, manifest, invalid = bank(tmp_path, "innocent_alias")
    if role is None:
        manifest.pop("split")
    else:
        manifest["split"] = role
    sidecar(second).write_text(json.dumps(manifest))
    before = paths_in(tmp_path)
    forbid_reads(monkeypatch, [first, second])
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("role", [None, "unknown"])
@pytest.mark.parametrize("declaration", ["manifest", "provenance"])
def test_explicit_null_or_unknown_training_declarations_precede_corpus_reads(
    tmp_path, monkeypatch, role, declaration,
):
    first, _, valid = bank(tmp_path, "first_training")
    second, manifest, invalid = bank(tmp_path, "second_training")
    if declaration == "manifest":
        manifest["split"] = role
    else:
        manifest["source_provenance"]["split"] = role
    sidecar(second).write_text(json.dumps(manifest))
    before = paths_in(tmp_path)
    forbid_reads(monkeypatch, [first, second])
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("role", [None, "unknown"])
def test_explicit_null_or_unknown_row_split_is_rejected_without_mutation(role):
    row = date_row()
    row["split"] = role
    original = deepcopy(row)
    with pytest.raises(ValueError):
        intervals.normalize_row(row)
    assert row == original


def test_absent_row_split_is_allowed_under_pinned_training_manifest(tmp_path):
    row = date_row()
    row.pop("split")
    _, _, entry = bank(tmp_path, rows=[row])
    plan, _ = intervals.plan_sources([entry])
    assert plan["corpora"][0]["transformation_count"] == 1


@pytest.mark.parametrize("role", ["train", "training"])
def test_explicit_training_split_is_accepted(tmp_path, role):
    _, _, entry = bank(tmp_path, manifest_updates={"split": role})
    plan, _ = intervals.plan_sources([entry])
    assert plan["corpora"][0]["rows"] == 1


@pytest.mark.parametrize("row_pin", [None, True, False, 0, -1, 1.0, "1"])
def test_invalid_explicit_row_pin_fails_batch_preflight_before_reads(tmp_path, monkeypatch, row_pin):
    first, _, valid = bank(tmp_path, "first_training")
    second, _, invalid = bank(tmp_path, "second_training")
    invalid["rows"] = row_pin
    before = paths_in(tmp_path)
    forbid_reads(monkeypatch, [first, second, sidecar(first), sidecar(second)])
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("name", ["dev", "final", "proxy", "independent", "retired", "development", "payload.sealed"])
def test_forbidden_later_path_is_rejected_before_any_source_or_sidecar_read(tmp_path, monkeypatch, name):
    first, _, valid = bank(tmp_path, "first_training")
    second, _, invalid = bank(tmp_path, name)
    before = paths_in(tmp_path)
    forbid_reads(monkeypatch, [first, second, sidecar(first), sidecar(second)])
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("alias_kind", [
    "source_symlink", "sidecar_symlink", "copied_manifest_role", "copied_manifest_path",
    "copied_manifest_list",
])
def test_training_alias_cannot_bypass_protected_identity(tmp_path, monkeypatch, alias_kind):
    first, _, valid = bank(tmp_path, "first_training")
    second, manifest, invalid = bank(tmp_path, "friendly_training")
    if alias_kind == "source_symlink":
        target = tmp_path / "independent.jsonl"
        second.rename(target)
        second.symlink_to(target)
        blocked = [first, second, target, sidecar(first), sidecar(second)]
    elif alias_kind == "sidecar_symlink":
        target = tmp_path / "proxy.manifest.json"
        sidecar(second).rename(target)
        sidecar(second).symlink_to(target)
        blocked = [first, second, sidecar(first), sidecar(second), target]
    else:
        if alias_kind == "copied_manifest_role":
            manifest["source_provenance"] = {"split": "dev"}
        elif alias_kind == "copied_manifest_path":
            manifest["source_provenance"] = {"source_path": "/synthetic/development/source.jsonl"}
        else:
            manifest["source_lineage"] = [{"split": "train"}, {"split": "independent"}]
        sidecar(second).write_text(json.dumps(manifest))
        blocked = [first, second]
    before = paths_in(tmp_path)
    forbid_reads(monkeypatch, blocked)
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("defect", ["source_hash", "manifest_hash", "rows", "manifest_rows", "label_counts", "duplicate_ids", "row_role"])
def test_late_validation_failure_produces_no_outputs(tmp_path, defect):
    _, _, valid = bank(tmp_path, "first_training")
    second, manifest, invalid = bank(tmp_path, "second_training")
    if defect == "source_hash":
        invalid["sha256"] = "0" * 64
    elif defect == "manifest_hash":
        manifest["sha256"] = "0" * 64
    elif defect == "rows":
        invalid["rows"] += 1
    elif defect == "manifest_rows":
        manifest["rows"] += 1
    elif defect == "label_counts":
        manifest["label_counts"]["DATE"] += 1
    else:
        rows = [date_row()]
        if defect == "duplicate_ids":
            rows.append(deepcopy(rows[0]))
        else:
            rows[0]["split"] = "dev"
        second, manifest, invalid = bank(tmp_path, "second_training", rows=rows)
    sidecar(second).write_text(json.dumps(manifest))
    before = paths_in(tmp_path)
    with pytest.raises(ValueError):
        intervals.plan_sources([valid, invalid])
    assert_no_new_files(tmp_path, before)


def test_duplicate_resolved_source_preflight_permits_no_file_reads(tmp_path, monkeypatch):
    source, _, entry = bank(tmp_path)
    alias = tmp_path / "alias_training.jsonl"
    alias.symlink_to(source)
    alias_entry = {**entry, "path": str(alias)}
    sidecar(alias).symlink_to(sidecar(source))
    forbid_reads(monkeypatch, [source, alias, sidecar(source), sidecar(alias)])
    with pytest.raises(ValueError):
        intervals.plan_sources([entry, alias_entry])


@pytest.mark.parametrize("slot", ["corpus", "manifest"])
def test_preexisting_later_output_blocks_all_derivatives(tmp_path, slot):
    _, _, first = bank(tmp_path, "first_training")
    _, _, second = bank(tmp_path, "second_training")
    plan, prepared = intervals.plan_sources([first, second])
    outputs = [Path(item["output_path"]) for item in plan["corpora"]]
    existing = outputs[-1] if slot == "corpus" else sidecar(outputs[-1])
    existing.write_bytes(b"immutable prior output")
    before = paths_in(tmp_path)
    with pytest.raises(FileExistsError):
        intervals.apply_prepared(prepared)
    assert_no_new_files(tmp_path, before)
    assert existing.read_bytes() == b"immutable prior output"


@pytest.mark.parametrize("changed_slot", ["corpus", "manifest"])
def test_changed_later_source_after_planning_blocks_every_output(tmp_path, changed_slot):
    _, _, first = bank(tmp_path, "first_training")
    second, _, second_entry = bank(tmp_path, "second_training")
    _, prepared = intervals.plan_sources([first, second_entry])
    changed = second if changed_slot == "corpus" else sidecar(second)
    changed.write_bytes(changed.read_bytes() + b"\n")
    before = paths_in(tmp_path)
    with pytest.raises(ValueError):
        intervals.apply_prepared(prepared)
    assert_no_new_files(tmp_path, before)


def test_plan_report_is_exclusive_and_never_overwrites_existing_file(tmp_path):
    _, _, entry = bank(tmp_path)
    plan, _ = intervals.plan_sources([entry])
    report = tmp_path / "synthetic_plan.json"
    report.write_bytes(b"previous plan")
    with pytest.raises(FileExistsError):
        intervals.write_plan(report, plan)
    assert report.read_bytes() == b"previous plan"
