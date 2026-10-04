"""Synthetic-only regression checks for frozen mixture preparation boundaries."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import stat

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/takeover/prepare_training_mixture.py"
SPEC = importlib.util.spec_from_file_location("takeover_training_mixture", SCRIPT)
mixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixture)


def row(rid, text, *, label=None, identity=None):
    value = {"id": rid, "text": text, "split": "train", "cohort": "realistic",
             "annotation_policy": "synthetic-test-policy", "spans": []}
    if label:
        value["spans"] = [{"start": 0, "end": len(text), "label": label, "text": text}]
    if identity:
        value.update(identity_group=identity, identity_groups=[identity])
    return value


def rewrite(path, payload, *, frozen=True):
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444 if frozen else 0o644)


def bank(root, name, rows, *, training=True, endings=None):
    path = root / "data" / f"{name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    raw = [json.dumps(value, ensure_ascii=False, separators=(", ", ": ")).encode() +
           (endings[index] if endings else b"\n") for index, value in enumerate(rows)]
    rewrite(path, b"".join(raw), frozen=training)
    labels = Counter(span["label"] for value in rows for span in value["spans"])
    support = {label: labels[label] for label in mixture.LABELS}
    manifest = {"corpus": str(path), "sha256": mixture.digest(path.read_bytes()), "split": "train" if training else "validation",
                "annotation_policy": "synthetic-test-policy", "rows": len(rows), "documents": len(rows), "count": len(rows),
                "label_support": support, "entity_support": support, "label_counts": support,
                "cohort_counts": dict(Counter(value.get("cohort", "unspecified") for value in rows)),
                "independent_final": False, "source_provenance": {"original_rows": 99999}}
    sidecar = Path(str(path) + ".manifest.json")
    rewrite(sidecar, mixture.json_bytes(manifest), frozen=training)
    return {"path": str(path), "sha256": manifest["sha256"], "manifest_path": str(sidecar),
            "manifest_sha256": mixture.digest(sidecar.read_bytes())}, raw


def audit_for(training, drops=None):
    entries = []
    for pin in training:
        rows = [json.loads(raw) for raw in Path(pin["path"]).read_bytes().splitlines() if raw.strip()]
        removed = (drops or {}).get(Path(pin["path"]).stem, [])
        metrics = {kind: {"distinct_shared_digests": 0, "sealed_rows_matching": 0, "source_rows_matching": 0}
                   for kind in mixture.AUDIT_KINDS}
        if removed:
            metrics["exact_text"] = {key: len(removed) for key in metrics["exact_text"]}
        entries.append({"path": pin["path"], "sha256": pin["sha256"], "rows": len(rows), "blocked": bool(removed),
                        "training_ids_to_remove": removed, "counts": metrics})
    return {"started_utc": "2026-10-04T00:00:00+00:00", "finished_utc": "2026-10-04T00:01:00+00:00",
            "sealed_pin": mixture.SEALED_PIN, "disclosure": mixture.DISCLOSURE,
            "any_blocked": any(entry["blocked"] for entry in entries), "corpora": entries}


def freeze_plan(root, plan, audit=None):
    if audit is not None:
        audit_path = root / "aggregate_audit.json"
        rewrite(audit_path, mixture.json_bytes(audit))
        plan["sanctioned_audit"] = {"path": str(audit_path), "sha256": mixture.digest(audit_path.read_bytes())}
    plan_path = root / "input_plan.json"
    rewrite(plan_path, mixture.json_bytes(plan), frozen=False)
    return {"path": str(plan_path), "sha256": mixture.digest(plan_path.read_bytes())}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(mixture, "ROOT", root)
    training, raw = bank(root, "clinical_train", [row("keep", "Zoë Ávila", label="NAME"), row("other", "Resting.")])
    development = []
    for index, name in enumerate(mixture.DEVELOPMENT_BANKS):
        pin, _ = bank(root, name, [row(f"dev-{index}", f"Development note {index}.")], training=False)
        development.append(pin)
    plan = {"schema_version": 1, "training": [training], "development": development,
            "report_path": str(root / "preparation.json")}
    audit = audit_for(plan["training"])
    reference = freeze_plan(root, plan, audit)
    return root, plan, audit, reference, raw


def prepare(fixture):
    _, plan, _, reference, _ = fixture
    return mixture.prepare(plan, plan_reference=reference)


def slots(plan):
    outputs = [Path(plan["report_path"])]
    for pin in plan["training"]:
        path = Path(pin["path"])
        output = path.with_name(path.stem + "__mix3clean.jsonl")
        outputs.extend([output, Path(str(output) + ".manifest.json")])
    return outputs


def repin_manifest(pin, **changes):
    path = Path(pin["manifest_path"])
    value = json.loads(path.read_bytes())
    value.update(changes)
    rewrite(path, mixture.json_bytes(value))
    pin["manifest_sha256"] = mixture.digest(path.read_bytes())


def test_filters_union_preserves_exact_bytes_and_recomputes_every_aggregate(fixture):
    root, plan, _, _, _ = fixture
    shared = "a" * 64
    rows = [row("keep-unicode", "Zoë Ávila", label="NAME"),
            row("sealed-only", "2041", label="DATE"),
            row("text-and-sealed", "Development note 0."),
            row("identity-only", "Different wording", identity=shared),
            row("keep-empty", "Resting.")]
    rows[-1]["cohort"] = "unusual"
    pin, original_lines = bank(root, "clinical_train", rows, endings=[b"\r\n", b"\n", b"\r\n", b"\n", b""])
    dev_rows = [row("dev-1", "Separate note.", identity=shared)]
    plan["development"][1], _ = bank(root, mixture.DEVELOPMENT_BANKS[1], dev_rows, training=False)
    plan["training"] = [pin]
    audit = audit_for([pin], {"clinical_train": ["sealed-only", "text-and-sealed"]})
    reference = freeze_plan(root, plan, audit)
    originals = {Path(pin[key]): Path(pin[key]).read_bytes() for key in ("path", "manifest_path")}
    report, prepared, snapshots, report_path = mixture.prepare(plan, plan_reference=reference)
    assert not any(path.exists() for path in slots(plan))
    final = mixture.apply_prepared(report, prepared, snapshots, report_path)
    corpus = report["corpora"][0]
    output = Path(corpus["output"]["path"])
    assert output.read_bytes() == original_lines[0] + original_lines[-1]
    manifest = json.loads(Path(corpus["output"]["manifest_path"]).read_bytes())
    assert all(manifest[key] == 2 for key in ("rows", "documents", "count"))
    assert manifest["spans"] == manifest["entity_count"] == manifest["phi_free_rows"] == 1
    assert manifest["cohort_counts"] == {"realistic": 1, "unusual": 1}
    assert all(manifest[key] == {label: int(label == "NAME") for label in mixture.LABELS}
               for key in ("label_counts", "label_support", "entity_support"))
    assert manifest["source_provenance"] == {"original_rows": 99999}
    assert manifest["training_mixture_preparation"]["source"] == pin
    assert manifest["sha256"] == mixture.digest(output.read_bytes())
    assert final["status"] == "frozen_complete"
    assert final["train_vs_development_overlap"] == {"exact_text_rows": 0, "canonical_identity_rows": 0}
    assert final["summary"]["removed_rows"] == 3
    reasons = {item["id"]: item["reasons"] for item in corpus["removals"]}
    assert reasons == {"sealed-only": ["sanctioned_v9r3_training_id"],
                       "text-and-sealed": ["sanctioned_v9r3_training_id", "development_exact_text"],
                       "identity-only": ["development_canonical_identity"]}
    for path in slots(plan):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert all(path.read_bytes() == payload for path, payload in originals.items())
    assert json.loads(report_path.read_bytes()) == final


def test_identity_name_unicode_normalization_and_group_digest_cross_match(fixture):
    root, plan, _, _, _ = fixture
    name = "Zoë Ávila"
    value = row("drop", "Name absent from this prose")
    value["full_identity"] = "Zoe\u0308 A\u0301VILA"
    keep = row("keep", "Untouched.")
    plan["training"][0], _ = bank(root, "clinical_train", [value, keep])
    hashed = mixture.digest(name.casefold().encode())
    plan["development"][0], _ = bank(root, mixture.DEVELOPMENT_BANKS[0], [row("dev", "Elsewhere", identity=hashed)], training=False)
    reference = freeze_plan(root, plan, audit_for(plan["training"]))
    report, *_ = mixture.prepare(plan, plan_reference=reference)
    assert report["corpora"][0]["removals"][0]["reasons"] == ["development_canonical_identity"]


@pytest.mark.parametrize("defect", ["source_hash", "manifest_hash", "audit_hash", "mutable_source", "mutable_manifest", "mutable_audit",
                                     "rows", "cohorts", "labels", "manifest_corpus_hash", "unknown_removal", "duplicate_removal",
                                     "audit_rows", "audit_source_hash", "audit_pin", "audit_status", "audit_counts", "audit_coverage"])
def test_preflight_errors_never_create_any_output(fixture, defect):
    root, plan, audit, _, _ = fixture
    pin = plan["training"][0]
    if defect in {"source_hash", "manifest_hash"}:
        path = Path(pin["path" if defect == "source_hash" else "manifest_path"])
        rewrite(path, path.read_bytes() + b"\n")
    elif defect in {"mutable_source", "mutable_manifest"}:
        Path(pin["path" if defect == "mutable_source" else "manifest_path"]).chmod(0o644)
    elif defect == "rows":
        repin_manifest(pin, rows=999)
    elif defect == "cohorts":
        repin_manifest(pin, cohort_counts={"unusual": 2})
    elif defect == "labels":
        repin_manifest(pin, label_counts={label: 0 for label in mixture.LABELS})
    elif defect == "manifest_corpus_hash":
        repin_manifest(pin, sha256="0" * 64)
    elif defect == "unknown_removal":
        audit = audit_for(plan["training"], {"clinical_train": ["unknown"]})
    elif defect == "duplicate_removal":
        audit = audit_for(plan["training"], {"clinical_train": ["keep", "keep"]})
    elif defect == "audit_rows":
        audit["corpora"][0]["rows"] = 3
    elif defect == "audit_source_hash":
        audit["corpora"][0]["sha256"] = "0" * 64
    elif defect == "audit_pin":
        audit["sealed_pin"] = "0" * 64
    elif defect == "audit_status":
        audit["any_blocked"] = True
    elif defect == "audit_counts":
        audit["corpora"][0]["counts"]["exact_text"]["source_rows_matching"] = 1
    elif defect == "audit_coverage":
        audit["corpora"] = []
    reference = freeze_plan(root, plan, audit)
    if defect in {"audit_hash", "mutable_audit"}:
        path = Path(plan["sanctioned_audit"]["path"])
        if defect == "audit_hash":
            rewrite(path, path.read_bytes() + b"\n")
        else:
            path.chmod(0o644)
    with pytest.raises(ValueError):
        mixture.prepare(plan, plan_reference=reference)
    assert not any(path.exists() for path in slots(plan))


@pytest.mark.parametrize("slot_index", [0, 1, 2])
def test_existing_outputs_are_preserved_even_before_reading_inputs(fixture, monkeypatch, slot_index):
    _, plan, _, _, _ = fixture
    target = slots(plan)[slot_index]
    target.write_bytes(b"existing artifact")
    reads = []
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (reads.append(path), original(path))[1])
    with pytest.raises(ValueError, match="overwrite"):
        prepare(fixture)
    assert reads == []
    assert original(target) == b"existing artifact"
    assert all(not path.exists() for path in slots(plan) if path != target)


@pytest.mark.parametrize("role", ["training", "development"])
@pytest.mark.parametrize("defect", ["flag", "role", "lineage_path", "nested_flag"])
def test_copied_alias_provenance_rejected_before_any_corpus_read(fixture, monkeypatch, role, defect):
    root, plan, audit, _, _ = fixture
    pin = plan[role][-1]
    changes = {"flag": {"independent_final": True}, "role": {"bank_role": "heldout_proxy"},
               "lineage_path": {"source_lineage": [{"path": "data/blind_qualification_v9_revision3.jsonl"}]},
               "nested_flag": {"envelope": {"original": {"is_proxy": True}}}}[defect]
    repin_manifest(pin, **changes)
    reference = freeze_plan(root, plan, audit)
    reads = []
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (reads.append(path), original(path))[1])
    with pytest.raises(ValueError):
        mixture.prepare(plan, plan_reference=reference)
    assert not any(path.suffix == ".jsonl" for path in reads)
    assert not any(path.exists() for path in slots(plan))


@pytest.mark.parametrize("defect", ["protected_source", "source_symlink", "manifest_symlink", "parent_symlink", "hardlink", "development_renamed",
                                     "missing_development", "duplicate_development", "heldout_as_training"])
def test_paths_and_complete_development_allowlist_are_checked_before_reads(fixture, monkeypatch, defect):
    root, plan, audit, _, _ = fixture
    pin = plan["training"][0]
    if defect == "protected_source":
        pin["path"] = str(root / "data" / "blind_qualification_v9_revision3.jsonl")
    elif defect == "source_symlink":
        alias = root / "data" / "alias_train.jsonl"
        alias.symlink_to(pin["path"])
        pin["path"] = str(alias)
        pin["manifest_path"] = str(alias) + ".manifest.json"
    elif defect == "manifest_symlink":
        manifest = Path(pin["manifest_path"])
        moved = manifest.with_name("manifest-original.json")
        manifest.rename(moved)
        manifest.symlink_to(moved)
    elif defect == "parent_symlink":
        alias = root / "alias"
        alias.symlink_to(root / "data", target_is_directory=True)
        pin["path"] = str(alias / Path(pin["path"]).name)
        pin["manifest_path"] = pin["path"] + ".manifest.json"
    elif defect == "hardlink":
        (root / "linked-copy.jsonl").hardlink_to(pin["path"])
    elif defect == "development_renamed":
        other, _ = bank(root, "ordinary_validation", [row("dev", "Different")], training=False)
        plan["development"][0] = other
    elif defect == "missing_development":
        plan["development"].pop()
    elif defect == "duplicate_development":
        plan["development"].append(deepcopy(plan["development"][0]))
    elif defect == "heldout_as_training":
        plan["training"] = [deepcopy(plan["development"][-1])]
    reference = freeze_plan(root, plan, audit)
    reads = []
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (reads.append(path), original(path))[1])
    with pytest.raises(ValueError):
        mixture.prepare(plan, plan_reference=reference)
    assert reads == []


@pytest.mark.parametrize("defect", ["bool_offset", "offset_range", "unknown_label", "overlap", "span_text", "duplicate_id",
                                     "policy", "identity_type", "identity_consistency", "row_role"])
def test_invalid_rows_are_rejected_even_when_sanctioned_for_removal(fixture, defect):
    root, plan, _, _, _ = fixture
    rows = [row("keep", "Zoë Ávila", label="NAME"), row("bad", "2041", label="DATE")]
    bad = rows[-1]
    if defect == "bool_offset":
        bad["spans"][0]["start"] = False
    elif defect == "offset_range":
        bad["spans"][0]["end"] = 100
    elif defect == "unknown_label":
        bad["spans"][0]["label"] = "OTHER"
    elif defect == "overlap":
        bad["spans"].append(deepcopy(bad["spans"][0]))
    elif defect == "span_text":
        bad["spans"][0]["text"] = "mismatched"
    elif defect == "duplicate_id":
        bad["id"] = "keep"
    elif defect == "policy":
        bad["annotation_policy"] = "different-policy"
    elif defect == "identity_type":
        bad["identity_groups"] = [123]
    elif defect == "identity_consistency":
        bad.update(identity_group="a" * 64, identity_groups=["b" * 64])
    elif defect == "row_role":
        bad["split"] = "validation"
    plan["training"][0], _ = bank(root, "clinical_train", rows)
    reference = freeze_plan(root, plan, audit_for(plan["training"], {"clinical_train": ["keep" if defect == "duplicate_id" else "bad"]}))
    with pytest.raises(ValueError):
        mixture.prepare(plan, plan_reference=reference)
    assert not any(path.exists() for path in slots(plan))


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_text_reports_same_annotations_but_rejects_conflicts_across_sources(fixture, conflict):
    root, plan, _, _, _ = fixture
    other, _ = bank(root, "second_train", [row("different-id", "Zoë Ávila", label="FACILITY" if conflict else "NAME")])
    plan["training"].append(other)
    reference = freeze_plan(root, plan, audit_for(plan["training"]))
    if conflict:
        with pytest.raises(ValueError, match="contradictory annotations"):
            mixture.prepare(plan, plan_reference=reference)
        assert not any(path.exists() for path in slots(plan))
    else:
        report, prepared, snapshots, path = mixture.prepare(plan, plan_reference=reference)
        assert report["summary"]["after"]["rows"] == 3
        assert len(report["same_annotation_duplicate_text_groups"]) == 1
        mixture.apply_prepared(report, prepared, snapshots, path)
        assert sum(json.loads(Path(c["output"]["manifest_path"]).read_bytes())["rows"] for c in report["corpora"]) == 3


def test_cross_source_duplicate_ids_fail_even_when_text_differs(fixture):
    root, plan, _, _, _ = fixture
    other, _ = bank(root, "second_train", [row("keep", "Distinct note")])
    plan["training"].append(other)
    reference = freeze_plan(root, plan, audit_for(plan["training"]))
    with pytest.raises(ValueError, match="duplicate row ID"):
        mixture.prepare(plan, plan_reference=reference)


@pytest.mark.parametrize("changed", ["corpus", "manifest", "audit", "plan", "output"])
def test_recheck_detects_changes_between_preparation_and_application(fixture, changed):
    _, plan, _, reference, _ = fixture
    report, prepared, snapshots, report_path = prepare(fixture)
    target = {"corpus": plan["training"][0]["path"], "manifest": plan["training"][0]["manifest_path"],
              "audit": plan["sanctioned_audit"]["path"], "plan": reference["path"], "output": str(prepared[0][0])}[changed]
    target = Path(target)
    rewrite(target, b"changed bytes", frozen=changed != "plan")
    with pytest.raises(ValueError):
        mixture.apply_prepared(report, prepared, snapshots, report_path)
    assert all(not path.exists() for path in slots(plan) if path != target)
    assert target.read_bytes() == b"changed bytes"


def test_exclusive_reservation_race_preserves_foreign_file_and_rolls_back_owned(fixture, monkeypatch):
    _, plan, _, _, _ = fixture
    report, prepared, snapshots, report_path = prepare(fixture)
    original = mixture.os.link
    foreign = prepared[1][0]

    def racing_link(source, path, **kwargs):
        if path == foreign:
            foreign.write_bytes(b"concurrent artifact")
        return original(source, path, **kwargs)

    monkeypatch.setattr(mixture.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        mixture.apply_prepared(report, prepared, snapshots, report_path)
    assert foreign.read_bytes() == b"concurrent artifact"
    assert all(not path.exists() for path in slots(plan) if path != foreign)


def test_aggregate_support_threshold_is_explicit_not_per_bank(fixture):
    root, plan, audit, _, _ = fixture
    report, *_ = prepare(fixture)
    assert report["summary"]["min_label_support"] == 0
    plan["min_label_support"] = 1
    reference = freeze_plan(root, plan, audit)
    with pytest.raises(ValueError, match="minimum label support"):
        mixture.prepare(plan, plan_reference=reference)


def test_cli_defaults_to_readonly_and_requires_plan_hash(fixture, capsys):
    _, plan, _, reference, _ = fixture
    args = ["--plan", reference["path"], "--plan-sha256", reference["sha256"]]
    mixture.main(args)
    assert json.loads(capsys.readouterr().out)["status"] == "validated_not_written"
    assert not any(path.exists() for path in slots(plan))
    with pytest.raises(SystemExit) as failure:
        mixture.main(["--plan", reference["path"], "--plan-sha256", "0" * 64, "--apply"])
    assert failure.value.code == 2
    assert not any(path.exists() for path in slots(plan))
    mixture.main([*args, "--apply"])
    assert json.loads(capsys.readouterr().out)["status"] == "frozen_complete"
    assert all(path.exists() for path in slots(plan))


def test_approved_development_links_work_but_changed_target_is_rejected(fixture, monkeypatch):
    root, plan, audit, _, _ = fixture
    main = root.parent / "main"
    (main / "data").mkdir(parents=True)
    monkeypatch.setattr(mixture, "MAIN_ROOT", main)
    pin = plan["development"][0]
    path = Path(pin["path"])
    target = main / "data" / path.name
    path.rename(target)
    path.symlink_to(target)
    reference = freeze_plan(root, plan, audit)
    report, *_ = mixture.prepare(plan, plan_reference=reference)
    assert report["summary"]["after"]["rows"] == 2
    path.unlink()
    unrelated = main / "data" / "other_validation.jsonl"
    unrelated.write_bytes(target.read_bytes())
    path.symlink_to(unrelated)
    with pytest.raises(ValueError, match="aliases"):
        mixture.prepare(plan, plan_reference=reference)


def test_legacy_metadata_is_preserved_with_explicit_diagnostics(fixture):
    root, plan, _, _, _ = fixture
    keep = row("keep", "Resting.")
    keep.update(identity_group="phi_free:train:1", identity_groups=[], full_identities=[], family="clinical", template_family="template")
    single = row("single", "Mononym", label="NAME", identity="d" * 64)
    single["full_identity"] = "Mononym"
    plan["training"][0], raw = bank(root, "clinical_train", [keep, single])
    repin_manifest(plan["training"][0], development_overlap_removal={"selection": [{"path": pin["path"]} for pin in plan["development"]]})
    reference = freeze_plan(root, plan, audit_for(plan["training"]))
    report, prepared, snapshots, report_path = mixture.prepare(plan, plan_reference=reference)
    after = report["summary"]["after"]
    assert after["family_template_differences"] == after["singleword_full_identity_values_ignored"] == 1
    assert after["canonical_identity_metadata_rows"] == 1
    assert after["name_rows_without_canonical_identity_metadata"] == 0
    mixture.apply_prepared(report, prepared, snapshots, report_path)
    assert prepared[0][0].read_bytes() == b"".join(raw)


def test_pinned_development_metadata_discrepancies_are_conservatively_filtered_and_reported(fixture):
    root, plan, _, _, _ = fixture
    first, second = "a" * 64, "b" * 64
    plan["training"][0], _ = bank(root, "clinical_train", [row("drop-a", "Different A", identity=first),
                                                               row("drop-b", "Different B", identity=second), row("keep", "Retain")])
    dev = row("dev", "Dev wording", identity=first)
    dev["identity_groups"] = [second]
    pin, _ = bank(root, mixture.DEVELOPMENT_BANKS[0], [dev], training=False)
    plan["development"][0] = pin
    mp = Path(pin["manifest_path"])
    manifest = json.loads(mp.read_bytes())
    manifest.pop("rows")
    manifest["cohort_counts"] = {"realistic": 99}
    rewrite(mp, mixture.json_bytes(manifest), frozen=False)
    pin["manifest_sha256"] = mixture.digest(mp.read_bytes())
    reference = freeze_plan(root, plan, audit_for(plan["training"]))
    report, *_ = mixture.prepare(plan, plan_reference=reference)
    assert report["summary"]["after"]["rows"] == 1
    assert len(report["development_validation"][0]["manifest_warnings"]) == 2
    assert mp.read_bytes() == mixture.json_bytes(manifest)
    assert {item["id"] for item in report["corpora"][0]["removals"]} == {"drop-a", "drop-b"}


def test_published_files_are_complete_frozen_and_report_is_last(fixture, monkeypatch):
    _, _, _, _, _ = fixture
    report, prepared, snapshots, report_path = prepare(fixture)
    original = mixture.os.link
    observed = []

    def check_publication(source, path, **kwargs):
        assert source.read_bytes()
        assert stat.S_IMODE(source.stat().st_mode) == 0o444
        assert not report_path.exists()
        if path == report_path:
            assert all(file.read_bytes() == payload for file, payload in prepared)
        observed.append(path)
        return original(source, path, **kwargs)

    monkeypatch.setattr(mixture.os, "link", check_publication)
    mixture.apply_prepared(report, prepared, snapshots, report_path)
    assert observed[-1] == report_path
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in observed)


def test_programmatic_plan_must_equal_pinned_file_before_data_reads(fixture, monkeypatch):
    root, plan, audit, _, _ = fixture
    plan["min_label_support"] = 1
    reference = freeze_plan(root, plan, audit)
    altered = deepcopy(plan)
    altered["min_label_support"] = 0
    reads = []
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (reads.append(path), original(path))[1])
    with pytest.raises(ValueError, match="in-memory plan differs from hash-pinned plan contents"):
        mixture.prepare(altered, plan_reference=reference)
    assert reads == [Path(reference["path"])]
    assert not any(path.exists() for path in slots(plan))
