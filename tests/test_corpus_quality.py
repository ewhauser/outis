import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from clinical_deid.data.annotation_policy import POLICY_VERSION, canonical_identity_group, identity_split
from clinical_deid.data.corpus_quality import ProtectedCorpusError, audit_corpora, read_development_bytes


def row(identifier="one", role="train", *, negative=False, name=None):
    if name is None:
        for number in range(100):
            candidate = f"Mira Patel{number}"
            if identity_split(canonical_identity_group(candidate)) == role:
                name = candidate
                break
    text = "She walked for 14 days." if negative else f"Patient name: {name}. She returned."
    start = text.find(name)
    return {"id": identifier, "text": text,
            "spans": [] if negative else [{"start": start, "end": start + len(name), "label": "NAME"}],
            "split": role, "annotation_policy": POLICY_VERSION,
            "identity_group": canonical_identity_group(name), "pair_id": f"{role}-{identifier}",
            "template_family": f"{role}-clinical", "phenomena": ["ordinary_pronouns"], "cohort": "realistic"}


def write(path, rows):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
    role = rows[0].get("split") if rows else "train"
    if role not in ("train", "validation"):
        role = "train"
    manifest(path, role)
    return path


def manifest(path, role="train", **extra):
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    sidecar.write_text(json.dumps({"split": role, "annotation_policy": POLICY_VERSION,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}))
    return sidecar


def codes(result):
    return {error["code"] for error in result["errors"]}


def test_audit_preserves_bytes_and_reports_real_token_counts(tmp_path):
    positive, negative = row(), row("two", negative=True)
    negative["identity_groups"] = []
    path = write(tmp_path / "training.jsonl", [positive, negative])
    before = path.read_bytes()
    result = audit_corpora({"train": path}, token_counter=lambda text: len(text.split()), tokenizer_name="test-word-counter")
    assert result["passed"]
    assert path.read_bytes() == before
    assert result["inputs"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["inputs"][0]["role"] == "train"
    assert result["label_support"]["train"]["NAME"]["spans"] == 1
    assert result["label_support"]["train"]["DATE"]["spans"] == 0
    assert result["document_strata"] == {"train/phi_bearing": 1, "train/phi_free": 1}
    assert result["distributions"]["document_tokens"]["train"]["count"] == 2
    assert result["tokenizer"] == "test-word-counter"
    assert result["context_cues"]["train/phi_free"]["counts"]["ordinary_pronoun"] == 1


def test_duplicate_text_id_and_conflicting_annotations_are_hard_defects(tmp_path):
    first = row()
    second = copy.deepcopy(first)
    second["spans"] = []
    path = write(tmp_path / "training.jsonl", [first, second])
    result = audit_corpora({"train": path})
    assert not result["passed"]
    assert {"duplicate_id", "duplicate_text", "conflicting_annotations"} <= codes(result)


def test_shared_primary_secondary_identity_pair_and_family_fail_across_splits(tmp_path):
    train = row()
    development = row("two", "validation")
    development["identity_groups"] = [development["identity_group"], train["identity_group"]]
    development["pair_id"] = train["pair_id"]
    development["template_family"] = train["template_family"]
    paths = {"train": write(tmp_path / "training.jsonl", [train]), "validation": write(tmp_path / "development.jsonl", [development])}
    result = audit_corpora(paths)
    assert {"identity_partition", "cross_split_identity_group", "cross_split_pair_id", "cross_split_template_family"} <= codes(result)


def test_within_split_contrasts_and_same_spelling_in_context_are_allowed(tmp_path):
    first = row(name="Care", negative=True)
    # Make the synthetic primary identity comply with the shared partition.
    first["split"] = identity_split(first["identity_group"])
    role = first["split"]
    first["text"] = "Care history reviewed. She returned."
    second = copy.deepcopy(first)
    second.update(id="two", text="Patient name: Care. She returned.", cohort="unusual_name")
    second["spans"] = [{"start": 14, "end": 18, "label": "NAME"}]
    path = write(tmp_path / "corpus.jsonl", [first, second])
    result = audit_corpora({role: path})
    assert result["passed"]
    assert result["cohort_support"][f"{role}/unusual_name"]["NAME"] == 1


def test_masked_templates_and_cue_imbalance_remain_diagnostics(tmp_path):
    train, development = row(), row("two", "validation")
    paths = {"train": write(tmp_path / "training.jsonl", [train]), "validation": write(tmp_path / "development.jsonl", [development])}
    result = audit_corpora(paths)
    assert result["passed"]
    assert result["masked_templates"]["cross_split_unique"] == 1
    assert "masked_template_overlap" in {item["code"] for item in result["warnings"]}
    assert result["distributions"]["document_tokens"] == {}
    assert result["tokenizer"] is None


def test_policy_explanations_fail_strict_but_are_visible_in_legacy_mode(tmp_path):
    sample = row(negative=True)
    sample["text"] = "Age below 90 remains clinical information."
    path = write(tmp_path / "training.jsonl", [sample])
    strict = audit_corpora({"train": path})
    assert "annotation_explanation" in codes(strict)
    legacy = audit_corpora({"train": path}, strict_policy=False)
    assert legacy["passed"]
    assert {"annotation_explanation", "legacy_mode"} <= {item["code"] for item in legacy["warnings"]}


@pytest.mark.parametrize("change,code", [
    ({"spans": [{"start": True, "end": 5, "label": "NAME"}]}, "invalid_annotation"),
    ({"annotation_policy": "old"}, "policy_version"),
    ({"identity_group": "ordinary-name"}, "identity_group_format"),
    ({"identity_groups": []}, "identity_groups_metadata"),
    ({"phenomena": ["one", "one"]}, "phenomena_metadata"),
    ({"phenomena": [{"bad": "shape"}]}, "phenomena_metadata"),
    ({"cohort": "combined"}, "cohort_metadata"),
    ({"split": []}, "split_mismatch"),
])
def test_invalid_new_metadata_and_annotations_fail(tmp_path, change, code):
    sample = row()
    sample.update(change)
    result = audit_corpora({"train": write(tmp_path / "training.jsonl", [sample])})
    assert code in codes(result)


def test_legacy_mode_does_not_claim_policy_compliance_or_relax_structure(tmp_path):
    sample = row()
    for key in ("annotation_policy", "identity_group", "pair_id", "template_family", "phenomena", "cohort"):
        sample.pop(key)
    path = write(tmp_path / "training.jsonl", [sample])
    assert audit_corpora({"train": path}, strict_policy=False)["passed"]
    assert not audit_corpora({"train": path})["passed"]
    write(path, [sample, sample])
    assert not audit_corpora({"train": path}, strict_policy=False)["passed"]


@pytest.mark.parametrize("filename", ["blind_qualification_v6.jsonl", "dataset.final.jsonl", "dataset_test.jsonl", "x.sealed.jsonl", "x.sealed", "attempt_ledger.jsonl"])
def test_protected_paths_fail_before_any_file_open(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    def unexpected_open(*args, **kwargs):
        raise AssertionError("Protected path must be rejected before opening")
    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(ProtectedCorpusError):
        read_development_bytes(path)


def test_symlink_alias_and_directory_alias_are_rejected_before_reading(tmp_path, monkeypatch):
    source = write(tmp_path / "training.jsonl", [row()])
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(source)
    nested = tmp_path / "linkdir"
    nested.symlink_to(tmp_path, target_is_directory=True)
    def unexpected_open(*args, **kwargs):
        raise AssertionError("Symlink must be rejected before opening")
    monkeypatch.setattr(os, "open", unexpected_open)
    for path in (alias, nested / source.name):
        with pytest.raises(ProtectedCorpusError):
            read_development_bytes(path)


def test_protected_rows_are_refused_without_diagnostics(tmp_path):
    sample = row()
    sample["split"] = "test"
    result = audit_corpora({"train": write(tmp_path / "development.jsonl", [sample])})
    assert codes(result) == {"protected_row"}
    assert result["valid_structural_rows"] == 0
    assert result["masked_templates"]["unique"] == 0
    assert result["context_cues"] == {}


def test_malformed_empty_and_non_utf8_corpora_fail(tmp_path):
    for content, code in [(b"", "empty_corpus"), (b"oops\n", "invalid_json"), (b"[]\n", "invalid_row"), (b"\xff", "invalid_encoding")]:
        path = tmp_path / "training.jsonl"
        path.write_bytes(content)
        manifest(path)
        assert code in codes(audit_corpora({"train": path}))


def test_manifest_preflight_rejects_hidden_protected_corpus_before_payload_read(tmp_path, monkeypatch):
    import clinical_deid.data.corpus_quality as module
    path = write(tmp_path / "development.jsonl", [row()])
    manifest(path, suite="final", independent_final=True)
    original = module._read_scoped_bytes
    def only_manifests(path, *, manifest=False):
        assert manifest, "A protected declaration must prevent payload reading"
        return original(path, manifest=manifest)
    monkeypatch.setattr(module, "_read_scoped_bytes", only_manifests)
    with pytest.raises(ProtectedCorpusError):
        audit_corpora({"train": path})


def test_all_manifests_preflight_before_any_payload_and_hash_mismatch_fails(tmp_path, monkeypatch):
    import clinical_deid.data.corpus_quality as module
    train = write(tmp_path / "training.jsonl", [row()])
    development = write(tmp_path / "development.jsonl", [row("two", "validation")])
    manifest(development, "validation", independent_final=True)
    original = module._read_scoped_bytes
    def only_manifests(path, *, manifest=False):
        assert manifest, "All declarations must be checked before any payload"
        return original(path, manifest=manifest)
    monkeypatch.setattr(module, "_read_scoped_bytes", only_manifests)
    with pytest.raises(ProtectedCorpusError):
        audit_corpora({"train": train, "validation": development})
    monkeypatch.setattr(module, "_read_scoped_bytes", original)
    manifest(train, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        audit_corpora({"train": train})


def test_strict_requires_manifest_but_legacy_does_not(tmp_path):
    path = write(tmp_path / "training.jsonl", [row()])
    path.with_suffix(path.suffix + ".manifest.json").unlink()
    with pytest.raises(ValueError, match="requires an adjacent manifest"):
        audit_corpora({"train": path})
    assert audit_corpora({"train": path}, strict_policy=False)["passed"]


def test_support_caveat_uses_actual_two_sided_wilson_threshold(tmp_path):
    sample = row()
    sample["text"] = " ".join(["Ana"] * 200)
    sample["spans"] = [{"start": i * 4, "end": i * 4 + 3, "label": "NAME"} for i in range(200)]
    result = audit_corpora({"train": write(tmp_path / "training.jsonl", [sample])})
    support = result["label_support"]["train"]["NAME"]
    assert support["spans"] == 200
    assert support["perfect_recall_wilson_lower"] == pytest.approx(.98115467)
    assert not support["could_demonstrate_99pct_wilson"]


def test_cli_writes_a_new_bound_report_and_refuses_overwrite(tmp_path):
    path = write(tmp_path / "training.jsonl", [row()])
    output = tmp_path / "quality.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_corpus_quality.py"
    command = [sys.executable, str(script), "--corpus", f"train={path}", "--tokenizer", "none", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    assert json.loads(original)["inputs"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
