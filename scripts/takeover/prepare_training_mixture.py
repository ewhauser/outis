"""Prepare a frozen training mixture from an explicit, hash-pinned JSON plan.

Check only (no files written):
  python scripts/takeover/prepare_training_mixture.py --plan PLAN.json --plan-sha256 HEX
Create new __mix3clean.jsonl / .manifest.json pairs and the frozen report:
  python scripts/takeover/prepare_training_mixture.py --plan PLAN.json --plan-sha256 HEX --apply

Plan schema (all paths absolute or relative to this repository, never to cwd):
  {"schema_version": 1,
   "training": [{"path": "data/TRAIN.jsonl", "sha256": "...",
                 "manifest_path": "data/TRAIN.jsonl.manifest.json", "manifest_sha256": "..."}],
   "development": [the same four pin fields for EACH of DEVELOPMENT_BANKS],
   "sanctioned_audit": {"path": "reports/takeover/overlap/AUDIT.json", "sha256": "..."},
   "report_path": "reports/takeover/mix3_001/preparation.json",
   "min_label_support": 0}

Training corpus/manifest pairs and the already completed sanctioned audit must
be regular 0444 files. The audit must cover exactly the planned training inputs.
Development paths are fixed below; their corpus/manifest bytes are hash pinned.
This helper never imports or executes the sealed auditor, discovers datasets,
reads protected banks, changes annotations, or deduplicates retained rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = ROOT.parents[2] if ROOT.parent.name == "worktrees" and ROOT.parent.parent.name in {".claude", ".codex"} else ROOT
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT",
          "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
DEVELOPMENT_BANKS = ("v12_validation", "remediation_v1_validation", "retired_v8_development_001",
                     "tk_train_geriatric__census1_v9clean", "tk_train_primary__census1_v9clean")
SEALED_PIN = "2f06e734bf8b0e8814d0f77a4a1b1048182fe0e9dcee33e439a63dd2b8f06c34"
BLOCKING = ("exact_text", "id", "record_bytes", "canonical_identity_digest",
            "whole_multiword_name", "canonical_vs_sealed_multiword")
AUDIT_KINDS = (*BLOCKING, "singleword_name")
DISCLOSURE = "aggregate counts only; training-side colliding ids for blocking kinds"
PROTECTED = ("blind_", "qualification", ".sealed", "proxyq_", "proxy_", "controls/validation",
             "recovery_data_001", "v15_name_next_step_analysis_001")
FORBIDDEN_ROLES = {"final", "test", "independent", "proxy", "blind", "sealed", "qualification"}
NON_TRAIN = FORBIDDEN_ROLES | {"dev", "development", "validation", "valid", "val", "retired", "heldout"}
ROLE_KEYS = {"split", "suite", "role", "bank_role", "dataset_role", "status"}
PATH_KEYS = {"path", "corpus", "data", "source", "source_path", "manifest_path", "normalized_from", "derived_from"}
PROVENANCE_KEYS = {"source_lineage", "source_provenance", "source_manifest", "provenance",
                   "date_interval_normalization", "normalization", "lineage"}
PIN_KEYS = {"path", "sha256", "manifest_path", "manifest_sha256"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def parse_json(payload):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)


def check_hash(value):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError("every input requires an explicit lowercase SHA-256 pin")
    return value


def safe_path(value, *, training=False, lineage=False):
    """Reject lexical/resolved protected paths and aliases without reading bytes."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("path must be nonempty")
    original = Path(value)
    path = original if original.is_absolute() else ROOT / original
    path = Path(os.path.abspath(path))
    resolved = path.resolve()
    for candidate in (path, resolved):
        if any(word in str(candidate).lower() for word in PROTECTED):
            raise ValueError("refusing protected or proxy path")
        if training:
            tokens = set(re.split(r"[^a-z0-9]+", candidate.name.lower()))
            if tokens & NON_TRAIN or any(part.lower() in NON_TRAIN for part in candidate.parent.parts):
                raise ValueError("refusing non-training path")
    allowed_dev_links = {ROOT / "data" / (name + suffix): MAIN_ROOT / "data" / (name + suffix)
                         for name in DEVELOPMENT_BANKS for suffix in (".jsonl", ".jsonl.manifest.json")}
    if resolved != path and not lineage:
        if training or allowed_dev_links.get(path) != resolved:
            raise ValueError("path aliases and symlinks are not supported outside fixed development links")
    return path


def check_metadata(value, *, training, require_split=False):
    """Check declared roles and nested provenance, not free-form note text."""
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    for key in ("independent_final", "independent", "is_proxy", "is_sealed", "is_qualification"):
        if value.get(key) is not None and value.get(key) is not False:
            raise ValueError("refusing final, independent, proxy or sealed provenance")
    if value.get("proxy_role") or (training and any(v for k, v in value.items() if k.startswith("retired"))):
        raise ValueError("refusing proxy or retired training provenance")
    tokens = {token for key in ROLE_KEYS for token in re.split(r"[^a-z0-9]+", str(value.get(key, "")).lower())}
    if tokens & (NON_TRAIN if training else FORBIDDEN_ROLES):
        raise ValueError("refusing non-training or protected role metadata")
    if training and (require_split or "split" in value) and value.get("split") not in ("train", "training"):
        raise ValueError("training sources and rows must explicitly declare split=train or training")
    for key, child in value.items():
        if key == "development_overlap_removal" and isinstance(child, dict):
            # Exclusion evidence is not a declaration of training provenance.
            check_metadata(child, training=False)
            for selection in child.get("selection", []):
                if not isinstance(selection, dict) or safe_path(selection.get("path")) not in {
                        ROOT / "data" / f"{name}.jsonl" for name in DEVELOPMENT_BANKS}:
                    raise ValueError("exclusion lineage must reference fixed development banks")
        elif key in PATH_KEYS and isinstance(child, str):
            safe_path(child, training=training, lineage=True)
        elif key in PATH_KEYS | PROVENANCE_KEYS:
            children = child if isinstance(child, list) else [child]
            for entry in children:
                if isinstance(entry, dict):
                    check_metadata(entry, training=training)
                elif isinstance(entry, str):
                    safe_path(entry, training=training, lineage=True)
        elif isinstance(child, dict) and key not in {"spans"}:
            # Role flags hidden in an otherwise unrecognized metadata envelope
            # are still authoritative. Path-like values are checked by key.
            check_metadata(child, training=training)
        elif isinstance(child, list) and key not in {"spans", "phenomena"}:
            for entry in child:
                if isinstance(entry, dict):
                    check_metadata(entry, training=training)


def check_file(path, *, frozen):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("inputs must be regular files without hard-link aliases")
    if frozen and stat.S_IMODE(info.st_mode) != 0o444:
        raise ValueError("training inputs, manifests and sanctioned audit must be frozen 0444")
    return info.st_dev, info.st_ino


def read_pinned(path, pin, *, frozen):
    safe_path(path)
    check_file(path, frozen=frozen)
    payload = path.read_bytes()
    if digest(payload) != pin:
        raise ValueError(f"input SHA-256 differs from pin: {path.name}")
    return payload


def new_slot(path):
    path = safe_path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite an output")
    if not path.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    return path


def identities(row, *, strict=True):
    """Metadata only; digest NFC/casefold full identities like the sanctioned audit."""
    result = set()
    groups = {}
    for key in ("identity_group", "identity_groups", "full_identity", "full_identities"):
        if key not in row:
            continue
        value = row[key]
        values = ([value] if value else []) if isinstance(value, str) else value
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("identity metadata must contain nonempty strings")
        if key.startswith("identity_group"):
            # Legacy PHI-free rows use opaque sentinels, not identity digests.
            # Match the sanctioned auditor: preserve, but do not compare them.
            converted = {v for v in values if HEX64.fullmatch(v)}
        else:
            # A few historical rows carry mononyms here. The sanctioned audit
            # compares full multiword names only; group digests still apply.
            converted = {digest(unicodedata.normalize("NFC", v).casefold().encode("utf-8"))
                         for v in values if len(v.split()) >= 2}
        if len(values) != len(set(values)):
            raise ValueError("duplicate canonical identity metadata")
        groups[key] = converted
        result |= converted
    for singular, plural in (("identity_group", "identity_groups"), ("full_identity", "full_identities")):
        if strict and singular in groups and plural in groups and not groups[singular] <= groups[plural]:
            raise ValueError("singular and plural identity metadata disagree")
    return result


def validate_row(row, manifest, *, training):
    check_metadata(row, training=training, require_split=training)
    rid, text, spans = row.get("id"), row.get("text"), row.get("spans")
    if not isinstance(rid, str) or not rid or not isinstance(text, str) or not isinstance(spans, list):
        raise ValueError("rows require nonempty string ids, string text and span lists")
    if not isinstance(row.get("cohort", "unspecified"), str) or not row.get("cohort", "unspecified"):
        raise ValueError("cohort must be a nonempty string")
    if "annotation_policy" in row and "annotation_policy" in manifest and row["annotation_policy"] != manifest["annotation_policy"]:
        raise ValueError("row and manifest annotation policies disagree")
    annotation = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("each span must be an object")
        start, end, label = (span.get(key) for key in ("start", "end", "label"))
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("invalid span offsets")
        if label not in LABELS:
            raise ValueError("unknown span label")
        if "text" in span and span["text"] != text[start:end]:
            raise ValueError("span text differs from exact offsets")
        annotation.append((start, end, label))
    annotation.sort()
    if any(left[1] > right[0] for left, right in zip(annotation, annotation[1:])):
        raise ValueError("duplicate or overlapping spans")
    return {"row": row, "id": rid, "text": text, "annotation": tuple(annotation), "identities": identities(row, strict=training)}


def counts(records):
    support = Counter(span[2] for item in records for span in item["annotation"])
    cohorts = Counter(item["row"].get("cohort", "unspecified") for item in records)
    return {"rows": len(records), "spans": sum(support.values()),
            "label_support": {label: support[label] for label in LABELS},
            "cohort_counts": dict(sorted(cohorts.items())),
            "phi_free_rows": sum(not item["annotation"] for item in records),
            "min_label_support": min(support[label] for label in LABELS),
            "canonical_identity_metadata_rows": sum(bool(item["identities"]) for item in records),
            "name_rows_without_canonical_identity_metadata": sum(
                any(span[2] == "NAME" for span in item["annotation"]) and not item["identities"] for item in records),
            "identity_group_metadata_disagreements": sum(
                isinstance(item["row"].get("identity_group"), str)
                and bool(HEX64.fullmatch(item["row"]["identity_group"]))
                and "identity_groups" in item["row"]
                and item["row"]["identity_group"] not in item["row"]["identity_groups"] for item in records),
            "family_template_differences": sum("family" in item["row"] and "template_family" in item["row"]
                                              and item["row"]["family"] != item["row"]["template_family"] for item in records),
            "singleword_full_identity_values_ignored": sum(
                isinstance(value, str) and bool(value) and len(value.split()) < 2
                for item in records for key in ("full_identity", "full_identities")
                for value in ([item["row"].get(key, "")] if isinstance(item["row"].get(key, ""), str) else item["row"][key]))}


def check_counts(manifest, actual, *, training=True):
    warnings = []
    if not any(key in manifest for key in ("rows", "documents", "count")):
        raise ValueError("manifest must declare a row count")
    for key in ("rows", "documents", "count"):
        if key in manifest and (type(manifest[key]) is not int or manifest[key] != actual["rows"]):
            raise ValueError("manifest row count differs")
    for key in ("label_support", "entity_support", "label_counts"):
        declared = manifest.get(key)
        if key == "label_support" or declared is not None:
            if not isinstance(declared, dict) or set(declared) != set(LABELS) or any(
                    type(declared[label]) is not int or declared[label] != actual["label_support"][label] for label in LABELS):
                raise ValueError("manifest label support differs")
    for key in ("cohort_counts", "cohorts"):
        if key not in manifest:
            continue
        if not isinstance(manifest[key], dict) or any(type(v) is not int or v < 0 for v in manifest[key].values()):
            raise ValueError("manifest cohort counts must be nonnegative integers")
        if manifest[key] != actual["cohort_counts"]:
            if training:
                raise ValueError("manifest cohort counts differ")
            warnings.append({"field": key, "reason": "stale development cohort summary; corpus and manifest remain unchanged",
                             "declared": manifest[key], "actual": actual["cohort_counts"]})
    for key in ("spans", "entity_count", "phi_free_rows"):
        expected = actual["spans" if key == "entity_count" else key]
        if key in manifest and (type(manifest[key]) is not int or manifest[key] != expected):
            raise ValueError("manifest entity or PHI-free count differs")
    return warnings


def audit_entries(audit, training):
    if not isinstance(audit, dict) or audit.get("sealed_pin") != SEALED_PIN or audit.get("disclosure") != DISCLOSURE:
        raise ValueError("expected a sanctioned V9r3 aggregate audit with the pinned sealed identity")
    if not audit.get("finished_utc") or not isinstance(audit.get("corpora"), list):
        raise ValueError("sanctioned audit is incomplete")
    result = {}
    for entry in audit["corpora"]:
        if not isinstance(entry, dict):
            raise ValueError("audit corpus entry must be an object")
        path = safe_path(entry.get("path"), training=True)
        if path in result or path not in training:
            raise ValueError("audit coverage must match explicit training inputs exactly")
        if entry.get("sha256") != training[path]["sha256"] or type(entry.get("rows")) is not int or entry["rows"] < 1:
            raise ValueError("audit source hash or row count differs")
        removals = entry.get("training_ids_to_remove")
        if not isinstance(removals, list) or any(not isinstance(i, str) or not i for i in removals) or len(set(removals)) != len(removals):
            raise ValueError("audit removal IDs must be unique nonempty strings")
        metrics = entry.get("counts")
        if not isinstance(metrics, dict) or set(metrics) != set(AUDIT_KINDS):
            raise ValueError("audit requires all sanctioned aggregate comparison kinds")
        for item in metrics.values():
            if not isinstance(item, dict) or set(item) != {"distinct_shared_digests", "sealed_rows_matching", "source_rows_matching"}:
                raise ValueError("audit must contain aggregate metrics only")
            if any(type(n) is not int or n < 0 for n in item.values()) or item["source_rows_matching"] > entry["rows"]:
                raise ValueError("invalid audit aggregate counts")
            if len({bool(n) for n in item.values()}) != 1:
                raise ValueError("audit aggregate zero counts disagree")
        blocked = any(metrics[k]["distinct_shared_digests"] for k in BLOCKING)
        matching = [metrics[k]["source_rows_matching"] for k in BLOCKING]
        if entry.get("blocked") is not blocked or not max(matching) <= len(removals) <= min(entry["rows"], sum(matching)):
            raise ValueError("audit blocking counts and training removal union disagree")
        result[path] = entry
    if set(result) != set(training) or audit.get("any_blocked") is not any(e["blocked"] for e in result.values()):
        raise ValueError("audit coverage or overall status differs")
    return result


def prepare(plan, *, plan_reference):
    """Fully validate and hold outputs in memory; never writes or performs an audit."""
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("plan schema_version must be 1")
    required = {"schema_version", "training", "development", "sanctioned_audit", "report_path"}
    if not required <= set(plan) or set(plan) - required - {"min_label_support"}:
        raise ValueError("plan has missing or unknown fields")
    minimum = plan.get("min_label_support", 0)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("min_label_support must be a nonnegative integer")
    report_path = new_slot(plan["report_path"])
    allowed_dev = {safe_path(ROOT / "data" / f"{name}.jsonl") for name in DEVELOPMENT_BANKS}
    entries, seen_paths, seen_inodes, outputs = [], set(), set(), {report_path}
    snapshots = []
    # Every input path and output slot is checked before reading even a manifest.
    for role in ("training", "development"):
        if not isinstance(plan[role], list) or not plan[role]:
            raise ValueError("training and development must be nonempty explicit lists")
        for pin in plan[role]:
            if not isinstance(pin, dict) or set(pin) != PIN_KEYS:
                raise ValueError("corpus entries require exactly four explicit corpus/manifest pins")
            path = safe_path(pin["path"], training=role == "training")
            sidecar = safe_path(pin["manifest_path"], training=role == "training")
            if path.suffix != ".jsonl" or sidecar != Path(str(path) + ".manifest.json"):
                raise ValueError("source must be JSONL with its exact adjacent manifest")
            if role == "development" and path not in allowed_dev:
                raise ValueError("development path is outside the five fixed mix2 selection banks")
            if role == "training" and path in allowed_dev:
                raise ValueError("a development selection bank cannot be a training source")
            for file, key in ((path, "sha256"), (sidecar, "manifest_sha256")):
                check_hash(pin[key])
                inode = check_file(file, frozen=role == "training")
                if file in seen_paths or inode in seen_inodes:
                    raise ValueError("duplicate input or filesystem alias")
                seen_paths.add(file)
                seen_inodes.add(inode)
                snapshots.append((file, pin[key], role == "training"))
            item = {**pin, "path": path, "manifest_path": sidecar, "role": role}
            if role == "training":
                if path.stem.endswith("__mix3clean"):
                    raise ValueError("refusing to filter an already prepared mixture derivative")
                item["output"] = new_slot(path.with_name(path.stem + "__mix3clean.jsonl"))
                item["output_manifest"] = new_slot(str(item["output"]) + ".manifest.json")
                for slot in (item["output"], item["output_manifest"]):
                    if slot in outputs:
                        raise ValueError("duplicate output slot")
                    outputs.add(slot)
            entries.append(item)
    if {item["path"] for item in entries if item["role"] == "development"} != allowed_dev:
        raise ValueError("exactly all five fixed mix2 development banks are required")
    audit_pin = plan["sanctioned_audit"]
    if not isinstance(audit_pin, dict) or set(audit_pin) != {"path", "sha256"}:
        raise ValueError("sanctioned_audit requires a path and SHA-256")
    audit_path = safe_path(audit_pin["path"])
    check_hash(audit_pin["sha256"])
    check_file(audit_path, frozen=True)
    if audit_path in seen_paths or audit_path in outputs:
        raise ValueError("audit path overlaps another slot")
    snapshots.append((audit_path, audit_pin["sha256"], True))
    plan_path = safe_path(plan_reference["path"])
    check_hash(plan_reference["sha256"])
    if plan_path in seen_paths | outputs | {audit_path}:
        raise ValueError("plan path overlaps another slot")
    snapshots.append((plan_path, plan_reference["sha256"], False))
    if parse_json(read_pinned(plan_path, plan_reference["sha256"], frozen=False)) != plan:
        raise ValueError("in-memory plan differs from hash-pinned plan contents")
    audit = parse_json(read_pinned(audit_path, audit_pin["sha256"], frozen=True))
    training = {item["path"]: item for item in entries if item["role"] == "training"}
    audited = audit_entries(audit, training)
    for item in entries:
        manifest = parse_json(read_pinned(item["manifest_path"], item["manifest_sha256"], frozen=item["role"] == "training"))
        check_metadata(manifest, training=item["role"] == "training", require_split=item["role"] == "training")
        if manifest.get("sha256") != item["sha256"]:
            raise ValueError("manifest corpus hash differs from pin and audit")
        item["manifest"] = manifest
    all_training_ids = set()
    for item in entries:
        payload = read_pinned(item["path"], item["sha256"], frozen=item["role"] == "training")
        records, ids, lines = [], set(), []
        for raw in payload.splitlines(keepends=True):
            if not raw.strip():
                lines.append((None, raw))
                continue
            record = validate_row(parse_json(raw), item["manifest"], training=item["role"] == "training")
            if record["id"] in ids or (item["role"] == "training" and record["id"] in all_training_ids):
                raise ValueError("duplicate row ID within a corpus or across training mixture")
            ids.add(record["id"])
            if item["role"] == "training":
                all_training_ids.add(record["id"])
            records.append(record)
            lines.append((record["id"], raw))
        if not records:
            raise ValueError("input corpora must not be empty")
        actual = counts(records)
        item["manifest_warnings"] = check_counts(item["manifest"], actual, training=item["role"] == "training")
        if item["role"] == "development" and actual["identity_group_metadata_disagreements"]:
            item["manifest_warnings"].append({"field": "identity_group/identity_groups",
                                             "rows": actual["identity_group_metadata_disagreements"],
                                             "reason": "historical singular/plural mismatch; conservative union of all valid canonical metadata used"})
        if item["role"] == "training":
            audit_item = audited[item["path"]]
            if audit_item["rows"] != len(records) or not set(audit_item["training_ids_to_remove"]) <= ids:
                raise ValueError("audit row count or training removal IDs differ from source")
        item.update(records=records, lines=lines, before=actual)
    # Contradictions are fatal even if an affected row would later be filtered.
    duplicates = defaultdict(list)
    for item in training.values():
        for record in item["records"]:
            duplicates[record["text"]].append((item, record))
    duplicate_report = []
    for values in duplicates.values():
        if len({record["annotation"] for _, record in values}) != 1:
            raise ValueError("contradictory annotations for exact duplicated training text")
        if len(values) > 1:
            duplicate_report.append({"training_rows": [{"path": str(item["path"]), "id": record["id"]} for item, record in values]})
    dev_text, dev_identity = defaultdict(set), defaultdict(set)
    dev_refs = []
    for item in entries:
        if item["role"] == "development":
            dev_refs.append({key: str(item[key]) for key in PIN_KEYS})
            for record in item["records"]:
                dev_text[record["text"]].add(item["path"].stem)
                for identity in record["identities"]:
                    dev_identity[identity].add(item["path"].stem)
    script_path = Path(__file__).resolve()
    script_sha = digest(script_path.read_bytes())
    snapshots.append((script_path, script_sha, False))
    lineage = {"plan": plan_reference, "script": str(script_path), "script_sha256": script_sha,
               "sanctioned_audit": {"path": str(audit_path), "sha256": audit_pin["sha256"]},
               "development": dev_refs, "sealed_bank_opened_by_this_helper": False,
               "retained_row_bytes_preserved": True}
    prepared, corpora, retained = [], [], []
    for item in training.values():
        removals, keep = [], []
        sanctioned = set(audited[item["path"]]["training_ids_to_remove"])
        for record in item["records"]:
            reasons, banks = [], set()
            if record["id"] in sanctioned:
                reasons.append("sanctioned_v9r3_training_id")
            if record["text"] in dev_text:
                reasons.append("development_exact_text")
                banks |= dev_text[record["text"]]
            for identity in record["identities"]:
                banks |= dev_identity.get(identity, set())
            if any(identity in dev_identity for identity in record["identities"]):
                reasons.append("development_canonical_identity")
            if reasons:
                removals.append({"id": record["id"], "reasons": reasons, "development_banks": sorted(banks)})
            else:
                keep.append(record)
        drop_ids = {row["id"] for row in removals}
        payload = b"".join(raw for rid, raw in item["lines"] if rid not in drop_ids)
        after = counts(keep)
        retained.extend(keep)
        source_ref = {key: str(item[key]) for key in PIN_KEYS}
        new_manifest = deepcopy(item["manifest"])
        if "cohorts" in new_manifest:
            new_manifest["cohorts"] = after["cohort_counts"]
        new_manifest.update(corpus=str(item["output"]), sha256=digest(payload), split="train",
                            rows=after["rows"], documents=after["rows"], count=after["rows"],
                            spans=after["spans"], entity_count=after["spans"], phi_free_rows=after["phi_free_rows"],
                            cohort_counts=after["cohort_counts"],
                            training_mixture_preparation={**lineage, "source": source_ref,
                                                          "removed_rows": len(removals), "removals": removals})
        for key in ("label_support", "entity_support", "label_counts"):
            new_manifest[key] = after["label_support"]
        manifest_payload = json_bytes(new_manifest)
        prepared.extend([(item["output"], payload), (item["output_manifest"], manifest_payload)])
        corpora.append({"source": source_ref, "output": {"path": str(item["output"]), "sha256": digest(payload),
                        "manifest_path": str(item["output_manifest"]), "manifest_sha256": digest(manifest_payload)},
                        "before": item["before"], "after": after, "removed_rows": len(removals), "removals": removals})
    aggregate = counts(retained)
    if not retained or any(value < minimum for value in aggregate["label_support"].values()):
        raise ValueError("post-filter mixture is empty or below requested minimum label support")
    overlap = {"exact_text_rows": sum(record["text"] in dev_text for record in retained),
               "canonical_identity_rows": sum(bool(record["identities"] & dev_identity.keys()) for record in retained)}
    if any(overlap.values()):
        raise ValueError("post-filter development overlap must be zero")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "validated_not_written", **lineage, "corpora": corpora,
              "summary": {"training_corpora": len(corpora), "before_rows": sum(c["before"]["rows"] for c in corpora),
                          "removed_rows": sum(c["removed_rows"] for c in corpora), "after": aggregate,
                          "min_label_support": min(aggregate["label_support"].values()),
                          "required_min_label_support": minimum,
                          "removal_reason_counts": dict(Counter(reason for c in corpora for row in c["removals"] for reason in row["reasons"]))},
              "development_validation": [{"path": str(item["path"]), "actual": item["before"],
                                          "manifest_warnings": item["manifest_warnings"]}
                                         for item in entries if item["role"] == "development"],
              "train_vs_development_overlap": overlap,
              "sanctioned_audit_ids_remaining": 0,
              "sealed_overlap_reaudited": False,
              "same_annotation_duplicate_text_groups": duplicate_report,
              "duplicate_text_policy": "Reported across inputs; never silently deduplicated.",
              "outputs_frozen_mode": "0444",
              "publication_policy": "Each file appears complete and frozen; this report is published last as the completion marker."}
    verify_snapshots(snapshots)
    for path in outputs:
        new_slot(path)
    return report, prepared, snapshots, report_path


def verify_snapshots(snapshots):
    for path, pin, frozen in snapshots:
        read_pinned(path, pin, frozen=frozen)


def apply_prepared(report, prepared, snapshots, report_path):
    """Publish complete frozen files exclusively; the report is the commit marker.

    A process kill may leave some complete output files, but never a partial
    corpus or a completion report preceding its corpora. Callers must require
    the frozen_complete report, whose pins identify the complete transaction.
    """
    verify_snapshots(snapshots)
    result = dict(report, status="frozen_complete")
    payloads = [*prepared, (report_path, json_bytes(result))]
    for path, _ in payloads:
        new_slot(path)
    staged, owned = [], []
    try:
        for path, payload in payloads:
            new_slot(path)
            fd, name = tempfile.mkstemp(prefix=".prepare_training_mixture-", suffix=".tmp", dir=path.parent)
            temporary = Path(name)
            staged.append((temporary, path))
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
        verify_snapshots(snapshots)
        for temporary, path in staged:
            new_slot(path)
            # Same-directory hard-link publication is atomic and cannot replace
            # an existing destination (unlike rename/replace).
            info = temporary.stat()
            os.link(temporary, path, follow_symlinks=False)
            owned.append((path, info.st_dev, info.st_ino))
            temporary.unlink()
    except BaseException:
        for path, device, inode in reversed(owned):
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = safe_path(args.plan)
        pin = check_hash(args.plan_sha256)
        plan = parse_json(read_pinned(path, pin, frozen=False))
        report, prepared, snapshots, report_path = prepare(plan, plan_reference={"path": str(path), "sha256": pin})
        if args.apply:
            report = apply_prepared(report, prepared, snapshots, report_path)
        print(json.dumps({"status": report["status"], "report_path": str(report_path) if args.apply else None,
                          "summary": report["summary"], "train_vs_development_overlap": report["train_vs_development_overlap"]}, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"training mixture preparation failed: {error}\n")


if __name__ == "__main__":
    main()
