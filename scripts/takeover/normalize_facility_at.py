"""Plan training-only FACILITY "at <place>" derivatives; dry-run is the default.

Implements the adopted convention that ``at <name>`` continues a facility name
(``Lakeshore Family Medicine at Ellery Falls`` is one FACILITY), matching the
whole-name training banks and frozen development gold. A FACILITY + `` at `` +
LOCATION pair is merged into one FACILITY unless the locality is qualified as
geography (comma, state/postal qualifier, digits), which stays separate. Named
unit-at-facility pairs (two FACILITY spans) and addresses are left unchanged.

Under HIPAA Safe Harbor both conventions remove the same characters; this only
removes a contradictory boundary signal from training.

Only explicit, hash-pinned training inputs are accepted. No development data,
model imports or inference. Outputs are new files opened exclusively.

Example (creates a plan only):
  python scripts/takeover/normalize_facility_at.py --protocol PROTOCOL.json \
      --protocol-sha256 SHA256 --output-dir data --plan NEW_PLAN.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT",
          "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
AUTHORITY = ("User-adopted training convention: 'at <name>' continues a FACILITY name; "
             "qualified geography, addresses and named units at a facility stay separate.")
PROTECTED = ("blind_", "qualification", ".sealed", "proxyq_", "proxy_", "retired",
             "validation", "development", "heldout_", "_dev", "test")
AT_GAP = re.compile(r"[ \t]+at[ \t]+", re.I)
QUALIFIER_AFTER = re.compile(r"[ \t]*,[ \t]*(?:[A-Z]{2}\b|\d{5}\b)")
EXCLUSIONS = ("qualified_geography", "named_unit_at_facility", "address_after_at")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def check_path(path) -> Path:
    path = Path(path)
    for candidate in (path.absolute(), path.resolve()):
        if any(word in candidate.name.lower() for word in PROTECTED):
            raise ValueError(f"refusing protected or non-training path: {candidate.name}")
    return path


def checked_spans(row):
    text, spans = row.get("text"), row.get("spans")
    if not isinstance(text, str) or not isinstance(spans, list):
        raise ValueError("row text must be a string and spans a list")
    for s in spans:
        a, b = s.get("start"), s.get("end")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError("invalid span offsets")
        if s.get("label") not in LABELS:
            raise ValueError("invalid span label")
        if "text" in s and s["text"] != text[a:b]:
            raise ValueError("span text differs from offsets")
    ordered = sorted(enumerate(spans), key=lambda item: item[1]["start"])
    if any(a[1]["end"] > b[1]["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("duplicate or overlapping spans")
    return ordered


def _record(span, text):
    return dict(span, text=text[span["start"]:span["end"]])


def _following(ordered, k):
    return ordered[k + 1][1] if k + 1 < len(ordered) else None


def normalize_row(row):
    """Return a copy, applied merges, and counted non-merges."""
    if row.get("split") not in (None, "train", "training"):
        raise ValueError("refusing non-training row")
    ordered = checked_spans(row)
    text = row["text"]
    changes, exclusions, replacements, removed = [], [], {}, set()
    for k, (i, left) in enumerate(ordered[:-1]):
        j, right = ordered[k + 1]
        if left["label"] != "FACILITY" or not AT_GAP.fullmatch(text[left["end"]:right["start"]]):
            continue
        record = {"row_id": row["id"], "old_spans": [_record(left, text), _record(right, text)]}
        if right["label"] == "FACILITY":
            exclusions.append(dict(record, reason="named_unit_at_facility"))
            continue
        if right["label"] == "ADDRESS":
            exclusions.append(dict(record, reason="address_after_at"))
            continue
        if right["label"] != "LOCATION" or i in removed:
            continue
        place = text[right["start"]:right["end"]]
        nxt = _following(ordered, k + 1)
        qualified = ("," in place or re.search(r"\d", place)
                     or QUALIFIER_AFTER.match(text, right["end"])
                     or (nxt is not None and nxt["label"] in ("LOCATION", "ADDRESS")
                         and re.fullmatch(r"[ \t]*,[ \t]*", text[right["end"]:nxt["start"]])))
        if qualified:
            exclusions.append(dict(record, reason="qualified_geography"))
            continue
        merged = deepcopy(left)
        merged["end"] = right["end"]
        if "text" in merged:
            merged["text"] = text[merged["start"]:merged["end"]]
        replacements[i] = merged
        removed.add(j)
        changes.append(dict(record, new_span=_record(merged, text)))
    output = deepcopy(row)
    output["spans"] = [replacements.get(i, deepcopy(s)) for i, s in enumerate(row["spans"]) if i not in removed]
    checked_spans(output)
    return output, changes, exclusions


def _new_output(path) -> Path:
    path = check_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    return path


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def plan_sources(entries, *, output_dir, suffix="facat1"):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", suffix):
        raise ValueError("suffix must be a simple lowercase name")
    if not entries:
        raise ValueError("explicit training inputs are required")
    output_dir = Path(output_dir)
    script_sha = digest(Path(__file__).read_bytes())
    corpora, prepared, seen = [], [], set()
    for entry in entries:
        source = check_path(entry["path"])
        if source.suffix != ".jsonl" or source.resolve() in seen:
            raise ValueError("each training corpus must be a distinct .jsonl file")
        seen.add(source.resolve())
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError("every source requires an explicit SHA-256 pin")
        manifest_path = check_path(source.with_suffix(source.suffix + ".manifest.json"))
        manifest_blob = manifest_path.read_bytes()
        manifest = json.loads(manifest_blob)
        if manifest.get("split") not in ("train", "training") or manifest.get("independent_final"):
            raise ValueError("manifest must declare a non-final training split")
        payload = source.read_bytes()
        if digest(payload) != entry["sha256"]:
            raise ValueError(f"source SHA-256 differs from pin: {source}")
        before, after, ids, lines = Counter(), Counter(), set(), []
        changes, exclusions, changed_rows = [], [], set()
        for line in payload.splitlines(keepends=True):
            if not line.strip():
                lines.append(line)
                continue
            row = json.loads(line)
            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("corpus ids must be unique nonempty strings")
            ids.add(rid)
            normalized, edits, skipped = normalize_row(row)
            before.update(s["label"] for s in row["spans"])
            after.update(s["label"] for s in normalized["spans"])
            changes.extend(edits)
            exclusions.extend(skipped)
            if edits:
                changed_rows.add(rid)
                ending = b"\n" if line.endswith(b"\n") else b""
                lines.append(json.dumps(normalized, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False).encode("utf-8") + ending)
            else:
                lines.append(line)
        if manifest.get("rows", len(ids)) != len(ids):
            raise ValueError("row count differs from source manifest")
        reasons = Counter(e["reason"] for e in exclusions)
        recorded = entry.get("record_path", str(entry["path"]))
        summary = {"source_path": recorded, "source_sha256": entry["sha256"],
                   "source_manifest_sha256": digest(manifest_blob), "rows": len(ids),
                   "transformation_count": len(changes), "changed_rows": len(changed_rows),
                   "exclusion_counts": {k: reasons[k] for k in EXCLUSIONS},
                   "label_counts_before": {k: before[k] for k in LABELS},
                   "label_counts_after": {k: after[k] for k in LABELS}}
        if changes:
            derivative = b"".join(lines)
            output = _new_output(output_dir / f"{source.stem}__{suffix}.jsonl")
            output_manifest = _new_output(output.with_suffix(output.suffix + ".manifest.json"))
            new_manifest = deepcopy(manifest)
            new_manifest.update(corpus=str(output), sha256=digest(derivative), rows=len(ids),
                                documents=len(ids), count=len(ids))
            for key in ("label_support", "entity_support", "label_counts"):
                if key in new_manifest:
                    new_manifest[key] = {label: after[label] for label in LABELS}
            if "spans" in new_manifest:
                new_manifest["spans"] = sum(after.values())
            if "entity_count" in new_manifest:
                new_manifest["entity_count"] = sum(after.values())
            new_manifest["facility_at_normalization"] = {
                "version": 1, "authority": AUTHORITY,
                "script": "scripts/takeover/normalize_facility_at.py", "script_sha256": script_sha,
                "source": {"path": recorded, "sha256": entry["sha256"],
                           "manifest_sha256": digest(manifest_blob)},
                "transformation_count": len(changes), "changed_rows": len(changed_rows),
                "row_text_ids_and_other_spans_preserved": True}
            summary.update(output_path=str(output), output_manifest_path=str(output_manifest),
                           proposed_output_sha256=digest(derivative))
            prepared.append({"source": source, "pin": entry["sha256"], "output": output,
                             "output_manifest": output_manifest, "payload": derivative,
                             "manifest": new_manifest, "script_sha256": script_sha})
        summary.update(transformations=changes, exclusions=exclusions)
        corpora.append(summary)
    plan = {"schema_version": 1, "mode": "dry_run", "authority": AUTHORITY, "script_sha256": script_sha,
            "scope": "Explicit hash-pinned training corpora only; no development, proxy or protected-bank reads.",
            "predicates": {"gap": AT_GAP.pattern, "qualifier_after": QUALIFIER_AFTER.pattern,
                           "qualified_place": "LOCATION containing a comma or digit, or followed by ', ST', a ZIP, "
                                              "or a comma-adjacent LOCATION/ADDRESS span"},
            "summary": {"corpora": len(corpora), "rows": sum(c["rows"] for c in corpora),
                        "transformations": sum(c["transformation_count"] for c in corpora),
                        "changed_rows": sum(c["changed_rows"] for c in corpora),
                        "changed_corpora": [c["source_path"] for c in corpora if c["transformation_count"]],
                        "exclusions": {k: sum(c["exclusion_counts"][k] for c in corpora) for k in EXCLUSIONS}},
            "corpora": corpora}
    return plan, prepared


def apply_prepared(prepared):
    for item in prepared:
        if digest(item["source"].read_bytes()) != item["pin"]:
            raise ValueError("source SHA-256 changed after planning")
        if digest(Path(__file__).read_bytes()) != item["script_sha256"]:
            raise ValueError("normalizer source changed after planning")
        _new_output(item["output"])
        _new_output(item["output_manifest"])
    for item in prepared:
        item["output"].parent.mkdir(parents=True, exist_ok=True)
        with item["output"].open("xb") as handle:
            handle.write(item["payload"])
        with item["output_manifest"].open("xb") as handle:
            handle.write(_json_bytes(item["manifest"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--protocol", help="frozen protocol whose training_files are the inputs")
    inputs.add_argument("--corpus", action="append", help="explicit TRAIN.jsonl=SHA256; repeat as needed")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--root", default=".", help="directory protocol-relative corpus paths resolve against")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="facat1")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    plan_path = _new_output(args.plan)
    provenance = None
    if args.protocol:
        blob = Path(args.protocol).read_bytes()
        if not args.protocol_sha256 or digest(blob) != args.protocol_sha256:
            raise ValueError("protocol requires a matching --protocol-sha256 pin")
        protocol = json.loads(blob)
        development = {d["path"] for d in protocol.get("development_files", [])}
        entries = []
        for e in protocol["training_files"]:
            if e["path"] in development:
                raise ValueError("training and development paths overlap")
            entries.append({"path": str(Path(args.root) / e["path"]), "record_path": e["path"],
                            "sha256": e["sha256"]})
        provenance = {"path": args.protocol, "sha256": args.protocol_sha256}
    else:
        entries = []
        for pair in args.corpus:
            path, sep, checksum = pair.rpartition("=")
            if not path or not sep:
                raise ValueError("--corpus requires PATH=SHA256")
            entries.append({"path": path, "sha256": checksum})
    plan, prepared = plan_sources(entries, output_dir=args.output_dir, suffix=args.suffix)
    if provenance:
        plan["protocol"] = provenance
    if args.apply:
        apply_prepared(prepared)
        plan["mode"] = "applied"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("xb") as handle:
        handle.write(_json_bytes(plan))
    print(json.dumps({"plan": args.plan, "applied": args.apply, **plan["summary"]}))


if __name__ == "__main__":
    main()
