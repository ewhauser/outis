"""Plan training-only DATE interval derivatives; dry-run is the default.

This implements the user's adopted training convention, not a universal claim
about clinical-phi-v2 or the census's narrower compressed-range example. Only
explicit, hash-pinned training inputs are accepted. No directory discovery,
model imports, inference, overlap auditor, or development data access occurs.

Example (creates a plan only):
  python scripts/takeover/normalize_date_intervals.py --protocol PROTOCOL.json \
      --protocol-sha256 SHA256 --plan NEW_PLAN.json

Alternatively repeat --corpus TRAIN.jsonl=SHA256. --apply additionally creates
new corpus/manifest pairs after all inputs and output slots pass preflight.
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
AUTHORITY = "User-adopted training convention: a written range is one DATE; not a universal core-policy claim."
PROTECTED = ("blind_", "qualification", ".sealed", "proxyq_", "proxy_", "retired",
             "controls/validation", "recovery_data_001", "v15_name_next_step_analysis_001")
NON_TRAIN = {"dev", "development", "validation", "valid", "val", "test", "final",
             "independent", "proxy", "blind", "sealed", "qualification", "retired", "heldout"}
ROLE_KEYS = ("split", "suite", "role", "bank_role", "dataset_role", "status")
PROVENANCE_KEYS = ("source", "source_lineage", "source_provenance", "normalized_from", "derived_from",
                   "source_manifest", "date_interval_normalization")
CONNECTOR = re.compile(r"\s*(?P<connector>[-–—−]|to|through|thru|until|and|or|,|;)\s*", re.I)
AUDIT_REVISION = re.compile(r"\b(?:moved|rescheduled|corrected|shifted|amended|postponed|revised|replaced)\b", re.I)
REVISION = re.compile(r"\b(?:moved|rescheduled|corrected|shifted|amended|postponed|revised|replaced|"
                      r"change[ds]?|adjusted|switched|delayed)\b", re.I)
COMPARISON = re.compile(r"\b(?:compar(?:e[ds]?|ing|ison)|versus|vs\.?)\b", re.I)
DISTINCT = re.compile(r"\b(?:on|dated|visit dates|appointment dates)\s*:?\s*$|"
                      r"\bcomparison is made to\b[^\n]*\bof\s*$|"
                      r"\b(?:supporting clinic notes|records) from\s*$", re.I)
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
EXISTING_INTERVAL = re.compile(
    rf"\b(?:to|through|thru|until|and|or)\b|[–—−]|\b\d{{4}}\s*-\s*\d{{4}}\b|"
    rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*-\s*\d{{1,2}}\b|"
    rf"\b\d{{1,2}}\s*-\s*\d{{1,2}}\s+{MONTH}\b", re.I)
NUMERIC_DATE = re.compile(r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b")
EXCLUSIONS = ("newline_boundary", "alternative_or_dates", "explicit_distinct_date_list",
              "uncertain_and_dates", "date_revision_context", "comparison_context",
              "existing_interval_endpoint", "incompatible_DATE_metadata", "chained_interval")
LINE_BREAKS = "\n\r\v\f\x85\u2028\u2029"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def check_path(path):
    """Lexical and resolved checks, before opening corpus, sidecar, or output."""
    path = Path(path)
    for candidate in (path.absolute(), path.resolve()):
        lowered = str(candidate).lower()
        filename_tokens = set(re.split(r"[^a-z0-9]+", candidate.name.lower()))
        if (any(word in lowered for word in PROTECTED)
                or filename_tokens & NON_TRAIN
                or any(part.lower() in NON_TRAIN for part in candidate.parent.parts)):
            raise ValueError("refusing protected, proxy, retired or non-training path")
    return path


def check_training(metadata, *, require_role=False):
    """Inspect declarations before corpus bytes; also reject copied aliases."""
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must be an object")
    if (any(metadata.get(key) not in (None, False) for key in ("independent_final", "independent"))
            or metadata.get("proxy_role") or metadata.get("is_proxy")
            or any(value for key, value in metadata.items() if key.startswith("retired"))):
        raise ValueError("refusing final, independent, proxy or retired metadata")
    tokens = {token for key in ROLE_KEYS
              for token in re.split(r"[^a-z0-9]+", str(metadata.get(key, "")).lower())}
    if tokens & NON_TRAIN:
        raise ValueError("refusing non-training role metadata")
    if (require_role or "split" in metadata) and metadata.get("split") not in ("train", "training"):
        raise ValueError("manifest must explicitly declare split=train or training")
    for key in ("path", "corpus", "data", "source", "source_path", "manifest_path"):
        if isinstance(metadata.get(key), str):
            check_path(metadata[key])
    for key in PROVENANCE_KEYS:
        if isinstance(metadata.get(key), dict):
            check_training(metadata[key])
        elif isinstance(metadata.get(key), list):
            for declaration in metadata[key]:
                if isinstance(declaration, dict):
                    check_training(declaration)
                elif isinstance(declaration, str):
                    check_path(declaration)


def checked_spans(row):
    text, spans = row.get("text"), row.get("spans")
    if not isinstance(text, str) or not isinstance(spans, list):
        raise ValueError("row text must be a string and spans a list")
    for s in spans:
        if not isinstance(s, dict):
            raise ValueError("span must be an object")
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


def _span_record(span, text):
    return dict(span, text=text[span["start"]:span["end"]])


def _pair_reason(text, left, right, connector, prefix, *, audit=False):
    gap = text[left["end"]:right["start"]]
    if "\n" in gap or "\r" in gap:
        return "newline_boundary"
    if connector == "or":
        return "alternative_or_dates"
    if connector in {",", ";"}:
        return "explicit_distinct_date_list"
    if connector == "and" and not re.search(r"\bbetween\s*$", prefix, re.I):
        return "explicit_distinct_date_list" if DISTINCT.search(prefix) else "uncertain_and_dates"
    if (AUDIT_REVISION if audit else REVISION).search(prefix):
        return "date_revision_context"
    if audit:
        return None
    if any(c in text[left["start"]:right["end"]] for c in LINE_BREAKS):
        return "newline_boundary"
    if COMPARISON.search(prefix):
        return "comparison_context"
    for span in (left, right):
        value = text[span["start"]:span["end"]]
        if (EXISTING_INTERVAL.search(value) or len(NUMERIC_DATE.findall(value)) > 1
                or len(re.findall(rf"\b{MONTH}\b", value, re.I)) > 1):
            return "existing_interval_endpoint"
    def extra(span):
        return {k: v for k, v in span.items() if k not in {"start", "end", "text"}}
    if extra(left) != extra(right):
        return "incompatible_DATE_metadata"
    return None


def normalize_row(row):
    """Return a copy, exact proposed transformations, and explicit exclusions."""
    check_training(row)
    ordered = checked_spans(row)
    text = row["text"]
    pairs = []
    for (i, left), (j, right) in zip(ordered, ordered[1:]):
        if left["label"] != "DATE" or right["label"] != "DATE":
            continue
        match = CONNECTOR.fullmatch(text[left["end"]:right["start"]])
        if not match:
            continue
        connector = match["connector"].lower()
        prefix = text[max(0, left["start"] - 100):left["start"]]
        reason = _pair_reason(text, left, right, connector, prefix)
        record = {"row_id": row["id"], "old_spans": [_span_record(left, text), _span_record(right, text)],
                  "connector": connector,
                  "audit_baseline_eligible": _pair_reason(text, left, right, connector, prefix, audit=True) is None}
        pairs.append((i, j, reason, record))
    # Reject every edge of a connected range chain; never choose an arbitrary
    # pair that could merge farther on a second normalization pass.
    uses = Counter(index for i, j, reason, _ in pairs if reason is None for index in (i, j))
    ambiguous = {index for index, count in uses.items() if count > 1}
    # "Between A and B and C" does not identify a two-endpoint interval.
    for i, j, reason, record in pairs:
        if reason is None and record["connector"] == "and":
            if any((a == j or b == i) and other["connector"] in {"and", "or"}
                   for a, b, _, other in pairs):
                ambiguous.update((i, j))
    while True:
        expanded = ambiguous | {index for i, j, reason, _ in pairs
                                if reason is None and ({i, j} & ambiguous) for index in (i, j)}
        if expanded == ambiguous:
            break
        ambiguous = expanded
    output, changes, exclusions = deepcopy(row), [], []
    replacements, removed = {}, set()
    for i, j, reason, record in pairs:
        if reason is None and {i, j} & ambiguous:
            reason = "chained_interval"
        if reason:
            exclusions.append(dict(record, reason=reason))
            continue
        left, right = row["spans"][i], row["spans"][j]
        new = deepcopy(left)
        new["end"] = right["end"]
        if "text" in left or "text" in right:
            new["text"] = text[new["start"]:new["end"]]
        replacements[i], removed = new, removed | {j}
        changes.append(dict(record, new_span=_span_record(new, text)))
    output["spans"] = [replacements.get(i, deepcopy(s)) for i, s in enumerate(row["spans"]) if i not in removed]
    checked_spans(output)
    return output, changes, exclusions


def _new_output(path):
    path = check_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    return path


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def plan_sources(entries, *, suffix="dateinterval1"):
    """Preflight all paths, then all manifests, then read only pinned training."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", suffix):
        raise ValueError("suffix must be a simple lowercase name")
    if not entries:
        raise ValueError("explicit training inputs are required")
    items, seen = [], set()
    for entry in entries:
        source = check_path(entry["path"])
        if source.suffix != ".jsonl":
            raise ValueError("training corpus must have .jsonl extension")
        manifest_path = check_path(source.with_suffix(source.suffix + ".manifest.json"))
        output = _new_output(source.with_name(f"{source.stem}__{suffix}.jsonl"))
        output_manifest = _new_output(output.with_suffix(output.suffix + ".manifest.json"))
        if source.resolve() in seen:
            raise ValueError("a training corpus may appear only once, including aliases")
        seen.add(source.resolve())
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError("every source requires an explicit SHA-256 pin")
        if "rows" in entry and (type(entry["rows"]) is not int or entry["rows"] < 1):
            raise ValueError("explicit row count must be a positive integer")
        items.append({"source": source, "manifest_path": manifest_path, "output": output,
                      "output_manifest": output_manifest, "pin": entry})
    # All role declarations pass before any corpus bytes are read.
    for item in items:
        blob = item["manifest_path"].read_bytes()
        manifest = json.loads(blob)
        check_training(manifest, require_role=True)
        if manifest.get("sha256") != item["pin"]["sha256"]:
            raise ValueError("source manifest SHA-256 differs from explicit source pin")
        item.update(source_manifest=manifest, manifest_sha256=digest(blob))
    script_sha = digest(Path(__file__).read_bytes())
    corpora, prepared = [], []
    for item in items:
        payload = item["source"].read_bytes()
        if digest(payload) != item["pin"]["sha256"]:
            raise ValueError("source SHA-256 differs from explicit source pin")
        original_support, support, ids = Counter(), Counter(), set()
        transformed_lines, changes, exclusions = [], [], []
        changed_rows = set()
        for line in payload.splitlines(keepends=True):
            if not line.strip():
                transformed_lines.append(line)
                continue
            row = json.loads(line)
            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("corpus ids must be unique nonempty strings")
            ids.add(rid)
            normalized, edits, skipped = normalize_row(row)
            original_support.update(s["label"] for s in row["spans"])
            support.update(s["label"] for s in normalized["spans"])
            changes.extend(edits)
            exclusions.extend(skipped)
            if edits:
                changed_rows.add(rid)
                ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b"\r" if line.endswith(b"\r") else b""
                transformed_lines.append(json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + ending)
            else:
                transformed_lines.append(line)
        manifest = item["source_manifest"]
        counts = [manifest[k] for k in ("rows", "documents", "count") if k in manifest]
        if (not ids or not counts or any(type(n) is not int or n != len(ids) for n in counts)
                or ("rows" in item["pin"] and item["pin"]["rows"] != len(ids))):
            raise ValueError("row count differs from source manifest or explicit pin")
        for key in ("label_support", "entity_support", "label_counts"):
            declared = manifest.get(key)
            if key == "label_support" or declared is not None:
                if not isinstance(declared, dict) or any(type(declared.get(k)) is not int or declared[k] != original_support[k] for k in LABELS):
                    raise ValueError("label counts differ from source manifest")
        derivative = b"".join(transformed_lines)
        lineage = {"version": 1, "authority": AUTHORITY, "script": "scripts/takeover/normalize_date_intervals.py",
                   "script_sha256": script_sha,
                   "source": {"path": str(item["source"]), "sha256": item["pin"]["sha256"],
                              "manifest_path": str(item["manifest_path"]), "manifest_sha256": item["manifest_sha256"]},
                   "source_manifest": deepcopy(manifest), "transformation_count": len(changes),
                   "changed_rows": len(changed_rows), "row_text_ids_and_non_DATE_spans_preserved": True}
        new_manifest = deepcopy(manifest)
        new_manifest.update(corpus=str(item["output"]), sha256=digest(derivative), rows=len(ids), documents=len(ids), count=len(ids),
                            date_interval_normalization=lineage)
        for key in ("label_support", "entity_support", "label_counts"):
            new_manifest[key] = {label: support[label] for label in LABELS}
        reasons = Counter(e["reason"] for e in exclusions)
        corpora.append({"source_path": str(item["source"]), "source_sha256": item["pin"]["sha256"],
                        "source_manifest_path": str(item["manifest_path"]), "source_manifest_sha256": item["manifest_sha256"],
                        "rows": len(ids), "output_path": str(item["output"]), "output_manifest_path": str(item["output_manifest"]),
                        "proposed_output_sha256": digest(derivative), "label_counts_before": {k: original_support[k] for k in LABELS},
                        "label_counts_after": {k: support[k] for k in LABELS}, "transformation_count": len(changes),
                        "changed_rows": len(changed_rows), "exclusion_counts": {k: reasons[k] for k in EXCLUSIONS},
                        "transformations": changes, "exclusions": exclusions})
        prepared.append(dict(item, payload=derivative, manifest=new_manifest, script_sha256=script_sha))
    baseline = [e for c in corpora for e in c["transformations"] + c["exclusions"] if e["audit_baseline_eligible"]]
    plan = {"schema_version": 1, "mode": "dry_run", "authority": AUTHORITY, "script_sha256": script_sha,
            "scope": "Explicit hash-pinned training corpora only; no development, proxy or protected-bank reads.",
            "predicates": {"connector": CONNECTOR.pattern, "revision": REVISION.pattern, "comparison": COMPARISON.pattern,
                           "distinct_list": DISTINCT.pattern, "existing_interval": EXISTING_INTERVAL.pattern,
                           "boundary": "Same line, adjacent DATE annotations, no intervening annotation; no overlapping or chained merges.",
                           "metadata": "DATE extra metadata must agree; all row metadata and non-DATE annotations preserved.",
                           "serialization": "Unchanged JSONL lines and line endings are byte-preserved; changed rows retain exact text and all other metadata values."},
            "summary": {"corpora": len(corpora), "rows": sum(c["rows"] for c in corpora),
                        "transformations": sum(c["transformation_count"] for c in corpora), "changed_rows": sum(c["changed_rows"] for c in corpora),
                        "audit_baseline_pairs": len(baseline),
                        "audit_baseline_rows": sum(len({e["row_id"] for e in c["transformations"] + c["exclusions"] if e["audit_baseline_eligible"]}) for c in corpora),
                        "conservative_exclusions_from_audit": dict(Counter(e["reason"] for c in corpora for e in c["exclusions"] if e["audit_baseline_eligible"])),
                        "exclusions": {k: sum(c["exclusion_counts"][k] for c in corpora) for k in EXCLUSIONS}},
            "corpora": corpora}
    _verify_prepared(prepared)
    return plan, prepared


def _verify_prepared(prepared):
    for item in prepared:
        for key in ("source", "manifest_path"):
            check_path(item[key])
        for key in ("output", "output_manifest"):
            _new_output(item[key])
        if digest(item["source"].read_bytes()) != item["pin"]["sha256"]:
            raise ValueError("source SHA-256 changed after planning")
        if digest(item["manifest_path"].read_bytes()) != item["manifest_sha256"]:
            raise ValueError("source manifest SHA-256 changed after planning")
        if digest(Path(__file__).read_bytes()) != item["script_sha256"]:
            raise ValueError("normalizer source changed after planning")


def write_plan(path, plan):
    path = _new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(plan))


def apply_prepared(prepared):
    """Explicit application only; never called by a default dry run."""
    _verify_prepared(prepared)
    # Exclusive opens prevent overwrites even if an output appears after preflight.
    for item in prepared:
        with item["output"].open("xb") as handle:
            handle.write(item["payload"])
        with item["output_manifest"].open("xb") as handle:
            handle.write(_json_bytes(item["manifest"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--protocol")
    inputs.add_argument("--corpus", action="append", help="explicit TRAIN.jsonl=SHA256; repeat as needed")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--suffix", default="dateinterval1")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    _new_output(args.plan)
    provenance = None
    if args.protocol:
        path = check_path(args.protocol)
        if not args.protocol_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256):
            raise ValueError("protocol requires an explicit SHA-256 pin")
        blob = path.read_bytes()
        if digest(blob) != args.protocol_sha256:
            raise ValueError("protocol SHA-256 differs")
        entries = json.loads(blob)["training_files"]
        provenance = {"path": str(path), "sha256": args.protocol_sha256}
    else:
        entries = []
        for pair in args.corpus:
            path, sep, checksum = pair.rpartition("=")
            if not path or not sep:
                raise ValueError("--corpus requires PATH=SHA256")
            entries.append({"path": path, "sha256": checksum})
    plan, prepared = plan_sources(entries, suffix=args.suffix)
    if provenance:
        if digest(check_path(args.protocol).read_bytes()) != provenance["sha256"]:
            raise ValueError("protocol changed during planning")
        plan["protocol"] = provenance
    write_plan(args.plan, plan)
    if args.apply:
        apply_prepared(prepared)
    print(json.dumps({"plan": args.plan, "applied": args.apply, **plan["summary"]}))


if __name__ == "__main__":
    main()
