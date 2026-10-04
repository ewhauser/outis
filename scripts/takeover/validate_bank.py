"""Structural validator for newly authored clinical-phi-v2 banks.

Checks the row schema, exact offsets, the frozen manifest, label support and
cohort mix, and reports aggregate diversity statistics. It never prints note
text unless --show-text is passed, so it can be run on held-out proxy banks
without exposing their content to the operator.

Usage:
  PYTHONPATH=src python scripts/takeover/validate_bank.py data/BANK.jsonl \
      --role proxy|train [--show-text] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

from clinical_deid.annotation_policy import COHORTS, POLICY_VERSION, annotation_errors  # noqa: E402
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

EXPLANATIONS = re.compile(
    r"\b(?:gold spans?|PHI|annotation policy|hard negative|training data|tokenizer|ownership join"
    r"|this is not phi|non-identifying|should (?:not )?be (?:tagged|labell?ed)|not a patient identifier"
    r"|no identified individual|remains? clinical information)\b",
    re.I,
)
REQUIRED = ("id", "text", "spans", "split", "suite", "annotation_policy", "cohort", "family",
            "template_family", "phenomena", "identity_group", "identity_groups", "pair_id")


def masked_frames(text, spans):
    pieces, previous = [], 0
    for span in sorted(spans, key=lambda s: s["start"]):
        pieces.append(text[previous:span["start"]])
        pieces.append(f"<{span['label']}>")
        previous = span["end"]
    pieces.append(text[previous:])
    masked = re.sub(r"\d", "9", "".join(pieces)).lower()
    masked = re.sub(r"[ \t]+", " ", masked)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", masked) if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--role", choices=("proxy", "train"), required=True)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--min-label-support", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.corpus)
    payload = path.read_bytes()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    problems = []
    manifest = {}
    if not manifest_path.exists():
        problems.append("manifest missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("sha256") != hashlib.sha256(payload).hexdigest():
            problems.append("manifest sha256 differs from corpus bytes")

    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    row_errors = []
    ids, texts = Counter(), Counter()
    support, cohorts, families, layouts = Counter(), Counter(), Counter(), Counter()
    frames, phi_frames, label_frames = set(), set(), {label: set() for label in ENTITY_TYPES}
    phi_free = 0
    token_lengths = []
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
    except Exception as error:  # pragma: no cover - environment dependent
        encoding = None
        problems.append(f"tiktoken unavailable: {type(error).__name__}")

    for index, row in enumerate(rows):
        rid = row.get("id", f"<row {index}>")
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            row_errors.append((rid, f"missing fields {missing}"))
        text, spans = row.get("text"), row.get("spans")
        for message in annotation_errors(text, spans):
            row_errors.append((rid, message))
        if not isinstance(text, str) or not isinstance(spans, list):
            continue
        ids[rid] += 1
        texts[text] += 1
        if row.get("annotation_policy") != POLICY_VERSION:
            row_errors.append((rid, "annotation_policy must be clinical-phi-v2"))
        if row.get("cohort") not in COHORTS:
            row_errors.append((rid, f"unknown cohort {row.get('cohort')!r}"))
        if row.get("family") != row.get("template_family"):
            row_errors.append((rid, "family must equal template_family"))
        expected_split = "validation" if args.role == "proxy" else "train"
        if row.get("split") != expected_split:
            row_errors.append((rid, f"split must be {expected_split!r}"))
        if "[[" in text or "]]" in text:
            row_errors.append((rid, "residual markup delimiters"))
        if EXPLANATIONS.search(text):
            row_errors.append((rid, "annotation explanation wording in note text"))
        for span in spans:
            if isinstance(span, dict) and "text" not in span:
                row_errors.append((rid, "span missing text field"))
                break
        cohorts[row.get("cohort")] += 1
        families[row.get("family")] += 1
        layouts[row.get("layout")] += 1
        if not spans:
            phi_free += 1
        for span in spans:
            if isinstance(span, dict) and span.get("label") in ENTITY_TYPES:
                support[span["label"]] += 1
        for frame in masked_frames(text, [s for s in spans if isinstance(s, dict) and "start" in s]):
            frames.add(frame)
            labels = set(re.findall(r"<([a-z]+)>", frame))
            if labels:
                phi_frames.add(frame)
                for label in labels:
                    if label.upper() in label_frames:
                        label_frames[label.upper()].add(frame)
        if encoding is not None:
            token_lengths.append(len(encoding.encode(text, disallowed_special=())))

    duplicate_ids = [key for key, count in ids.items() if count > 1]
    duplicate_texts = sum(1 for count in texts.values() if count > 1)
    if duplicate_ids:
        problems.append(f"{len(duplicate_ids)} duplicate ids")
    if duplicate_texts:
        problems.append(f"{duplicate_texts} duplicate texts")
    if manifest:
        for key in ("label_support", "entity_support", "label_counts"):
            declared = manifest.get(key)
            if declared is None:
                problems.append(f"manifest missing {key}")
            elif any(declared.get(label) != support[label] for label in ENTITY_TYPES):
                problems.append(f"manifest {key} differs from corpus support")
        for key in ("rows", "documents"):
            if manifest.get(key) != len(rows):
                problems.append(f"manifest {key} differs from row count")
        if manifest.get("annotation_policy") != POLICY_VERSION:
            problems.append("manifest annotation_policy must be clinical-phi-v2")
        if manifest.get("independent_final") is not False:
            problems.append("manifest independent_final must be false (development bank)")
    minimum = args.min_label_support if args.min_label_support is not None else (400 if args.role == "proxy" else 0)
    low = {label: support[label] for label in ENTITY_TYPES if support[label] < minimum}
    if low:
        problems.append(f"labels below {minimum}: {low}")
    long_over_512 = sum(1 for n in token_lengths if n > 512)

    summary = {
        "corpus": str(path),
        "rows": len(rows),
        "phi_free_rows": phi_free,
        "spans": sum(support.values()),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "cohorts": dict(cohorts),
        "families": len(families),
        "layouts": dict(layouts),
        "distinct_frames": len(frames),
        "distinct_phi_frames": len(phi_frames),
        "distinct_frames_per_label": {label: len(label_frames[label]) for label in ENTITY_TYPES},
        "token_length": {
            "median": statistics.median(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
            "over_512": long_over_512,
        },
        "row_error_count": len(row_errors),
        "row_errors_sample": [(rid, msg) for rid, msg in row_errors[:25]],
        "problems": problems,
        "passed": not problems and not row_errors,
    }
    if args.show_text:
        summary["examples"] = [
            {"id": row.get("id"), "text": row.get("text")[:600],
             "spans": [(s["label"], row["text"][s["start"]:s["end"]]) for s in row.get("spans", [])]}
            for row in rows[:3]
        ]
    output = json.dumps(summary, indent=1, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(output + "\n")
    print(output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
