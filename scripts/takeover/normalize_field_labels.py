"""Write training derivatives with identifier field labels moved outside MRN spans.

The convention census (docs/annotation-policy-v2.md) keeps a field caption such
as ``MRN:`` outside the span and an authored prefix such as ``QX-`` inside it.
Some training banks put the caption inside on part of their rows
(``MRN: RHB-71001``). This trims a leading ``MRN``, ``MRN:`` or ``MRN#`` from
MRN spans; the masked identifier value is unchanged.

Only training corpora are accepted. Outputs are new files beside the source,
named ``<source>__<suffix>.jsonl`` with an updated manifest.

  python scripts/takeover/normalize_field_labels.py data/tk3_train_rehab__...jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

CAPTION = re.compile(r"MRN(?:\s*[:#]\s*|\s+)(?=\S)")
PROTECTED = (
    "blind_",
    "qualification",
    ".sealed",
    "proxy",
    "retired",
    "validation",
    "development",
    "heldout_",
    "_dev",
    "test",
)


def trim(text: str, span: dict) -> dict | None:
    """Return the span without a leading caption, or None if it has none."""
    if span["label"] != "MRN":
        return None
    match = CAPTION.match(text, span["start"], span["end"])
    if not match or match.end() >= span["end"]:
        return None
    out = dict(span, start=match.end())
    if "text" in out:
        out["text"] = text[out["start"] : out["end"]]
    return out


def normalize_row(row: dict) -> tuple[dict, int]:
    if row.get("split") != "train":
        raise ValueError(f"refusing non-training row {row.get('id')}")
    output, changed = deepcopy(row), 0
    for i, span in enumerate(row["spans"]):
        new = trim(row["text"], span)
        if new is not None:
            output["spans"][i] = new
            changed += 1
    return output, changed


def derive(source: Path, suffix: str) -> dict:
    if (
        any(word in source.name.lower() for word in PROTECTED)
        or "_train" not in source.name
    ):
        raise ValueError(f"refusing non-training corpus: {source.name}")
    target = source.with_name(f"{source.stem}__{suffix}.jsonl")
    manifest_target = target.with_suffix(".jsonl.manifest.json")
    if target.exists() or manifest_target.exists():
        raise FileExistsError(target)
    source_bytes = source.read_bytes()
    manifest = json.loads(source.with_suffix(".jsonl.manifest.json").read_text())
    if manifest.get("sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError(f"{source.name} differs from its manifest")
    rows, changed, by_row = [], 0, Counter()
    for line in source_bytes.decode("utf-8").splitlines():
        if line.strip():
            row, n = normalize_row(json.loads(line))
            rows.append(row)
            changed += n
            by_row[n > 0] += 1
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    manifest = dict(manifest)
    manifest["corpus"] = str(target)
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["field_label_normalization"] = {
        "version": 1,
        "rule": "Leading 'MRN', 'MRN:' or 'MRN#' caption moved outside MRN spans (convention census).",
        "script": "scripts/takeover/normalize_field_labels.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {
            "path": str(source),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "spans_trimmed": changed,
        "rows_changed": by_row[True],
    }
    target.write_bytes(payload)
    manifest_target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "source": source.name,
        "output": target.name,
        "spans_trimmed": changed,
        "rows_changed": by_row[True],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", type=Path)
    parser.add_argument("--suffix", default="fieldlabel1")
    args = parser.parse_args()
    for source in args.corpora:
        print(json.dumps(derive(source, args.suffix)))


if __name__ == "__main__":
    main()
