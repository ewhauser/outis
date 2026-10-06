"""Convert two public synthetic de-identification benchmarks to the harness format.

* ``meddeid_en``: the MedDeID English synthetic benchmark (300 notes, CC BY
  4.0). Its terms allow final evaluation only: never train, tune or select a
  model on it.
* ``technetium_test5k``: a fixed random sample of 5,000 notes from the
  Technetium-I test split (74,700 notes, EUPL-1.2); ``technetium_test`` is the
  full split.

Both files are pinned by revision and SHA-256 and downloaded into the
gitignored ``downloads/`` folder. Gold spans keep the native label in
``source_label``; ``label`` is the closest project type; ``in_policy`` is False
for native categories the project policy does not annotate.

  python synthetic_benchmarks.py --benchmark meddeid_en
  python synthetic_benchmarks.py --benchmark technetium_test5k
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
import urllib.request

from common import HERE, ROOT, corpus_path, digest, digest_bytes, write_json, write_jsonl

sys.path.insert(0, str(ROOT / "src"))
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

SOURCES = {
    "meddeid_en": {
        "dataset": "MedDeID English synthetic benchmark", "licence": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark/resolve/"
               "288c5718e5112e75c857ddf511a30e6eb6b32cdb/data/test.jsonl",
        "sha256": "c6089f2e36cbca90571941ce3a7f29954aea14e1c6487cb950c157999589be7e",
        "file": "meddeid_english_synthetic_benchmark_test.jsonl",
        "terms": "final evaluation only; no training, validation, tuning or model selection",
    },
    "technetium": {
        "dataset": "Technetium-I test split", "licence": "EUPL-1.2",
        "url": "https://huggingface.co/datasets/temlm-foundation/Technetium-I/resolve/"
               "b6ffde5e2b43395b197f4a49ba01d1914394a6cb/test.jsonl",
        "sha256": "673e5c746136f16de1dadc2b9926264b227a3b16e964b3788e3cec7c36ff7c41",
        "file": "technetium_i_test.jsonl",
    },
}
SAMPLE_SEED, SAMPLE_SIZE = 20261006, 5000
DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
                       r"|\b(?:19|20)\d\d\b", re.I)
TECHNETIUM_MAP = {"NAME": "NAME", "DATE": "DATE", "LOCATION": "LOCATION", "AGE": "AGE", "ID": "ID",
                  "PHONE": "PHONE", "EMAIL": "EMAIL", "HOSPITAL": "FACILITY", "DEVICE": "ID", "CONTACT": "PHONE"}


def fetch(key: str) -> Path:
    source = SOURCES[key]
    path = HERE / "downloads" / source["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source["url"], timeout=600) as response:
            path.write_bytes(response.read())
    if digest(path) != source["sha256"]:
        raise RuntimeError(f"{path.name} differs from the pinned SHA-256")
    return path


def meddeid_label(native: str, value: str) -> tuple[str | None, bool]:
    """Map a MedDeID label to (project label, in_policy)."""
    category = native.split(":")[0]
    if category == "Name":
        return "NAME", True
    if category == "ID":
        return "ID", True
    if category == "Date":
        return "DATE", True
    if category == "Age_Birthdate":
        return ("DATE", True) if DATE_LIKE.search(value) else ("AGE", True)
    if native == "Organization:Healthcare":
        return "FACILITY", True
    if category == "Organization":
        return "FACILITY", False
    if category == "Address_Location":
        return ("ADDRESS", True) if re.search(r"\d", value) and re.search(r"[A-Za-z]{3}", value) else ("LOCATION", True)
    if category == "Contactdetails":
        if "@" in value:
            return "EMAIL", True
        if re.search(r"(?i)https?:|www\.", value):
            return "URL", True
        return "PHONE", True
    if category == "Profession":
        return None, False
    raise ValueError(f"unknown MedDeID label {native!r}")


def convert_meddeid(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        text, spans, seen = doc["text"], [], set()
        for span in doc["spans"]:
            label, in_policy = meddeid_label(span["label"], span["text"])
            if label is None or (span["begin"], span["end"]) in seen:
                continue
            if text[span["begin"]:span["end"]] != span["text"]:
                raise ValueError(f"offset mismatch in {doc['document_id']}")
            seen.add((span["begin"], span["end"]))
            spans.append({"start": span["begin"], "end": span["end"], "label": label,
                          "source_label": span["label"], "in_policy": in_policy})
        spans.sort(key=lambda s: (s["start"], s["end"]))
        rows.append({"id": doc["document_id"], "text": text, "spans": spans,
                     "family": "phi" if spans else "hard_negative",
                     "source_tags": [{"label": s["label"], "start": s["begin"], "end": s["end"]} for s in doc["spans"]]})
    return rows


def convert_technetium(path: Path, sample: bool) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        docs = [json.loads(line) for line in handle if line.strip()]
    if sample:
        keep = set(random.Random(SAMPLE_SEED).sample(range(len(docs)), SAMPLE_SIZE))
        docs = [d for i, d in enumerate(docs) if i in keep]
    rows = []
    for doc in docs:
        text, spans, seen = doc["text"], [], set()
        for a in doc["phi_annotations"]:
            if text[a["start"]:a["end"]] != a["text"]:
                raise ValueError(f"offset mismatch in {doc['note_id']}")
            if (a["start"], a["end"]) in seen:
                continue
            seen.add((a["start"], a["end"]))
            spans.append({"start": a["start"], "end": a["end"], "label": TECHNETIUM_MAP[a["entity_type"]],
                          "source_label": a["entity_type"], "in_policy": True})
        spans = resolve_technetium_overlaps(spans)
        rows.append({"id": doc["note_id"], "text": text, "spans": spans,
                     "family": "phi" if spans else "hard_negative",
                     "source_tags": [{"label": a["entity_type"], "start": a["start"], "end": a["end"]}
                                     for a in doc["phi_annotations"]]})
    return rows


def resolve_technetium_overlaps(spans: list[dict]) -> list[dict]:
    """Merge overlapping NAME spans (name parts are annotated with and without the full name)
    and drop AGE spans that overlap a span of another type (spurious matches inside phones, dates and IDs)."""
    others = [s for s in spans if s["source_label"] != "AGE"]
    ages = [s for s in spans if s["source_label"] == "AGE"
            and not any(o["start"] < s["end"] and s["start"] < o["end"] for o in others)]
    merged: list[dict] = []
    for span in sorted(others + ages, key=lambda s: (s["start"], -s["end"])):
        last = merged[-1] if merged else None
        if last and span["start"] < last["end"]:
            if last["label"] == span["label"] == "NAME" or span["end"] <= last["end"]:
                last["end"] = max(last["end"], span["end"])
                continue
            raise ValueError("unresolved crossing gold spans of different types")
        merged.append(dict(span))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("meddeid_en", "technetium_test5k", "technetium_test"), required=True)
    args = parser.parse_args()
    key = "meddeid_en" if args.benchmark == "meddeid_en" else "technetium"
    raw = fetch(key)
    rows = convert_meddeid(raw) if key == "meddeid_en" else convert_technetium(raw, args.benchmark == "technetium_test5k")
    path = corpus_path(args.benchmark)
    corpus = write_jsonl(path, rows)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    gold_characters = sum(len({i for s in r["spans"] for i in range(s["start"], s["end"])}) for r in rows)
    total_characters = sum(len(r["text"]) for r in rows)
    overlaps = sum(b["start"] < a["end"] for r in rows for a, b in zip(r["spans"], r["spans"][1:]))
    manifest = {
        "benchmark": args.benchmark, "source": SOURCES[key] | {"sha256_verified": True},
        "sample": {"seed": SAMPLE_SEED, "size": SAMPLE_SIZE} if args.benchmark == "technetium_test5k" else None,
        "overlap_resolution": "Technetium-I: overlapping NAME spans merged; AGE spans overlapping another type dropped" if key == "technetium" else None,
        "sha256": digest_bytes(corpus), "rows": len(rows), "documents": len(rows),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "source_label_support": dict(Counter(t["label"] for r in rows for t in r["source_tags"])),
        "gold_spans": sum(support.values()), "overlapping_spans": overlaps,
        "characters": {"total": total_characters, "gold": gold_characters, "non_gold": total_characters - gold_characters},
        "converter_sha256": digest_bytes(Path(__file__).read_bytes()),
    }
    write_json(path.with_suffix(".jsonl.manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("benchmark", "rows", "gold_spans", "overlapping_spans", "label_support")},
                     indent=2))


if __name__ == "__main__":
    main()
