"""Download, verify and convert ASQ-PHI to the project's frozen corpus format.

ASQ-PHI (Weatherhead, 2025; MIT licence; doi:10.17632/csz5dzp7nx.1) tags PHI
by value only, without offsets. Each value is aligned to every whole-token
occurrence in its query, because HIPAA Safe Harbor requires removing every
occurrence. Values that cannot be aligned exactly are aligned after apostrophe
normalisation, then case-insensitively; anything left over is recorded and the
document is kept with that value missing from the gold set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import urllib.request

from common import CORPORA, HERE, ROOT, digest_bytes, write_json, write_jsonl
from labels import ASQ_PHI_GOLD

sys.path.insert(0, str(ROOT / "src"))
from outis.types import ENTITY_TYPES  # noqa: E402


SOURCE = {
    "dataset": "ASQ-PHI",
    "doi": "10.17632/csz5dzp7nx.1",
    "licence": "MIT",
    "file": "data/synthetic_clinical_queries.txt",
    "url": "https://data.mendeley.com/public-files/datasets/csz5dzp7nx/files/"
           "96e2b100-c637-4b5d-8696-7b96bc1b6de0/file_downloaded",
    "sha256": "cf00e424b8d2347d019f9f34e2ad1510cb4d853605410f8314bef44df8021fc8",
    "documents": 1051,
    "tags": 2973,
}
DOWNLOAD = HERE / "downloads" / "asq_phi_synthetic_clinical_queries.txt"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})


def fetch() -> bytes:
    if DOWNLOAD.exists():
        payload = DOWNLOAD.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE["url"], timeout=60) as response:
            payload = response.read()
    if digest_bytes(payload) != SOURCE["sha256"]:
        raise RuntimeError("ASQ-PHI download differs from the pinned SHA-256")
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD.write_bytes(payload)
    return payload


def parse(text: str) -> list[tuple[str, list[dict[str, str]]]]:
    """Return (query, tags) pairs from the ===QUERY=== / ===PHI_TAGS=== format."""
    records = []
    for block in text.split("===QUERY===")[1:]:
        if "===PHI_TAGS===" not in block:
            raise ValueError("query block without a PHI_TAGS section")
        query, tag_block = block.split("===PHI_TAGS===", 1)
        query = query.strip()
        if not query:
            raise ValueError("empty query")
        tags = [json.loads(line) for line in tag_block.splitlines() if line.strip()]
        for tag in tags:
            if set(tag) != {"identifier_type", "value"} or not tag["value"].strip():
                raise ValueError(f"unexpected tag shape: {tag}")
        records.append((query, tags))
    return records


def _occurrences(text: str, value: str) -> tuple[list[tuple[int, int]], str]:
    """Find whole-token occurrences of value, trying stricter matches first."""
    attempts = (("exact", text, value),
                ("apostrophe", text.translate(_APOSTROPHES), value.translate(_APOSTROPHES)),
                ("casefold", text.translate(_APOSTROPHES).lower(), value.translate(_APOSTROPHES).lower()))
    for method, haystack, needle in attempts:
        needle = needle.strip()
        prefix = r"(?<![\w])" if needle[:1].isalnum() else ""
        suffix = r"(?![\w])" if needle[-1:].isalnum() else ""
        found = [(m.start(), m.end()) for m in re.finditer(prefix + re.escape(needle) + suffix, haystack)]
        if found:
            return found, method
    return [], "unaligned"


def align(query: str, tags: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Return gold spans and per-tag alignment records for one query."""
    candidates, records = [], []
    for tag in tags:
        source_label = tag["identifier_type"]
        if source_label not in ASQ_PHI_GOLD:
            raise KeyError(f"undeclared ASQ-PHI label {source_label!r}")
        found, method = _occurrences(query, tag["value"])
        records.append({"value": tag["value"], "label": source_label, "method": method, "occurrences": len(found)})
        candidates += [(start, end, ASQ_PHI_GOLD[source_label], source_label) for start, end in found]
    # Drop a candidate wholly inside a longer one (e.g. a year tagged separately
    # inside a full date). Partial overlaps are kept and counted.
    candidates = sorted(set(candidates), key=lambda c: (c[0], -(c[1] - c[0])))
    spans = []
    for start, end, label, source_label in candidates:
        if any(s["start"] <= start and end <= s["end"] and (s["start"], s["end"]) != (start, end) for s in spans):
            continue
        if any((s["start"], s["end"]) == (start, end) for s in spans):
            continue  # same text tagged twice under different labels: keep the first
        spans.append({"start": start, "end": end, "label": label, "source_label": source_label,
                      "text": query[start:end]})
    return spans, records


def convert(payload: bytes) -> tuple[list[dict], dict]:
    records = parse(payload.decode("utf-8"))
    rows, methods, partial_overlaps, multi = [], Counter(), 0, 0
    unaligned = []
    for index, (query, tags) in enumerate(records, 1):
        spans, alignment = align(query, tags)
        for record in alignment:
            methods[record["method"]] += 1
            multi += record["occurrences"] > 1
            if record["method"] == "unaligned":
                unaligned.append({"id": f"asq-{index:04d}", **record})
        ordered = sorted((s["start"], s["end"]) for s in spans)
        partial_overlaps += sum(b[0] < a[1] for a, b in zip(ordered, ordered[1:]))
        rows.append({"id": f"asq-{index:04d}", "text": query, "spans": spans,
                     "family": "phi" if tags else "hard_negative",
                     "source_tags": tags})
    if len(rows) != SOURCE["documents"] or sum(len(r["source_tags"]) for r in rows) != SOURCE["tags"]:
        raise RuntimeError("parsed ASQ-PHI counts differ from the published statistics")
    report = {"alignment_methods": dict(methods), "values_with_multiple_occurrences": multi,
              "partial_overlaps": partial_overlaps, "unaligned": unaligned}
    return rows, report


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    payload = fetch()
    rows, report = convert(payload)
    path = CORPORA / "asq_phi.jsonl"
    corpus = write_jsonl(path, rows)
    support = Counter(span["label"] for row in rows for span in row["spans"])
    gold_characters = sum(len({i for s in row["spans"] for i in range(s["start"], s["end"])}) for row in rows)
    total_characters = sum(len(row["text"]) for row in rows)
    manifest = {
        "benchmark": "asq_phi",
        "source": SOURCE,
        "sha256": digest_bytes(corpus),
        "rows": len(rows),
        "documents": len(rows),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "source_label_support": dict(Counter(t["identifier_type"] for r in rows for t in r["source_tags"])),
        "gold_spans": sum(support.values()),
        "characters": {"total": total_characters, "gold": gold_characters,
                       "non_gold": total_characters - gold_characters},
        "alignment": report,
        "converter_sha256": digest_bytes(Path(__file__).read_bytes()),
    }
    write_json(path.with_suffix(".jsonl.manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("rows", "gold_spans", "characters", "label_support")}
                     | {"alignment": {k: v for k, v in report.items() if k != "unaligned"},
                        "unaligned": len(report["unaligned"])}, indent=2))


if __name__ == "__main__":
    main()
