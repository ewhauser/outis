"""Convert the n2c2 (i2b2) 2014 de-identification XML files to the harness format.

The data is under the n2c2 data use agreement. Obtain it yourself through the
DBMI Data Portal, keep it outside every git repository, and point this script
at the extracted XML folder. The converted corpus is written under
``OUTIS_RESTRICTED_DIR`` (default ``~/working/deid/restricted``), never into the
repository. Only counts are printed.

Each file is ``<deIdi2b2><TEXT><![CDATA[...]]></TEXT><TAGS>...</TAGS></deIdi2b2>``;
every tag carries ``start``, ``end``, ``text`` and ``TYPE`` (the subtype, such as
PATIENT or HOSPITAL). Gold spans keep the native subtype in ``source_label``;
``label`` is the closest project type. PROFESSION has no project type and is
kept only in ``source_tags``.

  python n2c2_2014.py --split test --raw /path/to/testing-PHI-Gold-fixed
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from common import ROOT, corpus_path, digest_bytes, write_json, write_jsonl

sys.path.insert(0, str(ROOT / "src"))
from outis.types import ENTITY_TYPES  # noqa: E402

# Native subtype -> project label. None: no project type (kept in source_tags only).
TYPE_MAP = {
    "PATIENT": "NAME", "DOCTOR": "NAME", "USERNAME": "NAME",
    "PROFESSION": None,
    "HOSPITAL": "FACILITY", "ORGANIZATION": "FACILITY", "DEPARTMENT": "FACILITY",
    "ROOM": "ID",
    "STREET": "ADDRESS",
    "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION", "ZIP": "LOCATION",
    "LOCATION-OTHER": "LOCATION",
    "AGE": "AGE", "DATE": "DATE",
    "PHONE": "PHONE", "FAX": "PHONE", "EMAIL": "EMAIL", "URL": "URL", "IPADDR": "IP",
    "SSN": "SSN", "MEDICALRECORD": "MRN", "HEALTHPLAN": "ID", "ACCOUNT": "ACCOUNT",
    "LICENSE": "ID", "DEVICE": "ID", "IDNUM": "ID", "BIOID": "ID", "VEHICLE": "ID",
}
# Native subtypes that the project policy does not annotate in every case; the
# policy-aligned view treats them as neutral.
OUTSIDE_POLICY = frozenset({"PROFESSION", "STATE", "COUNTRY", "DEPARTMENT", "ORGANIZATION", "ROOM"})


def parse_file(payload: bytes) -> tuple[str, list[dict]]:
    root = ET.fromstring(payload)
    text_node = root.find("TEXT")
    if text_node is None or text_node.text is None:
        raise ValueError("missing TEXT element")
    tags = []
    tags_node = root.find("TAGS")
    for tag in [] if tags_node is None else list(tags_node):
        attrs = dict(tag.attrib)
        tags.append({"category": tag.tag, "type": attrs.get("TYPE", tag.tag), "start": int(attrs["start"]),
                     "end": int(attrs["end"]), "text": attrs.get("text", "")})
    return text_node.text, tags


def align(text: str, tag: dict) -> tuple[int, int, str] | None:
    """Return (start, end, method) or None. Prefer the given offsets; else the nearest exact match."""
    start, end, value = tag["start"], tag["end"], tag["text"]
    if text[start:end] == value:
        return start, end, "exact"
    if not value:
        return None
    hits, at = [], text.find(value)
    while at != -1:
        hits.append(at)
        at = text.find(value, at + 1)
    if not hits:
        return None
    best = min(hits, key=lambda h: abs(h - start))
    return best, best + len(value), "nearest"


def convert_files(files: list[Path]) -> tuple[list[dict], dict]:
    rows, methods, unaligned, overlaps, dropped_duplicates = [], Counter(), 0, 0, 0
    for path in sorted(files):
        text, tags = parse_file(path.read_bytes())
        spans, seen = [], set()
        for tag in tags:
            native = tag["type"].upper()
            if native not in TYPE_MAP:
                raise ValueError(f"unknown n2c2 subtype {native!r} in {path.name}")
            placed = align(text, tag)
            if placed is None:
                unaligned += 1
                continue
            start, end, method = placed
            methods[method] += 1
            label = TYPE_MAP[native]
            if label is None:
                continue
            if (start, end) in seen:
                dropped_duplicates += 1
                continue
            seen.add((start, end))
            spans.append({"start": start, "end": end, "label": label, "source_label": native,
                          "in_policy": native not in OUTSIDE_POLICY})
        spans.sort(key=lambda s: (s["start"], s["end"]))
        overlaps += sum(b["start"] < a["end"] for a, b in zip(spans, spans[1:]))
        rows.append({"id": path.stem, "text": text, "spans": spans,
                     "family": "phi" if spans else "hard_negative",
                     "source_tags": [{"type": t["type"], "start": t["start"], "end": t["end"]} for t in tags]})
    report = {"alignment_methods": dict(methods), "unaligned_tags": unaligned,
              "overlapping_spans": overlaps, "duplicate_offsets_dropped": dropped_duplicates}
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--raw", type=Path, required=True, help="folder of extracted n2c2 2014 XML files")
    args = parser.parse_args()
    files = sorted(args.raw.glob("*.xml"))
    if not files:
        raise SystemExit(f"no XML files in {args.raw}")
    benchmark = f"n2c2_2014_{args.split}"
    rows, report = convert_files(files)
    path = corpus_path(benchmark)
    corpus = write_jsonl(path, rows)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    gold_characters = sum(len({i for s in r["spans"] for i in range(s["start"], s["end"])}) for r in rows)
    total_characters = sum(len(r["text"]) for r in rows)
    manifest = {
        "benchmark": benchmark,
        "source": {"dataset": "n2c2 2014 de-identification (i2b2/UTHealth)", "split": args.split,
                   "files": len(files), "access": "n2c2 data use agreement via the DBMI Data Portal"},
        "sha256": digest_bytes(corpus), "rows": len(rows), "documents": len(rows),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "source_label_support": dict(Counter(t["type"].upper() for r in rows for t in r["source_tags"])),
        "gold_spans": sum(support.values()),
        "characters": {"total": total_characters, "gold": gold_characters,
                       "non_gold": total_characters - gold_characters},
        "alignment": report,
        "converter_sha256": digest_bytes(Path(__file__).read_bytes()),
    }
    write_json(path.with_suffix(".jsonl.manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("benchmark", "rows", "gold_spans", "label_support", "alignment")},
                     indent=2))


if __name__ == "__main__":
    main()
