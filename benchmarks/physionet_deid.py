"""Convert PhysioNet "Deidentified Medical Text" v1.0 to the harness format.

The corpus is 2,434 MIMIC-II nursing notes under the PhysioNet Credentialed
Health Data License 1.5.0: research use only, no sharing of access. Obtain it
yourself from https://physionet.org/content/deidentifiedmedicaltext/1.0/, keep
it outside every git repository, and point this script at the extracted folder.
The converted corpus is written under ``OUTIS_RESTRICTED_DIR`` (default
``~/working/deid/restricted``), never into the repository. Only counts are
printed.

``id.text`` holds the notes with every gold PHI instance replaced by a realistic
surrogate; ``id.res`` holds the same notes with each instance replaced by a
``[**category**]`` tag instead. Records are delimited by
``START_OF_RECORD=<patient>||||<note>||||`` and ``||||END_OF_RECORD``. Gold
spans are recovered by aligning the two: the text between ``id.res``'s literal
segments is the surrogate for each tag. Where two tags are separated only by a
short literal that also occurs inside a surrogate (a space between first and
last name), the boundary between them may be misplaced, but the gold
characters as a whole are exact.

  python physionet_deid.py --raw /path/to/deidentified-medical-text-1.0
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from common import ROOT, corpus_path, digest, digest_bytes, write_json, write_jsonl

sys.path.insert(0, str(ROOT / "src"))
from outis.types import ENTITY_TYPES  # noqa: E402

BENCHMARK = "physionet_deid"
FILES = ("id.text", "id.res")
_RECORD = re.compile(r"START_OF_RECORD=(\d+)\|\|\|\|(\d+)\|\|\|\|(.*?)\|\|\|\|END_OF_RECORD", re.S)
_TAG = re.compile(r"\[\*\*(.*?)\*\*\]", re.S)
# Tag category (trailing identifier digits removed) -> (project label, in policy).
# Checked in order; the first match wins. Ward names are units inside a
# hospital, which the policy, like n2c2's DEPARTMENT, does not always annotate.
CATEGORY_RULES = (
    (r"\d{1,4}(?:[-/]\d{1,4}){0,2}|Date range.*|Holiday.*|Month.*|Year.*|Day.*", "DATE", True),
    (r"Age over.*", "AGE", True),
    (r"(?:Telephone|Fax|Pager).*", "PHONE", True),
    (r"(?:Street|Apartment) Address.*", "ADDRESS", True),
    (r"E-?mail.*", "EMAIL", True),
    (r"URL.*", "URL", True),
    (r"Medical Record Number.*", "MRN", True),
    (r"Social Security Number.*", "SSN", True),
    (r"(?:Numeric Identifier|Serial Number|Unit Number|Job Number|Provider Number|Clip Number).*", "ID", True),
    (r"(?:State|Country).*", "LOCATION", False),
    (r"Wardname.*", "FACILITY", False),
    (r"Location.*", "LOCATION", True),
    (r"(?:Hospital|Company|University|College|Clinic).*", "FACILITY", True),
    (r"(?:.*Name.*|Initials.*|Name Initial.*|Known (?:patient )?(?:first|last)name.*|Doctor.*)", "NAME", True),
)


def category(inner: str) -> str:
    """Tag category without the per-instance number: '[**Hospital1 18**]' -> 'Hospital1'."""
    inner = inner.strip()
    if re.fullmatch(r"\d{1,4}(?:[-/]\d{1,4}){0,2}", inner):
        return inner
    return re.sub(r"\s+\d+$", "", inner)


def classify(inner: str) -> tuple[str, bool, str]:
    """(project label, in policy, source label) for one tag. Unknown categories are an error."""
    cat = category(inner)
    if not cat:
        # One release tag carries no category; its type is unknown, so it is neutral.
        return "ID", False, "UNSPECIFIED"
    for pattern, label, in_policy in CATEGORY_RULES:
        if re.fullmatch(pattern, cat):
            source = "DATE" if label == "DATE" and re.fullmatch(r"[\d/-]+", cat) else re.sub(r"\d", "", cat).strip()
            return label, in_policy, source
    raise ValueError(f"unknown PhysioNet tag category with shape {re.sub(r'[0-9]', '9', cat)!r}")


def records(payload: str) -> dict[tuple[str, str], str]:
    found = {(m.group(1), m.group(2)): m.group(3) for m in _RECORD.finditer(payload)}
    if len(found) != payload.count("START_OF_RECORD"):
        raise ValueError("duplicate or malformed record headers")
    return found


def align(text: str, tagged: str) -> list[tuple[int, int, str]] | None:
    """(start, end, tag inner) of each surrogate in text, or None when the two do not align."""
    parts = _TAG.split(tagged)
    literals, inners = parts[0::2], parts[1::2]
    if not inners:
        return [] if text == tagged else None
    pattern = "(.+?)".join(re.escape(lit) for lit in literals)
    match = re.fullmatch(pattern, text, re.S)
    if match is None:
        return None
    spans = []
    for group, inner in enumerate(inners, start=1):
        start, end = match.span(group)
        value = text[start:end]
        start += len(value) - len(value.lstrip())
        end -= len(value) - len(value.rstrip())
        if end > start:
            spans.append((start, end, inner))
    return spans


def convert(text_payload: str, res_payload: str) -> tuple[list[dict], dict]:
    texts, tagged = records(text_payload), records(res_payload)
    if texts.keys() != tagged.keys():
        raise ValueError("id.text and id.res hold different records")
    rows, unaligned, empty, lengths = [], 0, 0, Counter()
    for key in sorted(texts, key=lambda k: (int(k[0]), int(k[1]))):
        text = texts[key]
        placed = align(text, tagged[key])
        if placed is None:
            unaligned += 1
            continue
        empty += len(_TAG.findall(tagged[key])) - len(placed)
        spans = []
        for start, end, inner in placed:
            label, in_policy, source = classify(inner)
            lengths["over_40_chars" if end - start > 40 else "up_to_40_chars"] += 1
            spans.append({"start": start, "end": end, "label": label, "source_label": source,
                          "in_policy": in_policy})
        rows.append({"id": f"{key[0]}-{key[1]}", "text": text, "spans": spans,
                     "family": "phi" if spans else "hard_negative"})
    overlaps = sum(b["start"] < a["end"] for r in rows for a, b in zip(r["spans"], r["spans"][1:]))
    report = {"records": len(texts), "unaligned_records": unaligned, "empty_surrogates": empty,
              "overlapping_spans": overlaps, "surrogate_lengths": dict(lengths)}
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="folder holding id.text and id.res")
    args = parser.parse_args()
    payloads = {name: (args.raw / name).read_text(encoding="utf-8") for name in FILES}
    rows, report = convert(payloads["id.text"], payloads["id.res"])
    if report["unaligned_records"]:
        raise SystemExit(f"{report['unaligned_records']} records did not align; nothing written")
    path = corpus_path(BENCHMARK)
    corpus = write_jsonl(path, rows)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    gold_characters = sum(len({i for s in r["spans"] for i in range(s["start"], s["end"])}) for r in rows)
    total_characters = sum(len(r["text"]) for r in rows)
    manifest = {
        "benchmark": BENCHMARK,
        "source": {"dataset": "PhysioNet Deidentified Medical Text v1.0", "doi": "10.13026/C20M3F",
                   "files": {name: digest(args.raw / name) for name in FILES},
                   "access": "PhysioNet Credentialed Health Data License 1.5.0"},
        "sha256": digest_bytes(corpus), "rows": len(rows), "documents": len(rows),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "source_label_support": dict(Counter(s["source_label"] for r in rows for s in r["spans"]).most_common()),
        "out_of_policy_spans": sum(not s["in_policy"] for r in rows for s in r["spans"]),
        "gold_spans": sum(support.values()),
        "hard_negatives": sum(r["family"] == "hard_negative" for r in rows),
        "characters": {"total": total_characters, "gold": gold_characters,
                       "non_gold": total_characters - gold_characters},
        "alignment": report,
        "converter_sha256": digest_bytes(Path(__file__).read_bytes()),
    }
    write_json(path.with_suffix(".jsonl.manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("benchmark", "rows", "gold_spans", "hard_negatives",
                                               "label_support", "out_of_policy_spans", "alignment")}, indent=2))


if __name__ == "__main__":
    main()
