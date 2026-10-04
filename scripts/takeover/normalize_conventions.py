"""Normalize training banks to the independent-reader convention census.

Census (reports/takeover/convention_census_001.json): 10/10 independent policy
readers include a preceding weekday in DATE, include a phone extension in PHONE,
and annotate clinician NPI numbers as ID. Training banks authored under the
earlier takeover addendum used the opposite choices, so this rewrites:
  1. DATE spans directly preceded by a weekday name (full or abbreviated,
     optionally followed by a comma) start at the weekday.
  2. PHONE spans directly followed by an extension ("ext. 23", "x23",
     "extension 23") end after the extension digits.
  3. Unannotated 10-digit numbers after an "NPI" caption become ID spans.

  python scripts/takeover/normalize_conventions.py data/tk_train_ed.jsonl ... --suffix census1
writes data/<name>__census1.jsonl plus a manifest, and prints change counts.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID", "URL", "IP",
          "AGE", "LOCATION", "FACILITY")
WEEKDAY = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day|(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\.?"
BEFORE_DATE = re.compile(r"\b(" + WEEKDAY + r"),?\s+$", re.I)
EXTENSION = re.compile(r"^(\s*,?\s*(?:ext\.?|extension|x)\s*\d{1,6})\b", re.I)
NPI = re.compile(r"\bNPI\b[\s:#.]*(\d{10})\b")


def normalize(row, counts):
    text = row["text"]
    spans = [dict(s) for s in row["spans"]]
    occupied = sorted((s["start"], s["end"]) for s in spans)

    def free(a, b, ignore=None):
        return all(e <= a or s >= b or (s, e) == ignore for s, e in occupied)

    for span in spans:
        if span["label"] == "DATE":
            window = text[max(0, span["start"] - 16):span["start"]]
            match = BEFORE_DATE.search(window)
            if match:
                new_start = span["start"] - (len(window) - match.start(1))
                if free(new_start, span["start"], ignore=(span["start"], span["end"])):
                    occupied.remove((span["start"], span["end"]))
                    span["start"] = new_start
                    occupied.append((span["start"], span["end"]))
                    counts["date_weekday_included"] += 1
        elif span["label"] == "PHONE":
            match = EXTENSION.match(text[span["end"]:span["end"] + 24])
            if match:
                new_end = span["end"] + len(match.group(1))
                if free(span["end"], new_end, ignore=(span["start"], span["end"])):
                    occupied.remove((span["start"], span["end"]))
                    span["end"] = new_end
                    occupied.append((span["start"], span["end"]))
                    counts["phone_extension_included"] += 1
    for match in NPI.finditer(text):
        a, b = match.start(1), match.end(1)
        if free(a, b):
            spans.append({"start": a, "end": b, "label": "ID"})
            occupied.append((a, b))
            counts["npi_id_added"] += 1
    spans.sort(key=lambda s: s["start"])
    for span in spans:
        span["text"] = text[span["start"]:span["end"]]
    out = dict(row)
    out["spans"] = spans
    out["negative_only"] = not spans
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpora", nargs="+")
    parser.add_argument("--suffix", default="census1")
    args = parser.parse_args()
    for corpus in args.corpora:
        path = Path(corpus)
        counts = Counter()
        rows = [normalize(json.loads(line), counts) for line in path.read_text().splitlines() if line.strip()]
        out = path.with_name(f"{path.stem}__{args.suffix}.jsonl")
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
        out.write_text(payload)
        support = Counter(s["label"] for r in rows for s in r["spans"])
        source_manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
        manifest = {k: v for k, v in source_manifest.items() if k not in ("sha256", "label_support", "entity_support", "label_counts")}
        manifest.update({
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "rows": len(rows), "documents": len(rows),
            "label_support": {label: support[label] for label in LABELS},
            "entity_support": {label: support[label] for label in LABELS},
            "label_counts": {label: support[label] for label in LABELS},
            "normalized_from": {"path": str(path), "sha256": source_manifest.get("sha256")},
            "normalization": {"script": "scripts/takeover/normalize_conventions.py", "census": "reports/takeover/convention_census_001.json",
                              "changes": dict(counts)},
        })
        out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        print(json.dumps({"corpus": str(path), "output": str(out), "changes": dict(counts)}))


if __name__ == "__main__":
    main()
