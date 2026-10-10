"""Write derivatives of training and development banks with every age annotated.

The annotation policy changed on 2026-10-06: AGE covers every age of the
patient or a relative or household member, not only ages over 89. Banks
written before that leave ages of 89 and under unannotated. This script adds an
AGE span on each such age value, found by a fixed set of age-context patterns,
when the value overlaps no existing span and is not a public or guideline age.
It labels data only; it is never used at inference.

Outputs are new files ``<source stem>__allages1.jsonl`` with an updated
manifest. Sealed, blind, qualification and held-out banks are refused.

  python scripts/takeover/normalize_ages.py --sample-out SAMPLE.md data/BANK.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

SUFFIX = "allages1"
PROTECTED = ("blind_", "qualification", ".sealed", "proxy", "heldout_dev")
NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety"
)
SPELLED = rf"(?:{NUMBER_WORDS})(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
NUMBER = r"\d{1,3}"
YEARS_OLD = r"\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b"
INFANT_OLD = r"\s*-?\s*(?:months?|mos?|weeks?|wks?|days?)\s*-?\s*old\b"
PATTERNS = [
    re.compile(rf"(?<![\w./-])(?P<v>{NUMBER}|{SPELLED})(?={YEARS_OLD})", re.I),
    re.compile(rf"(?<![\w./-])(?P<v>{NUMBER}|{SPELLED})(?={INFANT_OLD})", re.I),
    re.compile(rf"(?<![\w./-])(?P<v>{NUMBER}|{SPELLED})(?=\s+years?\s+of\s+age\b)", re.I),
    re.compile(rf"(?<![\w./-])(?P<v>{NUMBER})(?=\s*(?:yo|y/o|y\.o\.)(?![\w]))", re.I),
    re.compile(rf"(?<![\w./-])(?P<v>{NUMBER})(?=[MF](?![\w]))"),
    re.compile(rf"\b(?:aged|age|age:)\s*(?P<v>{NUMBER}|{SPELLED})(?![\w.%/-])", re.I),
    re.compile(
        rf"\b(?:died|passed away|deceased|diagnosed|turned)\s+(?:at\s+(?:age\s+)?)?(?P<v>{NUMBER}|{SPELLED})(?![\w.%/-])",
        re.I,
    ),
]
# Public, guideline or population ages, ranges and thresholds are not a person's age.
PUBLIC_CONTEXT = re.compile(
    r"screen|guideline|recommend|uspstf|eligib|criteria|threshold|population|cohort|"
    r"study|trial|starting at|begin|from age|over age|under age|older than|younger than|"
    r"at least|and older|or older|and up|aged\s+\d+\s*(?:to|-|–)\s*\d+|ages\s+\d|"
    r"hypothetical|comparison exercise|for example|in general terms|general discussion",
    re.I,
)


def _value(word: str) -> float | None:
    from outis.policy import age_value

    return age_value(word)


def find_ages(text: str, spans: list[dict]) -> list[tuple[int, int]]:
    """Unannotated age values of 89 and under, as (start, end)."""
    occupied = [(s["start"], s["end"]) for s in spans]
    found: dict[tuple[int, int], None] = {}
    for pattern in PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span("v")
            if any(start < b and a < end for a, b in occupied):
                continue
            value = _value(match.group("v"))
            if value is None or value <= 0 or value > 89:
                continue
            sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), start - 90)
            window = text[sentence_start + 1 : min(len(text), end + 40)]
            if PUBLIC_CONTEXT.search(window):
                continue
            found[(start, end)] = None
    ordered = sorted(found)
    kept: list[tuple[int, int]] = []
    for start, end in ordered:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end))
    return kept


def relabel_row(row: dict) -> tuple[dict, list[tuple[int, int]]]:
    added = find_ages(row["text"], row["spans"])
    if not added:
        return row, []
    out = deepcopy(row)
    template = next((s for s in row["spans"]), None)
    for start, end in added:
        span = {"start": start, "end": end, "label": "AGE"}
        if template is not None and "text" in template:
            span["text"] = row["text"][start:end]
        out["spans"].append(span)
    out["spans"].sort(key=lambda s: (s["start"], s["end"]))
    return out, added


def _support(rows: list[dict]) -> Counter:
    return Counter(s["label"] for r in rows for s in r["spans"])


def derive(source: Path) -> tuple[Path, dict, list[tuple[dict, tuple[int, int]]]]:
    name = source.name.lower()
    if any(word in name for word in PROTECTED):
        raise ValueError(f"refusing protected bank: {source.name}")
    target = source.with_name(f"{source.stem}__{SUFFIX}.jsonl")
    manifest_path = Path(str(source) + ".manifest.json")
    payload = source.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{source.name} differs from its manifest")
    rows, additions = [], []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row, added = relabel_row(json.loads(line))
        rows.append(row)
        additions += [(row, a) for a in added]
    out = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    support = _support(rows)
    manifest = dict(manifest)
    manifest["corpus"] = str(target)
    manifest["sha256"] = hashlib.sha256(out).hexdigest()
    for key in ("label_support", "entity_support", "label_counts"):
        if isinstance(manifest.get(key), dict):
            manifest[key] = {label: support.get(label, 0) for label in manifest[key]}
            if "AGE" not in manifest[key]:
                manifest[key]["AGE"] = support.get("AGE", 0)
    for key in ("spans", "entity_count", "gold_spans"):
        if isinstance(manifest.get(key), int):
            manifest[key] = sum(support.values())
    manifest["age_relabel"] = {
        "version": 1,
        "rule": "AGE added on every unannotated age of 89 and under (policy change 2026-10-06)",
        "script": "scripts/takeover/normalize_ages.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"path": str(source), "sha256": hashlib.sha256(payload).hexdigest()},
        "ages_added": len(additions),
        "rows_changed": len({id(r) for r, _ in additions}),
    }
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes(out)
    Path(str(target) + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return target, manifest["age_relabel"], additions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", type=Path)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    pool = []
    for source in args.corpora:
        target, record, additions = derive(source)
        pool += [(source.name, row, span) for row, span in additions]
        print(json.dumps({"bank": source.name, "output": target.name,
                          "ages_added": record["ages_added"], "rows_changed": record["rows_changed"]}))
    if args.sample_out:
        rng = random.Random(20261006)
        lines = ["# Age relabel spot check", "", f"{min(args.sample_size, len(pool))} random additions of "
                 f"{len(pool)}. The added AGE span is in [brackets]; check it is a person's age.", ""]
        for bank, row, (start, end) in rng.sample(pool, min(args.sample_size, len(pool))):
            text = row["text"]
            context = (text[max(0, start - 70):start] + "[" + text[start:end] + "]" + text[end:end + 40])
            lines.append(f"- `{bank.split('__')[0]}`: " + " ".join(context.split()))
        args.sample_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
