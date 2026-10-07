"""Second all-ages relabel pass: ages that ``normalize_ages.py`` (v1) missed.

v1 added an AGE span on ages found by age-context patterns. Two gaps remained:

* a sentence-final age was skipped: v1's ``age N`` and ``died at N`` patterns
  refused a match followed by a period, meant for decimals, so "at age 68."
  and "died at 72." stayed unannotated;
* ages without an age word: "she is 72", "my mother was 81", spoken
  "i'm eighty-four" (the v1 README listed these as not covered).

This pass runs on the ``__allages1`` derivatives and writes
``<stem>__allages2.jsonl``. It adds only ages v1 did not, keeps v1's public,
guideline and hypothetical exclusions (a hypothetical person's age is not a
person's age under the policy), and skips measurements ("is 72 kg",
"is 32 weeks"). It labels data only; it is never used at inference. Sealed,
blind, qualification and held-out banks are refused.

  python scripts/takeover/normalize_ages2.py --sample-out SAMPLE.md data/BANK__allages1.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

_V1_PATH = Path(__file__).with_name("normalize_ages.py")
_spec = importlib.util.spec_from_file_location("normalize_ages_v1", _V1_PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

SUFFIX = "allages2"
NUMBER, SPELLED = v1.NUMBER, v1.SPELLED
# A value is not part of a decimal or a measurement when followed by this.
_END = r"(?![\w%/-]|\.\d)"
RELATIVE = (
    r"(?:mother|father|mom|dad|mum|wife|husband|son|daughter|brother|sister|grandmother|"
    r"grandfather|grandma|grandpa|grandson|granddaughter|partner|spouse|baby|child|infant|"
    r"aunt|uncle|cousin|niece|nephew|stepmother|stepfather|stepson|stepdaughter)"
)
# Only people: never "her pulse was 72" or "they were 2".
PERSON = (
    rf"(?:i|she|he|patient|pt|the patient|(?:my|his|her|their|the patient's|patient's)\s+{RELATIVE}|"
    rf"(?:the\s+)?{RELATIVE})"
)
VERB = r"(?:'s|’s|'m|’m| is| was| am| are| were| turned| turns| will be| would be)"
QUALIFIER = r"(?:\s+(?:now|only|almost|nearly|about|just|already|over|under))?"
# Conditions and events written "<condition> at N" in family histories.
CONDITION = (
    r"(?:MI|myocardial infarction|stroke|CVA|cancer|carcinoma|lymphoma|leukemia|melanoma|"
    r"diabetes|CABG|heart attack|heart failure|cardiac death|arthritis|glaucoma|dementia|"
    r"parkinson(?:'s)?(?: disease)?|alzheimer(?:'s)?(?: disease)?|COPD|osteoporosis|"
    r"hypertension|aneurysm|ESRD|kidney failure)"
)
# Words that make the number a measurement or a count, not an age.
MEASURE = re.compile(
    r"\s*(?:%|percent|kg|kgs|lb|lbs|pounds|g\b|mg|mcg|ml|cm|mm|inches|ft|l\b|liters?|litres?|lpm|feet|bpm|"
    r"beats|mmhg|units?|points?|weeks?|wks?|days?|months?|mos?|hours?|hrs?|minutes?|mins?|"
    r"times|out of|of\b|/|x\b|mmol|mg/dl|°|degrees|f\b|c\b|th\b|st\b|nd\b|rd\b|am\b|pm\b|"
    r"a\.m\.|p\.m\.|o'clock|hundred)",
    re.I,
)
PATTERNS = [
    # v1 patterns whose trailing guard refused a sentence-final period.
    re.compile(rf"\b(?:aged|age|age:)\s*(?P<v>{NUMBER}|{SPELLED}){_END}", re.I),
    re.compile(
        rf"\b(?:died|passed away|deceased|diagnosed|turned)\s+(?:at\s+(?:age\s+)?)?"
        rf"(?P<v>{NUMBER}|{SPELLED}){_END}",
        re.I,
    ),
    re.compile(rf"\bat the age of\s+(?P<v>{NUMBER}|{SPELLED}){_END}", re.I),
    # A person word, a linking verb and a bare number: "she is 72", "i'm eighty-four".
    re.compile(rf"\b{PERSON}{VERB}{QUALIFIER}\s+(?P<v>{NUMBER}|{SPELLED}){_END}", re.I),
    # Family-history forms: "mother lived to 56", "age at diagnosis 36", "MI at 66".
    re.compile(rf"\blived to\s+(?P<v>{NUMBER}|{SPELLED}){_END}", re.I),
    re.compile(rf"\bage at (?:diagnosis|onset|death)[:,]?\s+(?P<v>{NUMBER}){_END}", re.I),
    re.compile(rf"\b{CONDITION}\s+at\s+(?P<v>{NUMBER}){_END}", re.I),
]
# Patterns whose number must not be followed by a unit or count word.
MEASURED = {3, 6}


def find_ages(text: str, spans: list[dict]) -> list[tuple[int, int]]:
    """Unannotated age values of 89 and under that v1 did not add, as (start, end)."""
    occupied = [(s["start"], s["end"]) for s in spans]
    found: dict[tuple[int, int], None] = {}
    for number, pattern in enumerate(PATTERNS):
        for match in pattern.finditer(text):
            start, end = match.span("v")
            if any(start < b and a < end for a, b in occupied):
                continue
            if number in MEASURED and MEASURE.match(text, end):
                continue
            value = v1._value(match.group("v"))
            if value is None or value <= 0 or value > 89:
                continue
            sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), start - 90)
            window = text[sentence_start + 1 : min(len(text), end + 40)]
            if v1.PUBLIC_CONTEXT.search(window):
                continue
            found[(start, end)] = None
    kept: list[tuple[int, int]] = []
    for start, end in sorted(found):
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


def derive(source: Path) -> tuple[Path, dict, list[tuple[dict, tuple[int, int]]]]:
    name = source.name.lower()
    if any(word in name for word in v1.PROTECTED):
        raise ValueError(f"refusing protected bank: {source.name}")
    if "__allages1" not in source.name:
        raise ValueError(f"run on an __allages1 derivative, not {source.name}")
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
    support = Counter(s["label"] for r in rows for s in r["spans"])
    manifest = dict(manifest)
    manifest["corpus"] = str(target)
    manifest["sha256"] = hashlib.sha256(out).hexdigest()
    for key in ("label_support", "entity_support", "label_counts"):
        if isinstance(manifest.get(key), dict):
            manifest[key] = {label: support.get(label, 0) for label in manifest[key]}
            manifest[key].setdefault("AGE", support.get("AGE", 0))
    for key in ("spans", "entity_count", "gold_spans"):
        if isinstance(manifest.get(key), int):
            manifest[key] = sum(support.values())
    manifest["age_relabel_2"] = {
        "version": 2,
        "rule": "AGE added on sentence-final cued ages and uncued personal ages of 89 and under",
        "script": "scripts/takeover/normalize_ages2.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "v1_script_sha256": hashlib.sha256(_V1_PATH.read_bytes()).hexdigest(),
        "source": {"path": str(source), "sha256": hashlib.sha256(payload).hexdigest()},
        "ages_added": len(additions),
        "rows_changed": len({id(r) for r, _ in additions}),
    }
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes(out)
    Path(str(target) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return target, manifest["age_relabel_2"], additions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", type=Path)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="count additions without writing files")
    args = parser.parse_args()
    pool = []
    for source in args.corpora:
        if args.dry_run:
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            additions = [(r, a) for r in rows for a in find_ages(r["text"], r["spans"])]
            record = {"ages_added": len(additions), "rows_changed": len({id(r) for r, _ in additions})}
            target = source
        else:
            target, record, additions = derive(source)
        pool += [(source.name, row, span) for row, span in additions]
        print(json.dumps({"bank": source.name, "output": target.name, **record}))
    if args.sample_out:
        rng = random.Random(20261007)
        lines = ["# Age relabel 2 spot check", "", f"{min(args.sample_size, len(pool))} random additions of "
                 f"{len(pool)}. The added AGE span is in [brackets]; check it is a person's age.", ""]
        for bank, row, (start, end) in rng.sample(pool, min(args.sample_size, len(pool))):
            text = row["text"]
            context = text[max(0, start - 70):start] + "[" + text[start:end] + "]" + text[end:end + 40]
            lines.append(f"- `{bank.split('__')[0]}`: " + " ".join(context.split()))
        args.sample_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
