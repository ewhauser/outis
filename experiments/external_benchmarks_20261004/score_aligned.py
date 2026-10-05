"""Score stored predictions with benchmark-versus-policy differences neutralised.

Run with the project virtual environment. ``score.py`` measures every system
against the benchmark's own gold. A benchmark's gold also encodes its own
requirements: ASQ-PHI includes honorifics and field labels in spans, annotates
relative dates, and leaves bare years and ages unannotated. Where the
benchmark and the project policy (docs/annotation-policy-v2.md) disagree about
whether text must be masked, this scorer makes those characters *neutral*:
they count neither as PHI to mask nor as over-redaction, for every system.

Neutral characters are derived from the text and the gold spans alone, never
from predictions:

* every character that is not a letter or digit (a leaked space, comma or
  "#" reveals nothing);
* NAME: leading honorifics and trailing credentials (policy: NAME excludes
  honorifics and credentials);
* identifiers (MRN, SSN, ACCOUNT, ID): the field label before the first token
  that contains a digit ("Patient ID:", "record");
* DATE: relative expressions ("last week", "last Friday"), and the relative
  word in "last December" (policy: relative expressions are not annotated);
* LOCATION: lowercase words and generic unit words inside the span ("our
  Chicago clinic", "Mayo Clinic in Rochester"), since the policy keeps
  prepositions and generic care-site words outside place spans;
* outside gold: explicit ages up to 89 (``score.age_characters``) and bare
  four-digit years. Safe Harbor permits both; the project policy masks a
  patient-specific year and the benchmark never annotates one.

Writes ``results/<benchmark>.aligned.json`` and ``.aligned.md`` and leaves the
as-published results untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from common import RESULTS, ROOT, corpus_path, digest, write_json
from labels import COARSE, project_label
from score import age_characters, load_predictions, merge_names

sys.path.insert(0, str(ROOT / "src"))
from clinical_deid.evaluation.evaluate import (  # noqa: E402
    DIRECT_IDENTIFIERS,
    INDIRECT_IDENTIFIERS,
    cluster_lower,
    load_verified_corpus,
    wilson_lower,
)

IDENTIFIERS = {"MRN", "SSN", "ACCOUNT", "ID"}
_HONORIFIC = re.compile(r"^(?:(?:Dr|Mr|Mrs|Ms|Miss|Mx|Prof|Rev)\.?\s+)+", re.I)
_CREDENTIAL = re.compile(r"(?:,?\s+(?:MD|DO|RN|NP|PA-C|PhD|DDS|FACP))+\.?$")
_RELATIVE_LEAD = re.compile(r"^(?:last|next|this|past|previous|early|late)\s+", re.I)
_RELATIVE_UNIT = re.compile(
    r"^(?:week|weekend|month|year|night|morning|evening|spring|summer|fall|autumn|winter"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?$"
    r"|^(?:yesterday|today|tomorrow|recently)$"
    r"|^(?:a|an|one|two|three|four|five|six|several|few|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago$",
    re.I,
)
_GENERIC_UNITS = {"ER", "ED", "ICU", "OR", "PACU", "NICU"}
_BARE_YEAR = re.compile(r"(?<![\w/.-])(?:19|20)\d{2}(?![\w/.-]|-\d)")


def _value(text: str, start: int, end: int) -> set[int]:
    return {i for i in range(start, end) if text[i].isalnum()}


def neutral_in_span(text: str, start: int, end: int, label: str) -> set[int]:
    """Characters of one gold span that the project policy would leave unmasked."""
    value = text[start:end]
    neutral: set[int] = set()
    if label == "NAME":
        lead = _HONORIFIC.match(value)
        if lead:
            neutral.update(range(start, start + lead.end()))
        tail = _CREDENTIAL.search(value)
        if tail:
            neutral.update(range(start + tail.start(), end))
    elif label in IDENTIFIERS:
        token = next((m for m in re.finditer(r"\S+", value) if any(c.isdigit() for c in m.group())), None)
        if token:
            neutral.update(range(start, start + token.start()))
    elif label == "DATE":
        lead = _RELATIVE_LEAD.match(value)
        rest = value[lead.end():] if lead else value
        if lead:
            neutral.update(range(start, start + lead.end()))
        if _RELATIVE_UNIT.match(rest.strip()):
            neutral.update(range(start, end))
    elif label == "LOCATION":
        for word in re.finditer(r"[\w'’-]+", value):
            if word.group()[0].islower() or word.group() in _GENERIC_UNITS:
                neutral.update(range(start + word.start(), start + word.end()))
    return neutral


def bare_year_characters(text: str) -> set[int]:
    return {i for m in _BARE_YEAR.finditer(text) for i in range(m.start(), m.end())}


def gold_values(text: str, spans: list[dict]) -> list[tuple[str, set[int]]]:
    """(label, policy value characters) per gold span; spans left empty are dropped."""
    out = []
    for span in spans:
        chars = _value(text, span["start"], span["end"]) - neutral_in_span(
            text, span["start"], span["end"], span["label"])
        if chars:
            out.append((span["label"], chars))
    return out


def score(benchmark: str, system: str) -> dict:
    rows, manifest = load_verified_corpus(corpus_path(benchmark))
    predictions, receipt = load_predictions(benchmark, system, manifest["sha256"])
    gold_total = covered = non_gold = redacted = dropped = 0
    phi_docs = leak_free = negatives = negatives_clean = 0
    per_label: dict[str, dict] = {}
    by_doc: dict[str, list[tuple[int, int]]] = {"direct": [], "indirect": [], "NAME": []}
    for row in rows:
        text = row["text"]
        mapped = []
        for span in predictions[row["id"]]:
            label = project_label(system, span["label"], text[span["start"]:span["end"]])
            mapped.append({"start": span["start"], "end": span["end"], "label": COARSE[label] if label else None})
        predicted = {i for s in merge_names(mapped, text) for i in range(s["start"], s["end"])}
        all_gold = {i for s in row["spans"] for i in range(s["start"], s["end"])}
        values = gold_values(text, row["spans"])
        dropped += len(row["spans"]) - len(values)
        value_chars = set().union(*(chars for _, chars in values)) if values else set()
        neutral_outside = age_characters(text) | bare_year_characters(text)
        outside = {i for i, c in enumerate(text) if c.isalnum()} - all_gold - neutral_outside
        gold_total += len(value_chars)
        covered += len(value_chars & predicted)
        non_gold += len(outside)
        redacted += len(outside & predicted)
        counts = {"direct": [0, 0], "indirect": [0, 0], "NAME": [0, 0]}
        for label, chars in values:
            hit = chars <= predicted
            entry = per_label.setdefault(label, {"support": 0, "masked": 0})
            entry["support"] += 1
            entry["masked"] += hit
            group = "direct" if label in DIRECT_IDENTIFIERS else "indirect"
            for key in (group, "NAME") if label == "NAME" else (group,):
                counts[key][0] += hit
                counts[key][1] += 1
        for key, (hit, total) in counts.items():
            by_doc[key].append((hit, total))
        if values:
            phi_docs += 1
            leak_free += value_chars <= predicted
        else:
            negatives += 1
            negatives_clean += not (outside & predicted)
    for entry in per_label.values():
        entry["recall"] = entry["masked"] / entry["support"]

    def group(key: str) -> dict:
        hit = sum(h for h, _ in by_doc[key])
        total = sum(t for _, t in by_doc[key])
        return {"masked": hit, "support": total, "recall": hit / total if total else None,
                "recall_cluster_lower_95": cluster_lower(by_doc[key])}

    assert set(per_label) <= set(DIRECT_IDENTIFIERS) | set(INDIRECT_IDENTIFIERS)
    return {
        "benchmark": benchmark,
        "system": system,
        "view": "policy-aligned; see score_aligned.py docstring",
        "value_character_recall": covered / gold_total,
        "value_character_recall_wilson_lower_95": wilson_lower(covered, gold_total),
        "value_characters": gold_total,
        "non_phi_value_characters": non_gold,
        "non_phi_redaction": redacted / non_gold,
        "direct": group("direct"),
        "indirect": group("indirect"),
        "name": group("NAME"),
        "per_label": dict(sorted(per_label.items())),
        "gold_spans_entirely_neutral": dropped,
        "phi_documents_leak_free": {"count": leak_free, "documents": phi_docs, "rate": leak_free / phi_docs},
        "hard_negatives_clean": {"count": negatives_clean, "documents": negatives,
                                 "rate": negatives_clean / negatives if negatives else None},
        "prediction_receipt_sha256": receipt["predictions_sha256"],
        "corpus_manifest_sha256": digest(corpus_path(benchmark).with_suffix(".jsonl.manifest.json")),
        "scorer_sha256": digest(Path(__file__)),
    }


def summary_table(benchmark: str, results: list[dict]) -> str:
    def pct(value):
        return "–" if value is None else f"{100 * value:.2f}%"
    labels = [g for g in ("NAME", "ID", "MRN", "SSN", "PHONE", "EMAIL", "LOCATION", "DATE")
              if g in results[0]["per_label"]]
    lines = [
        f"## {benchmark}: policy-aligned masking",
        "",
        "Neutral characters (convention and requirement differences) are excluded "
        "for every system; see `score_aligned.py`.",
        "",
        "| System | Value-char recall | Non-PHI redaction | Direct masked | NAME masked (doc 95% lb) "
        "| Indirect masked | Leak-free PHI docs | Clean hard negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['system']} | {pct(r['value_character_recall'])} | {pct(r['non_phi_redaction'])} "
            f"| {pct(r['direct']['recall'])} | {pct(r['name']['recall'])} ({pct(r['name']['recall_cluster_lower_95'])}) "
            f"| {pct(r['indirect']['recall'])} | {pct(r['phi_documents_leak_free']['rate'])} "
            f"| {pct(r['hard_negatives_clean']['rate'])} |")
    lines += ["", "| System | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for r in results:
        lines.append(f"| {r['system']} | " + " | ".join(pct(r["per_label"][g]["recall"]) for g in labels) + " |")
    support = {g: results[0]["per_label"][g]["support"] for g in labels}
    lines += ["", "Span support after neutralisation: " + ", ".join(f"{g} {n}" for g, n in support.items())
              + f". Gold spans with no policy value left: {results[0]['gold_spans_entirely_neutral']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    args = parser.parse_args()
    results = [score(args.benchmark, system) for system in args.systems]
    write_json(RESULTS / f"{args.benchmark}.aligned.json", results)
    table = summary_table(args.benchmark, results)
    (RESULTS / f"{args.benchmark}.aligned.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
