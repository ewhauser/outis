"""Score stored predictions for an external benchmark and write a comparison.

Run with the project virtual environment. Headline metrics are label-agnostic,
matching the MedDeID paper so its published ASQ-PHI figures can be reproduced:

* character recall: gold PHI characters covered by any prediction;
* non-PII redaction: predicted characters outside gold, over all non-gold
  characters (raw, and with explicit ages below 90 removed from the numerator,
  because ASQ-PHI does not annotate them);
* leak-free PHI documents and clean hard negatives.

Every prediction counts towards those metrics, whatever its label, because the
system would remove that text. Typed metrics then use the project evaluator on
coarse groups (``labels.COARSE``). Adjacent NAME spans separated only by
whitespace are merged for every system, since some label sets split first and
last names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from common import RESULTS, ROOT, corpus_path, digest, is_restricted, predictions_dir, read_jsonl, write_json
from labels import COARSE, project_label

sys.path.insert(0, str(ROOT / "src"))
from clinical_deid.evaluation.evaluate import evaluate, load_verified_corpus, wilson_lower  # noqa: E402


PUBLISHED = {
    # MedDeID manuscript Table 3 / model card, ASQ-PHI, no metadata.
    "asq_phi": {
        "meddeid": {"character_recall": .9890, "non_pii_redaction": .0621, "non_pii_redaction_age_adjusted": .0089},
    },
}
_AGE = re.compile(
    r"(?i)\b(?:aged?\s+)?(\d{1,2})\s*(?:-\s*)?(?:years?|yrs?|y)\s*(?:-\s*)?(?:old|o\b|/o\b)"
    r"|\b(\d{1,2})\s*(?:yo|y/o|y\.o\.)\b"
    r"|\b(?:aged?|age:)\s*(\d{1,2})\b"
    r"|\b(\d{1,2})\s?(?:M|F)\b"
)


def age_characters(text: str) -> set[int]:
    """Characters of explicit ages below 90, which ASQ-PHI leaves unannotated."""
    found = set()
    for match in _AGE.finditer(text):
        value = next(group for group in match.groups() if group is not None)
        if int(value) < 90:
            found.update(range(match.start(), match.end()))
    return found


def merge_names(spans: list[dict], text: str) -> list[dict]:
    """Merge NAME spans that are adjacent or separated only by whitespace."""
    merged: list[dict] = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        last = merged[-1] if merged else None
        if (last and span["label"] == "NAME" and last["label"] == "NAME" and span["start"] >= last["end"]
                and not text[last["end"]:span["start"]].strip()):
            last["end"] = span["end"]
            continue
        merged.append(dict(span))
    return merged


class _Replay:
    """Feed precomputed predictions to the project evaluator in corpus order."""

    def __init__(self, texts: list[str], predictions: list[list[dict]]):
        self._queue = list(zip(texts, predictions))

    def detect(self, text: str) -> list[dict]:
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: list[str]) -> list[list[dict]]:
        batch, self._queue = self._queue[:len(texts)], self._queue[len(texts):]
        if [text for text, _ in batch] != list(texts):
            raise RuntimeError("evaluator requested texts out of corpus order")
        return [spans for _, spans in batch]


def _chars(spans) -> set[int]:
    return {i for s in spans for i in range(s["start"], s["end"])}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def load_predictions(benchmark: str, system: str, corpus_sha256: str) -> tuple[dict, dict]:
    path = predictions_dir(benchmark) / f"{benchmark}.{system}.jsonl"
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    if receipt["corpus_sha256"] != corpus_sha256:
        raise ValueError(f"{system} predictions were made on a different corpus")
    if receipt["predictions_sha256"] != digest(path):
        raise ValueError(f"{system} predictions differ from their receipt")
    return {record["id"]: record["spans"] for record in read_jsonl(path)}, receipt


def score(benchmark: str, system: str) -> dict:
    rows, manifest = load_verified_corpus(corpus_path(benchmark))
    predictions, receipt = load_predictions(benchmark, system, manifest["sha256"])
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError(f"{system} predictions do not cover the corpus exactly")
    gold_total = covered = non_gold = exposed_non_gold = exposed_age = 0
    span_tp = span_fp = span_fn = 0
    phi_docs = phi_leak_free = negatives = negatives_clean = 0
    unmapped: dict[str, int] = {}
    coarse_rows, coarse_predictions = [], []
    for row in rows:
        text, gold = row["text"], row["spans"]
        native = predictions[row["id"]]
        mapped = []
        for span in native:
            label = project_label(system, span["label"], text[span["start"]:span["end"]])
            if label is None:
                unmapped[span["label"]] = unmapped.get(span["label"], 0) + 1
            mapped.append({"start": span["start"], "end": span["end"],
                           "label": COARSE[label] if label else None})
        merged = merge_names(mapped, text)
        # Label-agnostic character and span metrics use every prediction.
        gold_chars, predicted_chars = _chars(gold), _chars(merged)
        outside = predicted_chars - gold_chars
        gold_total += len(gold_chars)
        covered += len(gold_chars & predicted_chars)
        non_gold += len(text) - len(gold_chars)
        exposed_non_gold += len(outside)
        exposed_age += len(outside & age_characters(text))
        gold_bounds = {(s["start"], s["end"]) for s in gold}
        predicted_bounds = {(s["start"], s["end"]) for s in merged}
        span_tp += len(gold_bounds & predicted_bounds)
        span_fp += len(predicted_bounds - gold_bounds)
        span_fn += len(gold_bounds - predicted_bounds)
        if gold:
            phi_docs += 1
            phi_leak_free += gold_chars <= predicted_chars
        else:
            negatives += 1
            negatives_clean += not predicted_chars
        coarse_rows.append({"id": row["id"], "text": text, "family": row.get("family", "unspecified"),
                            "spans": [{"start": s["start"], "end": s["end"], "label": COARSE[s["label"]]}
                                      for s in gold]})
        coarse_predictions.append([{"start": s["start"], "end": s["end"], "label": s["label"]}
                                   for s in merged if s["label"]])
    if non_gold != manifest["characters"]["non_gold"]:
        raise RuntimeError("non-gold character count differs from the corpus manifest")
    replay = _Replay([row["text"] for row in coarse_rows], coarse_predictions)
    # Restricted benchmarks: no failure examples, which would copy note text into results.
    typed = evaluate(coarse_rows, replay, max_failures=0 if is_restricted(benchmark) else 50, minimum_label_support=30)
    typed["scope"] = f"External benchmark {benchmark}; coarse label groups; not a clinical accuracy claim."
    typed.pop("goal", None)
    return {
        "benchmark": benchmark,
        "system": system,
        "label_agnostic": {
            "character_recall": covered / gold_total,
            "character_recall_wilson_lower_95": wilson_lower(covered, gold_total),
            "gold_characters": gold_total,
            "non_pii_characters": non_gold,
            "non_pii_redacted_characters": exposed_non_gold,
            "non_pii_redaction": exposed_non_gold / non_gold,
            "non_pii_redaction_age_adjusted": (exposed_non_gold - exposed_age) / non_gold,
            "explicit_age_characters_redacted": exposed_age,
            "exact_span": _prf(span_tp, span_fp, span_fn),
            "phi_documents_leak_free": {"count": phi_leak_free, "documents": phi_docs,
                                        "rate": phi_leak_free / phi_docs if phi_docs else None},
            "hard_negatives_clean": {"count": negatives_clean, "documents": negatives,
                                     "rate": negatives_clean / negatives if negatives else None},
        },
        "coarse_typed": typed,
        "unmapped_prediction_labels": unmapped,
        "published_reference": PUBLISHED.get(benchmark, {}).get(system),
        "prediction_receipt": receipt,
        "corpus_manifest_sha256": digest(corpus_path(benchmark).with_suffix(".jsonl.manifest.json")),
        "scorer_sha256": digest(Path(__file__)),
    }


def summary_table(benchmark: str, results: list[dict]) -> str:
    def pct(value):
        return "–" if value is None else f"{100 * value:.2f}%"
    lines = [
        f"## {benchmark}: label-agnostic",
        "",
        "| System | Char recall | Non-PII redaction | Age-adjusted | Exact-span P | Exact-span R | Exact-span F1 "
        "| Leak-free PHI docs | Clean hard negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        la = result["label_agnostic"]
        lines.append(
            f"| {result['system']} | {pct(la['character_recall'])} | {pct(la['non_pii_redaction'])} "
            f"| {pct(la['non_pii_redaction_age_adjusted'])} | {pct(la['exact_span']['precision'])} "
            f"| {pct(la['exact_span']['recall'])} | {pct(la['exact_span']['f1'])} "
            f"| {pct(la['phi_documents_leak_free']['rate'])} | {pct(la['hard_negatives_clean']['rate'])} |")
    groups = [g for g in dict.fromkeys(COARSE.values())
              if any(r["coarse_typed"]["per_label"][g]["support"] for r in results)]
    lines += ["", f"## {benchmark}: coarse typed exact spans", "",
              "| System | Typed P | Typed R | Typed F1 | " + " | ".join(f"{g} R" for g in groups) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(groups)]
    for result in results:
        typed = result["coarse_typed"]
        per = typed["per_label"]
        lines.append(f"| {result['system']} | {pct(typed['typed_span']['precision'])} "
                     f"| {pct(typed['typed_span']['recall'])} | {pct(typed['typed_span']['f1'])} | "
                     + " | ".join(pct(per[g]["recall"]) for g in groups) + " |")
    references = [(r["system"], r["published_reference"]) for r in results if r["published_reference"]]
    if references:
        lines += ["", "## Published reference figures", "",
                  "| System | Char recall | Non-PII redaction | Age-adjusted |", "|---|---:|---:|---:|"]
        for system, ref in references:
            lines.append(f"| {system} (published) | {pct(ref['character_recall'])} "
                         f"| {pct(ref['non_pii_redaction'])} | {pct(ref['non_pii_redaction_age_adjusted'])} |")
    support = {g: results[0]["coarse_typed"]["per_label"][g]["support"] for g in groups}
    lines += ["", "Gold support: " + ", ".join(f"{g} {n}" for g, n in support.items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    args = parser.parse_args()
    results = []
    for system in args.systems:
        result = score(args.benchmark, system)
        write_json(RESULTS / f"{args.benchmark}.{system}.json", result)
        results.append(result)
    table = summary_table(args.benchmark, results)
    (RESULTS / f"{args.benchmark}.summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
