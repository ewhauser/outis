"""Honest, typed-span evaluation for clinical PHI detection.

Token accuracy is intentionally absent: abundant non-PHI tokens can make an
unsafe detector appear accurate.  The CLI verifies a frozen corpus manifest
before loading a detector and separates model-only from hybrid results.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from .types import ENTITY_TYPES, Span


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _metrics(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Lower endpoint of the two-sided 95% Wilson score interval."""
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    adjustment = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return max(0.0, (centre - adjustment) / denominator)


def _canonical(raw: Any, length: int) -> tuple[int, int, str]:
    if isinstance(raw, dict):
        start, end, label = raw["start"], raw["end"], raw["label"]
    else:
        start, end, label = raw.start, raw.end, raw.label
    if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("Span offsets must be integer Unicode character offsets")
    if not 0 <= start < end <= length or label not in ENTITY_TYPES:
        raise ValueError(f"Invalid span: ({start}, {end}, {label}) for text length {length}")
    return start, end, label


def _characters(spans: set[tuple[int, int, str]], typed: bool = False) -> set[Any]:
    if typed:
        return {(offset, label) for start, end, label in spans for offset in range(start, end)}
    return {offset for start, end, _ in spans for offset in range(start, end)}


def evaluate(rows: Iterable[dict[str, Any]], predictor: Callable[[str], list[Span]] | Any,
             *, backend: str = "unspecified", max_failures: int = 100,
             minimum_label_support: int = 200, batch_size: int = 16) -> dict[str, Any]:
    """Measure a frozen corpus with a callable or object exposing detect(text).

    A leak-free document has every gold PHI character covered by a predicted
    span.  This privacy metric ignores type disagreements; exact typed-span F1
    reports those independently.  Failures include only this synthetic corpus.
    """
    if backend not in ("unspecified", "model", "hybrid"):
        raise ValueError("backend must be model, hybrid, or unspecified")
    detect = predictor.detect if hasattr(predictor, "detect") else predictor
    if not callable(detect):
        raise TypeError("predictor must be callable or expose detect(text)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = list(rows)
    overall = Counter(tp=0, fp=0, fn=0)
    character = Counter(tp=0, fp=0, fn=0)
    typed_character = Counter(tp=0, fp=0, fn=0)
    per_label = {label: Counter(tp=0, fp=0, fn=0) for label in ENTITY_TYPES}
    per_family: dict[str, Counter] = {}
    failures = []
    documents = leak_free = negative_documents = negative_clean = 0
    duplicate_predictions = 0
    identifiers: set[str] = set()
    started = time.perf_counter()
    def predictions():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            texts = [row["text"] for row in batch]
            if hasattr(predictor, "detect_batch"):
                predicted_batch = predictor.detect_batch(texts)
                if len(predicted_batch) != len(batch):
                    raise ValueError("detect_batch returned a different number of predictions than input texts")
                yield from zip(batch, predicted_batch)
            else:
                for row, text in zip(batch, texts):
                    yield row, detect(text)
    for row, prediction in predictions():
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError("Corpus text must be a string")
        identifier = str(row["id"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate corpus ID: {identifier}")
        identifiers.add(identifier)
        gold = {_canonical(span, len(text)) for span in row["spans"]}
        raw_predictions = list(prediction)
        predicted = {_canonical(span, len(text)) for span in raw_predictions}
        duplicate_predictions += len(raw_predictions) - len(predicted)
        hits, extras, missing = gold & predicted, predicted - gold, gold - predicted
        overall.update(tp=len(hits), fp=len(extras), fn=len(missing))
        family = str(row.get("family", "unspecified"))
        counts = per_family.setdefault(family, Counter(tp=0, fp=0, fn=0, documents=0, leak_free=0))
        counts.update(tp=len(hits), fp=len(extras), fn=len(missing), documents=1)
        for label in ENTITY_TYPES:
            per_label[label].update(tp=sum(span[2] == label for span in hits),
                                    fp=sum(span[2] == label for span in extras),
                                    fn=sum(span[2] == label for span in missing))
        expected_characters, predicted_characters = _characters(gold), _characters(predicted)
        character.update(tp=len(expected_characters & predicted_characters),
                         fp=len(predicted_characters - expected_characters),
                         fn=len(expected_characters - predicted_characters))
        expected_typed, predicted_typed = _characters(gold, True), _characters(predicted, True)
        typed_character.update(tp=len(expected_typed & predicted_typed),
                               fp=len(predicted_typed - expected_typed), fn=len(expected_typed - predicted_typed))
        safe = expected_characters <= predicted_characters
        documents += 1
        leak_free += safe
        counts["leak_free"] += safe
        if not gold:
            negative_documents += 1
            negative_clean += not predicted
        if (missing or extras) and len(failures) < max_failures:
            def details(spans):
                return [{"start": start, "end": end, "label": label, "text": text[start:end]}
                        for start, end, label in sorted(spans)]
            failures.append({"id": identifier, "family": family, "text": text,
                             "missing": details(missing), "extra": details(extras), "leak_free": safe})
    if not documents:
        raise ValueError("Cannot evaluate an empty corpus")
    span_result = _metrics(overall["tp"], overall["fp"], overall["fn"])
    label_result = {}
    for label, counts in per_label.items():
        stats = _metrics(counts["tp"], counts["fp"], counts["fn"])
        stats["recall_wilson_lower_95"] = wilson_lower(counts["tp"], counts["tp"] + counts["fn"])
        stats["sufficient_support"] = stats["support"] >= minimum_label_support
        label_result[label] = stats
    recall_lower = wilson_lower(overall["tp"], overall["tp"] + overall["fn"])
    checks = {
        "typed_span_precision_at_least_99_percent": span_result["precision"] >= .99,
        "typed_span_recall_at_least_99_percent": span_result["recall"] >= .99,
        "typed_span_f1_at_least_99_percent": span_result["f1"] >= .99,
        "phi_character_recall_at_least_99_percent": _ratio(character["tp"], character["tp"] + character["fn"]) >= .99,
        "recall_wilson_lower_95_at_least_99_percent": recall_lower >= .99,
        "all_labels_have_sufficient_support": all(stats["sufficient_support"] for stats in label_result.values()),
        "every_label_recall_at_least_99_percent": all(stats["recall"] >= .99 for stats in label_result.values()),
    }
    return {"schema_version": 1, "backend": backend, "documents": documents,
            "typed_span": span_result, "character": _metrics(character["tp"], character["fp"], character["fn"]),
            "typed_character": _metrics(typed_character["tp"], typed_character["fp"], typed_character["fn"]),
            "recall_wilson_lower_95": recall_lower, "per_label": label_result,
            "per_family": {family: {**_metrics(c["tp"], c["fp"], c["fn"]), "documents": c["documents"],
                                     "leak_free_rate": c["leak_free"] / c["documents"]} for family, c in per_family.items()},
            "document_leak_free": {"count": leak_free, "rate": leak_free / documents,
                                   "wilson_lower_95": wilson_lower(leak_free, documents)},
            "negative_documents": {"count": negative_documents, "clean_count": negative_clean,
                                   "clean_rate": _ratio(negative_clean, negative_documents)},
            "goal": {"threshold": .99, "minimum_label_support": minimum_label_support,
                     "checks": checks, "passed": all(checks.values())},
            "duplicate_predictions_removed": duplicate_predictions,
            "failures": failures, "failures_truncated": len(failures) >= max_failures,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Independently authored synthetic clinical challenge only; no production clinical accuracy claim."}


def load_verified_corpus(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    payload = path.read_bytes()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Frozen corpus manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    actual = hashlib.sha256(payload).hexdigest()
    if manifest.get("sha256") != actual:
        raise ValueError("Corpus SHA-256 differs from its frozen manifest")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if manifest.get("rows") != len(rows):
        raise ValueError("Corpus row count differs from its frozen manifest")
    support = Counter(span["label"] for row in rows for span in row["spans"])
    if any(manifest.get("label_support", {}).get(label) != support[label] for label in ENTITY_TYPES):
        raise ValueError("Corpus label support differs from its frozen manifest")
    return rows, manifest


def model_manifest(path: str | Path) -> dict[str, str]:
    """Pin local detector artifacts by file content, excluding transient files."""
    path = Path(path)
    if not path.is_dir():
        return {}
    result = {}
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix in (".json", ".pt", ".bin", ".safetensors", ".txt", ".model"):
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(candidate.relative_to(path))] = digest.hexdigest()
    return result


def detector_source_manifest() -> dict[str, str]:
    """Pin transitive package-local detector helpers as well as entry points."""
    directory = Path(__file__).parent
    pending = [directory / name for name in ("model.py", "rules.py", "types.py")]
    result = {}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(directory))
        if name in result:
            continue
        payload = path.read_bytes()
        result[name] = hashlib.sha256(payload).hexdigest()
        for node in ast.walk(ast.parse(payload.decode("utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = directory / (node.module.replace(".", "/") + ".py")
                if dependency.is_file():
                    pending.append(dependency)
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("model", "hybrid"), default="hybrid")
    parser.add_argument("--max-failures", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    rows, manifest = load_verified_corpus(args.data)
    before = model_manifest(args.model_dir)
    source_before = detector_source_manifest()
    from .model import Detector
    detector = Detector(args.model_dir, backend=args.backend, device=args.device, batch_size=args.batch_size)
    result = evaluate(rows, detector, backend=args.backend, max_failures=args.max_failures, batch_size=args.batch_size)
    after = model_manifest(args.model_dir)
    if before != after:
        raise RuntimeError("Model artifacts changed during evaluation")
    if source_before != detector_source_manifest():
        raise RuntimeError("Detector implementation changed during evaluation")
    if manifest != load_verified_corpus(args.data)[1]:
        raise RuntimeError("Corpus changed during evaluation")
    result["corpus_manifest"] = manifest
    result["model_artifacts"] = before
    result["detector_sources"] = source_before
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "typed_span": result["typed_span"],
                      "character": result["character"], "document_leak_free": result["document_leak_free"],
                      "goal": result["goal"]}, indent=2))


if __name__ == "__main__":
    main()
