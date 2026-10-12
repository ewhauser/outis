"""Development evaluation of any detector on a manifest-verified bank.

One neural pass produces the detector's spans; there is no rule layer. Reports
the GOALS.md privacy block, the seven synthetic goal gates, per-label and
per-cohort metrics. Held-out banks run with failures suppressed so their
content is never written or shown (use --show-failures only on development
banks whose errors may be inspected).

  PYTHONPATH=src python scripts/takeover/eval_bank.py --data data/X.jsonl \
      --model-dir models/M --output reports/takeover/eval/X__M.json [--show-failures]

Refuses banks whose manifest declares independent_final or a final/test role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

from outis.evaluation.evaluate import (  # noqa: E402
    detector_source_manifest,
    evaluate,
    load_verified_corpus,
    model_manifest,
)
from outis.evaluation.runtime import neural_runtime_manifest  # noqa: E402
from outis.types import ENTITY_TYPES  # noqa: E402

PROTECTED_WORDS = ("blind_qualification_v6", "blind_qualification_v7", "blind_qualification_v9",
                   "qualification_v9", "v7_qualification_author_qa", ".sealed", "controls/validation.jsonl",
                   "v15_name_next_step_analysis_001/recovery_data_001")


def check_path(path):
    path = Path(path)
    if any(word in str(candidate).lower() for candidate in (path, path.resolve()) for word in PROTECTED_WORDS):
        raise ValueError("refusing a protected path")
    return path


def check_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("bank metadata must be an object")
    if any(metadata.get(key) not in (None, False) for key in ("independent_final", "independent")):
        raise ValueError("refusing a final/independent bank: sealed qualification is out of scope for development evaluation")
    tokens = set()
    for key in ("suite", "split", "role", "bank_role", "dataset_role"):
        tokens.update(re.split(r"[^a-z0-9]+", str(metadata.get(key, "")).lower()))
    if tokens & {"final", "test", "independent", "sealed", "qualification"}:
        raise ValueError("refusing a final/independent bank: sealed qualification is out of scope for development evaluation")
    return bool(metadata.get("proxy_role") or metadata.get("is_proxy") or tokens & {"proxy", "blind"})


def load_bank(path, *, detailed=False):
    """Establish the bank's role before reading any corpus bytes."""
    path = check_path(path)
    sidecar = check_path(path.with_suffix(path.suffix + ".manifest.json"))
    declaration = json.loads(sidecar.read_text())
    proxy = check_metadata(declaration)
    proxy |= any("proxyq_" in str(candidate).lower() for candidate in (path, path.resolve(), sidecar.resolve()))
    # Preserve proxy identity even when somebody makes an ordinary copied alias.
    proxy |= any("proxyq_" in str(declaration.get(key, "")).lower() for key in ("corpus", "data", "source"))
    if detailed and proxy:
        raise ValueError("held-out proxy banks may not have failures or predictions written")
    rows, manifest = load_verified_corpus(path)
    if manifest != declaration:
        raise ValueError("corpus manifest changed while checking its development role")
    if not rows:
        raise ValueError("evaluation corpus must not be empty")
    ids = set()
    for row in rows:
        row_proxy = check_metadata(row)
        if detailed and row_proxy:
            raise ValueError("held-out proxy banks may not have failures or predictions written")
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("corpus ids must be unique nonempty strings")
        ids.add(rid)
    return rows, manifest


def guarded_model_manifest(path):
    path = check_path(path)
    # Preflight every symlink destination before the manifest helper hashes files.
    for candidate in path.rglob("*"):
        check_path(candidate)
    artifacts = model_manifest(path)
    if "config.json" not in artifacts:
        raise ValueError("a local detector checkpoint with config.json is required")
    return artifacts


def source_manifest():
    sources = detector_source_manifest()
    for name, path in {
        "hf_encoder.py": ROOT / "src/outis/detection/hf_encoder.py",
        "evaluate.py": ROOT / "src/outis/evaluation/evaluate.py",
        "runtime.py": ROOT / "src/outis/evaluation/runtime.py",
        "scripts/takeover/eval_bank.py": Path(__file__),
    }.items():
        sources[name] = hashlib.sha256(check_path(path).read_bytes()).hexdigest()
    return sources


def check_output(path):
    path = check_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("evaluation outputs are immutable; choose a new output path")
    return path


def write_new_json(path, value):
    path = check_output(path)
    payload = json.dumps(value, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(payload)


class Precomputed:
    def __init__(self, spans_by_text):
        self.spans_by_text = spans_by_text

    def detect(self, text):
        return self.spans_by_text[text]

    def detect_batch(self, texts):
        return [self.spans_by_text[t] for t in texts]


def summarize(result):
    keep = {k: result[k] for k in ("typed_span", "privacy", "character", "recall_wilson_lower_95", "goal", "documents",
                                    "document_leak_free", "phi_bearing_documents", "phi_bearing_documents_excluding_ages_89_and_under",
                                    "negative_documents") if k in result}
    keep["per_label"] = {label: {m: result["per_label"][label][m] for m in
                                 ("true_positives", "false_positives", "false_negatives", "precision", "recall", "support")}
                         for label in ENTITY_TYPES}
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--predictions", help="optional path to save predictions (development banks only)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch size and limit must be positive")
    output_path = check_output(args.output)
    prediction_path = check_output(args.predictions) if args.predictions else None
    if prediction_path and output_path.resolve() == prediction_path.resolve():
        raise ValueError("report and prediction outputs must be different paths")
    if args.predictions and args.limit is not None:
        raise SystemExit("source-selection predictions require the complete development bank")
    rows, manifest = load_bank(args.data, detailed=bool(args.show_failures or args.predictions))
    complete_rows = rows
    corpus_rows = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    artifacts = guarded_model_manifest(args.model_dir)
    model_sha256 = hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runtime = neural_runtime_manifest(args.model_dir)
    sources = source_manifest()

    from outis.detection.hf_encoder import load_detector

    started = time.perf_counter()
    detector = load_detector(args.model_dir, device=args.device, batch_size=args.batch_size,
                             precision=args.precision)
    load_seconds = time.perf_counter() - started
    texts = [row["text"] for row in rows]
    raw = {}
    t0 = time.perf_counter()
    chunk = 16
    for offset in range(0, len(texts), chunk):
        batch = texts[offset:offset + chunk]
        predicted = detector.detect_batch(batch)
        if len(predicted) != len(batch):
            raise ValueError("detector must return one span list for every input row")
        for text, spans in zip(batch, predicted):
            if text in raw and spans != raw[text]:
                raise ValueError("repeated corpus text produced inconsistent predictions")
            raw[text] = spans
    neural_seconds = time.perf_counter() - t0

    max_failures = len(rows) if args.show_failures else 0
    report = {"data": args.data, "corpus_sha256": manifest.get("sha256"), "model_dir": args.model_dir,
              "rows": len(rows), "device": args.device, "batch_size": args.batch_size,
              "precision": getattr(detector, "precision", args.precision),
              "seconds": {"load": load_seconds, "neural": neural_seconds},
              "failures_written": bool(args.show_failures),
              "model_sha256": model_sha256, "model_artifacts": artifacts,
              "neural_runtime": runtime, "source_manifest": sources}
    for backend, table in (("model", raw),):
        result = evaluate(rows, Precomputed(table), max_failures=max(max_failures, 0) or 0)
        entry = summarize(result)
        if args.show_failures:
            entry["failures"] = result.get("failures", [])
        cohorts = {}
        for cohort in sorted({row.get("cohort", "unspecified") for row in rows}):
            subset = [row for row in rows if row.get("cohort", "unspecified") == cohort]
            sub = evaluate(subset, Precomputed(table), max_failures=0)
            cohorts[cohort] = {"documents": len(subset), "typed_span": sub["typed_span"],
                               "character": sub.get("character"),
                               "phi_bearing_documents": sub.get("phi_bearing_documents")}
        entry["per_cohort"] = cohorts
        report[backend] = entry
    # Do every integrity check before either output can become a usable artifact.
    if artifacts != guarded_model_manifest(args.model_dir):
        raise SystemExit("model artifacts changed during evaluation")
    if runtime != neural_runtime_manifest(args.model_dir, getattr(detector, "encoding", None)):
        raise SystemExit("neural runtime changed during evaluation")
    if sources != source_manifest():
        raise SystemExit("detector or evaluator sources changed during evaluation")
    final_rows, final_manifest = load_bank(args.data, detailed=bool(args.show_failures or args.predictions))
    if final_rows != complete_rows or final_manifest != manifest:
        raise SystemExit("corpus changed during evaluation")
    check_output(output_path)
    if prediction_path:
        check_output(prediction_path)
        write_new_json(prediction_path, {
            "schema_version": 1, "corpus_sha256": manifest["sha256"], "corpus_rows": corpus_rows,
            "model_dir": str(Path(args.model_dir).resolve()), "model_sha256": model_sha256,
            "model_artifacts": artifacts,
            "neural_runtime": runtime, "source_manifest": sources,
            "predictions": [{"id": row["id"], "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                             "spans": [s.as_dict() for s in raw[row["text"]]]} for row in rows]})
    write_new_json(output_path, report)
    m = report["model"]
    print(json.dumps({"data": args.data, "model": args.model_dir,
                      "typed_span": {k: round(m["typed_span"][k], 5) for k in ("precision", "recall", "f1")} | {
                          "fp": m["typed_span"]["false_positives"], "fn": m["typed_span"]["false_negatives"]},
                      "privacy": m.get("privacy"),
                      "seconds": report["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
