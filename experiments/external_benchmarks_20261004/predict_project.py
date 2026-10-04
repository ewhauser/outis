"""Run this project's frozen Detector over a converted external corpus.

Run with the project virtual environment.
"""
from __future__ import annotations

import argparse
import sys
import time

from common import ROOT, load_corpus, write_predictions

sys.path.insert(0, str(ROOT / "src"))
from clinical_deid.evaluation.evaluate import detector_source_manifest, model_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "clinical-adapter"))
    parser.add_argument("--base-dir", default=None,
                        help="privacy-filter-base directory; defaults to the adapter's sibling")
    parser.add_argument("--backend", choices=("model", "hybrid"), default="hybrid")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system", default=None,
                        help="system key; defaults to project (hybrid) or project-model, and must start with project")
    args = parser.parse_args()
    if args.system is not None and not args.system.startswith("project"):
        parser.error("--system must start with 'project'")
    rows, _ = load_corpus(args.benchmark)
    artifacts, sources = model_manifest(args.model_dir), detector_source_manifest()
    from clinical_deid.detection.adapter import Detector
    detector = Detector(args.model_dir, backend=args.backend, device=args.device,
                        base_dir=args.base_dir, batch_size=args.batch_size)
    started = time.perf_counter()
    predictions = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        for spans in detector.detect_batch([row["text"] for row in batch]):
            predictions.append([{"start": s.start, "end": s.end, "label": s.label,
                                 "score": float(s.score), "source": s.source} for s in spans])
    finished = time.perf_counter()
    if artifacts != model_manifest(args.model_dir) or sources != detector_source_manifest():
        raise RuntimeError("Detector artifacts or sources changed during prediction")
    system = args.system or ("project" if args.backend == "hybrid" else "project-model")
    path = write_predictions(args.benchmark, system, rows, predictions,
                             {"backend": args.backend, "device": args.device, "batch_size": args.batch_size,
                              "model_artifacts": artifacts, "detector_sources": sources},
                             started, finished)
    print(path)


if __name__ == "__main__":
    main()
