"""Run this project's model-only detector over a converted external corpus.

Run with the project virtual environment. Earlier prediction files in
predictions/ came from the removed Privacy-Filter adapter and are kept as
history; new runs default to a separate system key so they never overwrite
them.
"""
from __future__ import annotations

import argparse
import sys
import time

from common import ROOT, load_corpus, write_predictions

sys.path.insert(0, str(ROOT / "src"))
from outis.evaluation.evaluate import detector_source_manifest, model_manifest  # noqa: E402
from outis.paths import DEFAULT_MODEL_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / DEFAULT_MODEL_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system", default="project",
                        help="system key; must start with project")
    args = parser.parse_args()
    if not args.system.startswith("project"):
        parser.error("--system must start with 'project'")
    rows, _ = load_corpus(args.benchmark)
    artifacts, sources = model_manifest(args.model_dir), detector_source_manifest()
    from outis.detection.hf_encoder import load_detector
    detector = load_detector(args.model_dir, device=args.device, batch_size=args.batch_size)
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
    path = write_predictions(args.benchmark, args.system, rows, predictions,
                             {"backend": "model", "device": args.device, "batch_size": args.batch_size,
                              "model_artifacts": artifacts, "detector_sources": sources},
                             started, finished)
    print(path)


if __name__ == "__main__":
    main()
