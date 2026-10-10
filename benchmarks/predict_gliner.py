"""Run nvidia/gliner-PII over a converted external corpus.

Run with ``.venv-gliner``. The threshold defaults to the model card's 0.3.
Long texts are split into overlapping character windows because GLiNER
truncates its input.
"""
from __future__ import annotations

import argparse
import time

from common import load_corpus, windows, write_predictions
from labels import GLINER_ARMS


REPO = "nvidia/gliner-PII"
REVISION = "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--arm", choices=sorted(GLINER_ARMS), default="trained")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    rows, _ = load_corpus(args.benchmark)
    from gliner import GLiNER
    model = GLiNER.from_pretrained(REPO, revision=REVISION, map_location=args.device)
    model.eval()
    labels = sorted(GLINER_ARMS[args.arm])
    started = time.perf_counter()
    predictions = []
    for row in rows:
        spans = []
        for offset, excerpt in windows(row["text"], args.window, args.overlap):
            for entity in model.predict_entities(excerpt, labels, threshold=args.threshold, flat_ner=True):
                spans.append({"start": offset + entity["start"], "end": offset + entity["end"],
                              "label": entity["label"], "score": float(entity["score"])})
        predictions.append(spans)
    finished = time.perf_counter()
    path = write_predictions(args.benchmark, f"gliner-{args.arm}", rows, predictions,
                             {"repo": REPO, "revision": REVISION, "arm": args.arm, "labels": labels,
                              "threshold": args.threshold, "flat_ner": True, "device": args.device,
                              "window": args.window, "overlap": args.overlap},
                             started, finished)
    print(path)


if __name__ == "__main__":
    main()
