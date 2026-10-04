"""Run stighellemans/meddeid-english-synth over a converted external corpus.

Run with ``.venv-meddeid``. The model weights are AGPL-3.0; this runner only
downloads and runs them locally for evaluation and does not vendor or modify
them. No patient metadata is supplied, which matches the paper's
no-metadata condition.
"""
from __future__ import annotations

import argparse
import time

from common import load_corpus, write_predictions


REPO = "stighellemans/meddeid-english-synth"
REVISION = "633af0cb7896756c3ee13df0c1b36352b291414e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--language-profile", choices=("en-US", "en-GB"), default="en-US")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    rows, _ = load_corpus(args.benchmark)
    from meddeid import Deidentifier
    import meddeid
    engine = Deidentifier.from_pretrained(REPO, revision=REVISION,
                                          language_profile=args.language_profile, device=args.device)
    started = time.perf_counter()
    predictions = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        for result in engine.deidentify_many([(row["text"], None) for row in batch]):
            predictions.append([{"start": int(s["begin"]), "end": int(s["end"]), "label": s["label"],
                                 "score": float(s.get("score", 1.0))} for s in result.spans])
    finished = time.perf_counter()
    path = write_predictions(args.benchmark, "meddeid", rows, predictions,
                             {"repo": REPO, "revision": REVISION, "meddeid_version": meddeid.__version__,
                              "language_profile": args.language_profile, "metadata": None,
                              "device": args.device, "batch_size": args.batch_size},
                             started, finished)
    print(path)


if __name__ == "__main__":
    main()
