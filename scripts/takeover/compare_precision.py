"""Compare a detector's fp32 and bf16 emissions on a development bank.

Reports counts only (no note text): the largest logit difference, how many
tokens change their top tag, and how many decoded spans appear or disappear.

  PYTHONPATH=src python scripts/takeover/compare_precision.py \
      --data data/X.jsonl --model-dir models/M --output out.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from takeover.eval_bank import check_output, load_bank, write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    check_output(args.output)
    # detailed=True refuses held-out proxy banks: this compares per-document output.
    rows, manifest = load_bank(args.data, detailed=True)
    rows = rows[: args.limit] if args.limit else rows

    from outis.detection.decoding import TAGS, decode_bioes
    from outis.detection.hf_encoder import load_detector, refine_boundaries

    detector = load_detector(args.model_dir, device=args.device, batch_size=args.batch_size)
    tokens = argmax_changed = docs_changed = spans_lost = spans_gained = 0
    lost_by_label, gained_by_label = {}, {}
    max_abs = 0.0
    abs_sum = 0.0
    for offset in range(0, len(rows), 16):
        texts = [row["text"] for row in rows[offset : offset + 16]]
        detector.precision = "fp32"
        full = detector.emissions(texts)
        detector.precision = "bf16"
        half = detector.emissions(texts)
        for text, (a, offsets), (b, _) in zip(texts, full, half):
            if not len(offsets):
                continue
            diff = (a - b).abs()
            max_abs = max(max_abs, float(diff.max()))
            abs_sum += float(diff.sum())
            tokens += len(offsets)
            argmax_changed += int((a.argmax(-1) != b.argmax(-1)).sum())
            sa = {(s.start, s.end, s.label) for s in refine_boundaries(decode_bioes(a, offsets, text), text)}
            sb = {(s.start, s.end, s.label) for s in refine_boundaries(decode_bioes(b, offsets, text), text)}
            docs_changed += sa != sb
            for _, _, label in sa - sb:
                spans_lost += 1
                lost_by_label[label] = lost_by_label.get(label, 0) + 1
            for _, _, label in sb - sa:
                spans_gained += 1
                gained_by_label[label] = gained_by_label.get(label, 0) + 1
        print(f"{min(offset + 16, len(rows))}/{len(rows)} docs", file=sys.stderr, flush=True)
    report = {
        "data": args.data, "corpus_sha256": manifest.get("sha256"), "model_dir": args.model_dir,
        "device": args.device, "documents": len(rows), "tokens": tokens, "tags": len(TAGS),
        "logit_max_abs_diff": max_abs, "logit_mean_abs_diff": abs_sum / max(tokens * len(TAGS), 1),
        "tokens_top_tag_changed": argmax_changed,
        "documents_with_span_changes": docs_changed,
        "fp32_spans_missing_in_bf16": spans_lost, "missing_by_label": lost_by_label,
        "bf16_spans_not_in_fp32": spans_gained, "extra_by_label": gained_by_label,
    }
    write_new_json(args.output, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
