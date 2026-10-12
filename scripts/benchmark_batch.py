"""Measure ``outis batch`` throughput on synthetic notes only.

Notes come from the public synthetic corpus (the ``validation`` split of
``ewhauser/outis-synthetic-phi`` at a pinned revision) or, with
``--source generated``, from fictional templates built here without network
access. Real, restricted, sealed or blind corpora are never read.

The script writes a JSONL table, runs ``outis batch`` on it in a subprocess
and reports content tokens per second for detection, replacement and the whole
run, with the hardware, dtype, batch size and model revision. A content token
is one encoder-tokenizer token of note text, special tokens excluded (the
DeBERTa-v3 SentencePiece tokenizer for the default model). Figures are single
runs and indicative only.

    uv run --extra parquet scripts/benchmark_batch.py --device cpu
    uv run --extra parquet scripts/benchmark_batch.py --device cuda --batch-size 32
"""

import argparse
import json
import os
import platform
import random
import secrets
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

DATASET = "ewhauser/outis-synthetic-phi"
DATASET_REVISION = "46b09a6341de41773acdd663a759a9b8d4e25247"
DATASET_FILE = "data/validation-00000-of-00001.parquet"


def dataset_notes(count, rows_per_note):
    """Join consecutive validation rows into notes of ``rows_per_note`` rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION
    )
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    needed = count * rows_per_note
    if needed > len(texts):
        raise SystemExit(f"the validation split has {len(texts)} rows; asked {needed}")
    return [
        "\n\n".join(texts[i * rows_per_note : (i + 1) * rows_per_note])
        for i in range(count)
    ]


def generated_notes(count, rows_per_note, seed=0):
    """Fictional templated notes; every name and number is invented here."""
    rng = random.Random(seed)
    syllables = ["ka", "lo", "mi", "ren", "tas", "vel", "dor", "an", "sie", "bru"]

    def name():
        return " ".join(
            "".join(rng.choice(syllables) for _ in range(rng.randint(2, 3))).title()
            for _ in range(2)
        )

    def day():
        return (date(2020, 1, 1) + timedelta(days=rng.randrange(2000))).isoformat()

    templates = [
        "{name} was admitted on {day} with shortness of breath. Creatinine 1.{n} "
        "mg/dL; potassium 4.{n} mmol/L. Seen by Dr. {other}.",
        "Follow-up on {day}: {name} reports improved appetite and walking. "
        "MRN {mrn}. Call back on 555-01{n}{n}.",
        "Discharge summary for {name}, reviewed with {other} on {day}. Continue "
        "metoprolol 25 mg twice daily and recheck renal function in two weeks.",
        "Telephone note: {other} spoke with {name} about results from {day}. No "
        "new symptoms; plan unchanged.",
    ]
    notes = []
    for _ in range(count):
        parts = [
            rng.choice(templates).format(
                name=name(),
                other=name(),
                day=day(),
                n=rng.randint(0, 9),
                mrn=rng.randrange(10**6, 10**7),
            )
            for _ in range(rows_per_note)
        ]
        notes.append("\n\n".join(parts))
    return notes


def hardware(device):
    import torch

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    if device.startswith("cuda") or (device == "auto" and torch.cuda.is_available()):
        info["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
        info["gpu"] = "Apple MPS"
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", choices=["dataset", "generated"], default="dataset")
    parser.add_argument("--notes", type=int, default=1000)
    parser.add_argument(
        "--rows-per-note",
        type=int,
        default=4,
        help="Synthetic rows joined into each note, to approach note length",
    )
    parser.add_argument("--mode", choices=["redact", "synthetic"], default="redact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model-dir")
    parser.add_argument("--work-dir", type=Path, help="Keep inputs and outputs here")
    parser.add_argument("--report", type=Path, help="Also write the report as JSON")
    args = parser.parse_args()

    from outis.paths import DEFAULT_MODEL_REPO, DEFAULT_MODEL_REVISION

    notes = (
        dataset_notes(args.notes, args.rows_per_note)
        if args.source == "dataset"
        else generated_notes(args.notes, args.rows_per_note)
    )
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="outis-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    source = work / "notes.jsonl"
    with open(source, "w", encoding="utf-8") as stream:
        for index, text in enumerate(notes):
            row = {
                "note_id": f"bench-{index:06d}",
                "patient_id": f"bench-patient-{index % 97:03d}",
                "text": text,
            }
            stream.write(json.dumps(row) + "\n")
    output = work / f"out-{datetime.now(UTC):%Y%m%dT%H%M%S}.jsonl"
    summary_path = output.with_suffix(".summary.json")
    command = [
        sys.executable, "-m", "outis.cli", "batch",
        "--input", str(source), "--output", str(output),
        "--mode", args.mode, "--patient-column", "patient_id",
        "--device", args.device, "--summary-json", str(summary_path),
    ]  # fmt: skip
    if args.batch_size:
        command += ["--batch-size", str(args.batch_size)]
    if args.model_dir:
        command += ["--model-dir", args.model_dir]
    env = dict(os.environ)
    if args.mode == "synthetic":
        env.setdefault("OUTIS_SECRET", secrets.token_urlsafe(48))
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode not in (0, 1):
        raise SystemExit(completed.returncode)
    summary = json.loads(summary_path.read_text())
    detected = summary["processed"] - summary["failure_codes"].get("text_too_long", 0)
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": args.source,
        "dataset": f"{DATASET}@{DATASET_REVISION}"
        if args.source == "dataset"
        else None,
        "notes": summary["notes_total"],
        "rows_per_note": args.rows_per_note,
        # Notes over the per-text limit fail before detection and add no tokens.
        "mean_content_tokens_per_note": round(
            summary["content_tokens"] / max(detected, 1), 1
        ),
        "content_tokens": summary["content_tokens"],
        "mode": args.mode,
        "device": summary["device"],
        "dtype": summary["precision"],
        "batch_size": summary["batch_size"],
        "model": args.model_dir or f"{DEFAULT_MODEL_REPO}@{DEFAULT_MODEL_REVISION}",
        "model_version": summary["model_version"],
        "hardware": hardware(args.device),
        "tokens_per_second": summary["tokens_per_second"],
        "seconds": summary["seconds"],
        "failed": summary["failed"],
        "failure_codes": summary["failure_codes"],
        "oom_retries": summary["oom_retries"],
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
