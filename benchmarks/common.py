"""Standard-library helpers shared by every runner in this experiment.

Each external detector runs in its own virtual environment, so this module must
not import the project package, torch, or any third-party library.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Iterator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPORA = HERE / "corpora"
PREDICTIONS = HERE / "predictions"
RESULTS = HERE / "results"
# Benchmarks under a data use agreement. Their converted corpora and raw
# predictions (which contain note text) live outside every git repository;
# only aggregate scores are written to RESULTS.
RESTRICTED_BENCHMARKS = frozenset({"n2c2_2014_train", "n2c2_2014_test"})


def restricted_root() -> Path:
    """Directory for restricted data, from OUTIS_RESTRICTED_DIR; refused inside any git repository."""
    root = Path(os.environ.get("OUTIS_RESTRICTED_DIR", "~/working/deid/restricted")).expanduser().resolve()
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            raise ValueError(f"restricted data must not live inside a git repository: {candidate}")
    return root


def is_restricted(benchmark: str) -> bool:
    return benchmark in RESTRICTED_BENCHMARKS


# Benchmarks kept uninspected (evaluation-only terms or used as held-out
# evidence): scoring writes no failure examples for them.
UNINSPECTED_BENCHMARKS = RESTRICTED_BENCHMARKS | {"meddeid_en", "technetium_test5k", "technetium_test"}


def writes_failures(benchmark: str) -> bool:
    return benchmark not in UNINSPECTED_BENCHMARKS


def corpora_dir(benchmark: str) -> Path:
    return restricted_root() / "corpora" if is_restricted(benchmark) else CORPORA


def predictions_dir(benchmark: str) -> Path:
    return restricted_root() / "predictions" if is_restricted(benchmark) else PREDICTIONS


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> bytes:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def corpus_path(benchmark: str) -> Path:
    return corpora_dir(benchmark) / f"{benchmark}.jsonl"


def load_corpus(benchmark: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a converted corpus and check it against its frozen manifest."""
    path = corpus_path(benchmark)
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    if digest(path) != manifest["sha256"]:
        raise ValueError(f"{path} differs from its frozen manifest")
    rows = read_jsonl(path)
    if len(rows) != manifest["rows"]:
        raise ValueError(f"{path} row count differs from its frozen manifest")
    return rows, manifest


def windows(text: str, size: int, overlap: int) -> Iterator[tuple[int, str]]:
    """Yield (offset, excerpt) character windows that cover the whole text.

    Window starts move back to the nearest whitespace so a token is not cut,
    unless that would stop the window from advancing.
    """
    if size <= overlap:
        raise ValueError("window size must exceed overlap")
    start = 0
    while True:
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + overlap + 1, end)
            if cut > start + overlap:
                end = cut
        yield start, text[start:end]
        if end >= len(text):
            return
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start


def deduplicate(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-scoring prediction for each (start, end, label)."""
    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for span in spans:
        key = (span["start"], span["end"], span["label"])
        if key not in best or span.get("score", 1.0) > best[key].get("score", 1.0):
            best[key] = span
    return [best[key] for key in sorted(best)]


def write_predictions(benchmark: str, system: str, rows: list[dict[str, Any]],
                      predictions: list[list[dict[str, Any]]], configuration: dict[str, Any],
                      started: float, finished: float) -> Path:
    """Write native-label predictions plus a receipt binding them to the corpus."""
    if len(rows) != len(predictions):
        raise ValueError("prediction count differs from corpus row count")
    records = []
    for row, spans in zip(rows, predictions):
        text = row["text"]
        for span in spans:
            if not (0 <= span["start"] < span["end"] <= len(text)):
                raise ValueError(f"invalid predicted offsets in {row['id']}: {span}")
            span["text"] = text[span["start"]:span["end"]]
        records.append({"id": row["id"], "spans": deduplicate(spans)})
    path = predictions_dir(benchmark) / f"{benchmark}.{system}.jsonl"
    payload = write_jsonl(path, records)
    corpus_manifest = json.loads(corpus_path(benchmark).with_suffix(".jsonl.manifest.json").read_text())
    write_json(path.with_suffix(".receipt.json"), {
        "benchmark": benchmark,
        "system": system,
        "corpus_sha256": corpus_manifest["sha256"],
        "predictions_sha256": digest_bytes(payload),
        "documents": len(records),
        "predicted_spans": sum(len(record["spans"]) for record in records),
        "configuration": configuration,
        "elapsed_seconds": round(finished - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runner_sha256": digest(Path(sys.argv[0]).resolve()),
    })
    return path
