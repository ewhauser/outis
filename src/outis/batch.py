"""Corpus-scale de-identification of a note table: ``outis batch``.

One process reads a JSONL or Parquet table, de-identifies each note's text and
writes one output row per input row. Detection runs over ``batch_size`` notes
at a time through the encoder's own window batch; each note is then rendered
by :meth:`outis.Deidentifier.deidentify`, exactly as the library does it.

Output is appended to a journal that only ever grows by whole batches. After a
batch's rows are written and flushed to disk, a small progress file records
how many input rows and output bytes are complete. ``--resume`` truncates the
journal to that recorded length and continues from the next batch, so a crash
mid-write can leave neither a torn nor a duplicated row. JSONL output is the
journal itself; Parquet output is converted from it once every row is done.

Two modes split detection from replacement. ``--detect-only`` writes a span
file (``outis-spans-v1`` JSONL, one record per note, no text) instead of
de-identified rows. ``--spans`` replaces the spans of such a file, read in
lockstep with the input, without loading the model, so a corpus can be
detected once and replaced again under another key, project or style. See
``outis.spans`` for the format.

Nothing here writes note text, note IDs or other input values to stderr.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as clock_time
from decimal import Decimal
from pathlib import Path

FORMATS = ("jsonl", "parquet")
STATE_VERSION = 1
PROGRESS_SUFFIX = ".progress.json"
JOURNAL_SUFFIX = ".partial.jsonl"
# The service's per-text limit; longer notes are failed rows, never truncated.
MAX_NOTE_CHARS = 100_000
MAX_ANCHOR_CHARS = 256
OUTPUT_FIELDS = ("status", "error", "entities", "warnings", "model_version", "mappings")
ENTITY_FIELDS = (
    "start",
    "end",
    "label",
    "score",
    "source",
    "entity_id",
    "processed_start",
    "processed_end",
)
# Service rejections, by their sanitized message, as stable non-PHI codes.
ERROR_CODES = {
    "Text must contain at most 100000 characters": "text_too_long",
    "Policy processing budget exceeded": "policy_budget_exceeded",
    "Detection count exceeds the supported limit": "too_many_entities",
    "Replacement output exceeds the size limit": "output_too_large",
    "Batch replacement output exceeds the size limit": "output_too_large",
    "Restoration marker collides with original text": "marker_collision",
}


# The model_version of rows replaced from a span file without one.
SUPPLIED = "supplied"


class BatchError(Exception):
    """An input or usage error. Messages name rows and columns, never values."""


@dataclass
class BatchConfig:
    input: Path
    output: Path
    text_column: str = "text"
    id_column: str = "note_id"
    patient_column: str | None = None
    pass_through: list[str] = field(default_factory=list)
    mode: str = "redact"
    policy_file: Path | None = None
    batch_size: int | None = None
    resume: bool = False
    format: str | None = None
    include_originals: bool = False
    model_dir: str | None = None
    device: str = "auto"
    summary_json: Path | None = None
    project_id: str | None = None
    site_dictionaries: str | None = None
    detect_only: bool = False
    spans: Path | None = None
    skip_fingerprint_check: bool = False


def add_parser(subparsers):
    """Register ``outis batch`` on the CLI's subparsers."""
    parser = subparsers.add_parser(
        "batch",
        help="De-identify a JSONL or Parquet table of notes",
        description="De-identify every note in a JSONL or Parquet table. "
        "Progress and a summary go to stderr; no note text or ID is printed.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=FORMATS,
        help="Input and output format (default: from each file's extension; "
        ".parquet and .pq are Parquet, anything else JSONL)",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="note_id")
    parser.add_argument(
        "--patient-column",
        help="Stable patient ID column; required in synthetic mode",
    )
    parser.add_argument(
        "--pass-through",
        action="append",
        default=[],
        metavar="COLUMN",
        help="Copy this input column to the output unchanged (repeatable). "
        "Pass-through values are not de-identified.",
    )
    parser.add_argument("--mode", choices=["redact", "synthetic"], default="redact")
    parser.add_argument(
        "--policy",
        type=Path,
        dest="policy_file",
        help='JSON file {"policy": {...}, "replacement": {...}}; both keys optional',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Notes per detection call and encoder windows per forward pass "
        "(default: the encoder's batch size, 8). Halved on out-of-memory errors.",
    )
    parser.add_argument(
        "--project-id",
        help="Project ID for every note; required by replacement scope "
        '"project" (see docs/replacement.md)',
    )
    parser.add_argument(
        "--site-dictionaries",
        default=os.environ.get("OUTIS_SITE_DICTIONARIES") or None,
        help="Site dictionary manifest or directory "
        "(default: OUTIS_SITE_DICTIONARIES; none when unset)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run with the same arguments",
    )
    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Copy each entity's original text into the output (contains PHI)",
    )
    parser.add_argument(
        "--model-dir",
        help="Detector checkpoint directory (default: the published model)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Also write the final summary (counts and timings only) to this file",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--detect-only",
        action="store_true",
        help="Write a JSONL span file (offsets and labels, no text) instead of "
        "de-identified rows; replace it later with --spans",
    )
    stages.add_argument(
        "--spans",
        type=Path,
        metavar="SPAN_FILE",
        help="Replace the spans in this span file instead of detecting; its "
        "records must follow the input's row order. Loads no model",
    )
    parser.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="With --spans, do not check span records' text fingerprints. Only "
        "use this when the input is known to be the detected text",
    )
    return parser


def main(args):
    """Run ``outis batch`` from parsed arguments; return the process exit code."""
    config = BatchConfig(
        input=args.input,
        output=args.output,
        text_column=args.text_column,
        id_column=args.id_column,
        patient_column=args.patient_column,
        pass_through=list(args.pass_through),
        mode=args.mode,
        policy_file=args.policy_file,
        batch_size=args.batch_size,
        resume=args.resume,
        format=args.format,
        include_originals=args.include_originals,
        model_dir=args.model_dir,
        device=args.device,
        summary_json=args.summary_json,
        project_id=args.project_id,
        site_dictionaries=args.site_dictionaries,
        detect_only=args.detect_only,
        spans=args.spans,
        skip_fingerprint_check=args.skip_fingerprint_check,
    )
    try:
        summary = run(config)
    except BatchError as exc:
        print(f"outis batch: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\noutis batch: interrupted; completed batches are saved. "
            "Re-run with --resume to continue.",
            file=sys.stderr,
        )
        return 130
    return 1 if summary["output"]["failed"] else 0


# ---------------------------------------------------------------------------
# Detection with out-of-memory recovery


def is_out_of_memory(exc):
    """True for CUDA, MPS and host allocation failures."""
    if isinstance(exc, MemoryError):
        return True
    torch = sys.modules.get("torch")
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    # MPS reports exhaustion as a plain RuntimeError.
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def release_memory():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Failed:
    """Detection outcome of a note that could not be processed."""

    __slots__ = ("code",)

    def __init__(self, code):
        self.code = code


class PrefetchedDetector:
    """Detect a whole batch up front; serve the service's per-note calls from it.

    ``Deidentifier`` asks its detector for one note at a time. This wrapper
    runs the wrapped detector once over a batch, with out-of-memory recovery,
    and answers those calls from the results. Other attributes (model
    directory, configuration, tokenizer) are the wrapped detector's.

    On an out-of-memory error the cache is released and the batch is split in
    half, and the encoder's window batch (``batch_size``) is halved with it,
    down to one note in one-window passes. A note that still fails is reported
    as ``Failed("out_of_memory")``. Any other detector error is isolated to the
    notes that raise it.
    """

    def __init__(self, detector):
        self.inner = detector
        self.results = {}
        self.oom_retries = 0
        self.batch_size = getattr(detector, "batch_size", None)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def prefetch(self, texts):
        outcomes = self._run(list(texts), self.batch_size) if texts else []
        self.results = {
            text: spans
            for text, spans in zip(texts, outcomes)
            if not isinstance(spans, Failed)
        }
        return outcomes

    def detect_batch(self, texts):
        missing = [text for text in texts if text not in self.results]
        found = dict(zip(missing, self.inner.detect_batch(missing))) if missing else {}
        return [self.results[t] if t in self.results else found[t] for t in texts]

    def detect(self, text):
        return self.detect_batch([text])[0]

    def _call(self, texts, batch_size):
        if batch_size is None:
            found = self.inner.detect_batch(texts)
        else:
            configured = self.inner.batch_size
            self.inner.batch_size = batch_size
            try:
                found = self.inner.detect_batch(texts)
            finally:
                self.inner.batch_size = configured
        if not isinstance(found, (list, tuple)) or len(found) != len(texts):
            raise RuntimeError("Detector returned invalid batch output")
        return list(found)

    def _run(self, texts, batch_size):
        try:
            return self._call(texts, batch_size)
        except Exception as exc:
            if not is_out_of_memory(exc):
                if len(texts) == 1:
                    return [Failed("detector_error")]
                return [
                    found for text in texts for found in self._run([text], batch_size)
                ]
        release_memory()
        self.oom_retries += 1
        smaller = None if batch_size is None else max(1, batch_size // 2)
        if len(texts) > 1:
            half = (len(texts) + 1) // 2
            return self._run(texts[:half], smaller) + self._run(texts[half:], smaller)
        if batch_size is not None and batch_size > 1:
            return self._run(texts, smaller)
        return [Failed("out_of_memory")]


def count_tokens(detector, texts):
    """Content tokens: the encoder tokenizer's tokens without special tokens.

    Detectors without a tokenizer count whitespace-separated words.
    """
    tokenizer = getattr(detector, "tokenizer", None)
    if tokenizer is None or not texts:
        return sum(len(text.split()) for text in texts)
    encoded = tokenizer(list(texts), add_special_tokens=False, verbose=False)
    encoded = encoded["input_ids"]
    return sum(map(len, encoded))


# ---------------------------------------------------------------------------
# Input


def table_format(path, explicit):
    if explicit:
        return explicit
    return "parquet" if Path(path).suffix.lower() in {".parquet", ".pq"} else "jsonl"


def _pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError:
        raise BatchError(
            "Parquet needs the parquet extra: pip install 'outis[parquet]'"
        ) from None
    return pyarrow


def jsonable(value):
    """A Parquet value as a JSON value; dates and decimals become strings."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date, clock_time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, bytes):
        raise BatchError("Binary columns cannot be read or passed through")
    return str(value)


def read_rows(path, fmt, columns):
    """Yield (row_number, row) with only the requested columns."""
    if fmt == "parquet":
        pa = _pyarrow()
        try:
            table = pa.parquet.ParquetFile(path)
        except (OSError, pa.ArrowException):
            raise BatchError("Input is not a readable Parquet file") from None
        names = set(table.schema_arrow.names)
        for column in columns:
            if column not in names:
                raise BatchError(f"Input has no column {column!r}")
        number = 0
        for chunk in table.iter_batches(batch_size=1024, columns=list(columns)):
            for row in chunk.to_pylist():
                number += 1
                yield number, {key: jsonable(value) for key, value in row.items()}
        return
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise BatchError(f"Line {number} is not valid JSON") from None
            if not isinstance(row, dict):
                raise BatchError(f"Line {number} is not a JSON object")
            yield number, {column: row.get(column) for column in columns}


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _kind(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str" if isinstance(value, str) else "json"


@dataclass
class InputScan:
    rows: int = 0
    characters: int = 0
    id_kind: str | None = None
    # Value kinds seen per patient and pass-through column: Parquet types.
    kinds: dict = field(default_factory=dict)


def scan_input(config, fmt, requires_patient):
    """Validate every row before any output is written."""
    columns = [config.id_column, config.text_column]
    if config.patient_column:
        columns.append(config.patient_column)
    columns.extend(config.pass_through)
    typed = [config.patient_column] if config.patient_column else []
    typed += config.pass_through
    scan = InputScan(kinds={column: set() for column in typed})
    seen = set()
    for number, row in read_rows(config.input, fmt, columns):
        where = "Line" if fmt == "jsonl" else "Row"
        note_id = row[config.id_column]
        kind = _kind(note_id)
        if note_id is None or kind not in {"str", "int"} or note_id == "":
            raise BatchError(
                f"{where} {number}: note ID must be a nonempty string or an integer"
            )
        if scan.id_kind is None:
            scan.id_kind = kind
        elif kind != scan.id_kind:
            raise BatchError(f"{where} {number}: note IDs mix strings and integers")
        if note_id in seen:
            raise BatchError(f"{where} {number}: duplicate note ID")
        seen.add(note_id)
        text = row[config.text_column]
        if not isinstance(text, str):
            raise BatchError(f"{where} {number}: text must be a string")
        if config.patient_column:
            patient = row[config.patient_column]
            if patient is None and not requires_patient:
                pass
            elif (
                _kind(patient) not in {"str", "int"}
                or not str(patient).strip()
                or len(str(patient)) > MAX_ANCHOR_CHARS
            ):
                raise BatchError(
                    f"{where} {number}: patient ID must be a nonempty string or "
                    f"an integer of at most {MAX_ANCHOR_CHARS} characters"
                )
        for column in typed:
            if row[column] is not None:
                scan.kinds[column].add(_kind(row[column]))
        scan.rows += 1
        scan.characters += len(text)
    if not scan.rows:
        raise BatchError("Input has no rows")
    return scan


def scan_spans(config, fmt, fingerprint_secret, check):
    """Check the span file against the input, row by row, before any output.

    Records must follow the input's row order, as ``--detect-only`` writes
    them, so replacement can stream both files. Returns how many records carry
    a text fingerprint.
    """
    from itertools import zip_longest

    from outis.spans import (
        SpanError,
        check_fingerprint,
        iter_span_file,
        supplied_entities,
    )

    columns = [config.id_column, config.text_column]
    if config.patient_column:
        columns.append(config.patient_column)
    where_row = "Line" if fmt == "jsonl" else "Row"
    fingerprinted = 0
    try:
        pairs = zip_longest(
            read_rows(config.input, fmt, columns), iter_span_file(config.spans)
        )
        for item, entry in pairs:
            if entry is None:
                raise BatchError(
                    "The span file has fewer records than the input has rows"
                )
            line, record = entry
            where = f"Span file line {line}"
            if item is None:
                raise BatchError(f"{where}: no matching input row")
            number, row = item
            if record["note_id"] != row[config.id_column]:
                raise BatchError(
                    f"{where}: note_id differs from input {where_row.lower()} "
                    f"{number}; records must follow the input order"
                )
            patient = _patient(row, config)
            recorded = record.get("patient_id")
            if (
                recorded is not None
                and patient is not None
                and recorded != str(patient)
            ):
                raise BatchError(f"{where}: patient_id differs from the input")
            if record.get("error"):
                continue
            text = row[config.text_column]
            try:
                supplied_entities(text, record["entities"])
            except SpanError as exc:
                raise BatchError(f"{where}: {exc}") from None
            fingerprint = record.get("text_fingerprint")
            if fingerprint is None:
                continue
            fingerprinted += 1
            if not check:
                continue
            if fingerprint_secret is None:
                raise BatchError(
                    "Span records carry text fingerprints; set "
                    "OUTIS_FINGERPRINT_SECRET or OUTIS_SECRET to the key used at "
                    "detection, or pass --skip-fingerprint-check"
                )
            try:
                check_fingerprint(fingerprint_secret, text, fingerprint)
            except SpanError:
                raise BatchError(
                    f"{where}: text_fingerprint does not match the text; if "
                    "OUTIS_SECRET changed since detection, set "
                    "OUTIS_FINGERPRINT_SECRET to the key used then, or pass "
                    "--skip-fingerprint-check"
                ) from None
            except ValueError:
                raise BatchError(
                    "OUTIS_FINGERPRINT_SECRET or OUTIS_SECRET must contain at "
                    "least 32 bytes"
                ) from None
    except SpanError as exc:
        # Malformed span file lines: the message names the line only.
        raise BatchError(str(exc)) from None
    return fingerprinted


# ---------------------------------------------------------------------------
# Durable output


def _fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return  # Not supported on this platform (Windows).
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_state(path, state):
    """Replace the progress file atomically: a crash leaves the old or the new one."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=1, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def read_state(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BatchError("Progress file is missing or unreadable") from None
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise BatchError("Progress file has an unsupported version")
    return state


def _arrow_type(pa, kinds):
    if kinds == {"bool"}:
        return pa.bool_()
    if kinds == {"int"}:
        return pa.int64()
    if kinds and kinds <= {"int", "float"}:
        return pa.float64()
    return pa.string()


def parquet_schema(pa, config, scan, include_mappings):
    fields = [(config.id_column, pa.int64() if scan.id_kind == "int" else pa.string())]
    for column in scan.kinds:
        fields.append((column, _arrow_type(pa, scan.kinds[column])))
    entity = pa.struct(
        [
            ("start", pa.int64()),
            ("end", pa.int64()),
            ("label", pa.string()),
            ("score", pa.float64()),
            ("source", pa.string()),
            ("entity_id", pa.string()),
            ("processed_start", pa.int64()),
            ("processed_end", pa.int64()),
            ("original", pa.string()),
        ]
    )
    fields += [
        (config.text_column, pa.string()),
        ("status", pa.string()),
        ("error", pa.string()),
        ("entities", pa.list_(entity)),
        ("warnings", pa.list_(pa.string())),
        ("model_version", pa.string()),
    ]
    if include_mappings:
        mapping = pa.struct(
            [("marker", pa.string()), ("original", pa.string()), ("label", pa.string())]
        )
        fields.append(("mappings", pa.list_(mapping)))
    return pa.schema(fields)


def write_parquet(journal, output, schema, pa, config, scan):
    """Convert the completed journal to Parquet without holding it in memory."""
    # Mixed or nested values are stored as JSON text.
    as_text = {
        column
        for column in scan.kinds
        if pa.types.is_string(schema.field(column).type)
        and scan.kinds[column] - {"str"}
    }
    temporary = output.with_name(output.name + ".tmp")
    writer = pa.parquet.ParquetWriter(temporary, schema)
    try:
        rows = []
        with open(journal, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for column in as_text:
                    if row.get(column) is not None and not isinstance(row[column], str):
                        row[column] = json.dumps(row[column], ensure_ascii=False)
                rows.append(row)
                if len(rows) == 4096:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    with open(temporary, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    _fsync_directory(output.parent)


# ---------------------------------------------------------------------------
# Progress


def _duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    """Counts and rates on stderr. Never receives note text or IDs."""

    def __init__(self, total, skipped, stream, interval):
        self.total = total
        self.skipped = skipped
        self.stream = stream
        self.interval = interval
        self.tty = getattr(stream, "isatty", lambda: False)()
        self.started = time.monotonic()
        self.last = None
        self.width = 0

    def update(self, processed, failed, tokens, *, final=False):
        now = time.monotonic()
        if not final and self.last is not None and now - self.last < self.interval:
            return
        self.last = now
        elapsed = max(now - self.started, 1e-9)
        done = self.skipped + processed
        remaining = self.total - done
        eta = _duration(remaining * elapsed / processed) if processed else "--:--:--"
        line = (
            f"outis batch: {done:,}/{self.total:,} notes "
            f"({100 * done / self.total:.1f}%) | {tokens / elapsed:,.0f} tok/s | "
            f"ETA {eta} | {failed:,} failed"
        )
        if self.tty:
            # Pad over the previous, possibly longer, line.
            self.width, previous = len(line), self.width
            line = line.ljust(previous)
            self.stream.write("\r" + line + ("\n" if final else ""))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# Running


def _load_detector(config):
    from outis.detection.hf_encoder import load_detector
    from outis.paths import resolve_model_dir

    kwargs = {} if config.batch_size is None else {"batch_size": config.batch_size}
    return load_detector(
        resolve_model_dir(config.model_dir), device=config.device, **kwargs
    )


def load_policy(path):
    """Read ``{"policy": ..., "replacement": ...}`` and validate both parts."""
    from outis.policy import validate_policy
    from outis.replacement.transform import ReplacementOptions

    if path is None:
        return None, None
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BatchError("Policy file is missing or not valid JSON") from None
    if not isinstance(document, dict) or set(document) - {"policy", "replacement"}:
        raise BatchError(
            'Policy file must be a JSON object with optional "policy" and '
            '"replacement" keys'
        )
    policy, replacement = document.get("policy"), document.get("replacement")
    try:
        validate_policy(policy)
    except ValueError as exc:
        raise BatchError(str(exc)) from None
    if replacement is not None:
        try:
            ReplacementOptions.model_validate(replacement)
        except (ValueError, TypeError):
            raise BatchError("Invalid replacement options") from None
    return policy, replacement


def _needs_patient(mode, replacement):
    from outis.replacement.transform import ReplacementOptions, legacy_options

    options = (
        legacy_options(mode)
        if replacement is None
        else ReplacementOptions.model_validate(replacement)
    )
    synthetic = options.type == "synthetic" or "synthetic" in options.per_type.values()
    return synthetic or options.entity_ids == "keyed"


def _entities(result, text, include_originals):
    entities = []
    for entity in result["entities"]:
        item = {key: entity[key] for key in ENTITY_FIELDS if key in entity}
        if include_originals:
            item["original"] = text[entity["start"] : entity["end"]]
        entities.append(item)
    return entities


def _failure_code(exc):
    if isinstance(exc, ValueError):
        return ERROR_CODES.get(str(exc), "rejected")
    if isinstance(exc, RuntimeError):
        return "detector_error"
    return "internal_error"


def run(config, *, stderr=None, progress_interval=None):
    """Process the input table; return the summary (counts and timings only)."""
    from outis.replacement.primitives import validated_secret
    from outis.service import Deidentifier
    from outis.site_dictionaries import load_site_dictionaries

    stderr = sys.stderr if stderr is None else stderr
    if config.batch_size is not None and config.batch_size < 1:
        raise BatchError("--batch-size must be a positive integer")
    if config.mode not in {"redact", "synthetic"}:
        raise BatchError("--mode must be redact or synthetic")
    if config.detect_only and config.spans is not None:
        raise BatchError("--detect-only and --spans cannot be combined")
    if config.skip_fingerprint_check and config.spans is None:
        raise BatchError("--skip-fingerprint-check needs --spans")
    in_format = table_format(config.input, config.format)
    out_format = table_format(config.output, config.format)
    if config.detect_only:
        if out_format != "jsonl":
            raise BatchError("--detect-only writes a JSONL span file")
        if config.pass_through or config.include_originals:
            raise BatchError(
                "--detect-only takes neither --pass-through nor --include-originals"
            )
    if in_format == "parquet" or out_format == "parquet":
        _pyarrow()
    names = [config.id_column, config.text_column, *config.pass_through]
    if config.patient_column:
        names.append(config.patient_column)
    if len(set(names)) != len(names):
        raise BatchError("ID, text, patient and pass-through columns must differ")
    if set(names) & set(OUTPUT_FIELDS):
        raise BatchError("Column names may not be any of " + ", ".join(OUTPUT_FIELDS))
    policy, replacement = load_policy(config.policy_file)
    requires_patient = not config.detect_only and _needs_patient(
        config.mode, replacement
    )
    secret = os.environ.get("OUTIS_SECRET") or None
    # Fingerprints have their own key, so OUTIS_SECRET can be rotated.
    fingerprint_secret = os.environ.get("OUTIS_FINGERPRINT_SECRET") or secret
    if requires_patient:
        if not config.patient_column:
            raise BatchError("Synthetic and keyed replacement need --patient-column")
        try:
            validated_secret(secret)
        except ValueError:
            raise BatchError(
                "Synthetic and keyed replacement need OUTIS_SECRET with at least 32 bytes"
            ) from None
    if not config.input.is_file():
        raise BatchError("Input file not found")
    if config.spans is not None and not config.spans.is_file():
        raise BatchError("Span file not found")
    try:
        # Messages name dictionaries and limits, never terms. Supplied spans
        # were found with any dictionaries already, so --spans loads none.
        dictionaries = (
            None
            if config.spans is not None
            else load_site_dictionaries(config.site_dictionaries)
        )
    except ValueError as exc:
        raise BatchError(str(exc)) from None
    output = config.output
    progress_path = output.with_name(output.name + PROGRESS_SUFFIX)
    journal = (
        output
        if out_format == "jsonl"
        else output.with_name(output.name + JOURNAL_SUFFIX)
    )
    if output.resolve() == config.input.resolve():
        raise BatchError("Output must not overwrite the input")
    if config.spans is not None and output.resolve() == config.spans.resolve():
        raise BatchError("Output must not overwrite the span file")
    if not config.resume and any(
        path.exists() for path in (output, progress_path, journal)
    ):
        raise BatchError(
            "Output already exists; pass --resume to continue an interrupted "
            "run, or choose a new output path"
        )
    # With nothing to resume, --resume starts the run, so a job can always pass it.
    resuming = config.resume and progress_path.exists()
    if config.resume and not resuming and (output.exists() or journal.exists()):
        raise BatchError("Output exists without its progress file; cannot resume")

    print("outis batch: validating input", file=stderr)
    scan_started = time.monotonic()
    scan = scan_input(config, in_format, requires_patient)
    check = config.spans is not None and not config.skip_fingerprint_check
    if config.spans is not None:
        fingerprinted = scan_spans(config, in_format, fingerprint_secret, check)
        if fingerprinted and not check:
            print("outis batch: text fingerprints were not checked", file=stderr)
    scan_seconds = time.monotonic() - scan_started
    fingerprints = False
    if config.detect_only and fingerprint_secret is not None:
        from outis.spans import text_fingerprint

        try:
            text_fingerprint(fingerprint_secret, "")
        except ValueError:
            raise BatchError(
                "OUTIS_FINGERPRINT_SECRET or OUTIS_SECRET must contain at least "
                "32 bytes to fingerprint span records"
            ) from None
        fingerprints = True
    elif config.detect_only:
        print(
            "outis batch: neither OUTIS_FINGERPRINT_SECRET nor OUTIS_SECRET is "
            "set, so span records carry no text fingerprint",
            file=stderr,
        )
    load_started = time.monotonic()
    if config.spans is not None:
        detector = None
        deid = Deidentifier.for_replacement(
            secret=secret, fingerprint_secret=fingerprint_secret
        )
    else:
        print("outis batch: loading model", file=stderr)
        detector = PrefetchedDetector(_load_detector(config))
        deid = Deidentifier(
            config.model_dir,
            secret=secret,
            detector=detector,
            site_dictionaries=dictionaries,
            fingerprint_secret=fingerprint_secret,
        )
    load_seconds = time.monotonic() - load_started
    # The per-note arguments that every note shares.
    if config.detect_only:
        request = {"policy": policy, "project_id": config.project_id or "main"}
        if fingerprints:
            request["include_fingerprint"] = True
    else:
        request = {"mode": config.mode, "replacement": replacement}
        if config.spans is not None:
            # Only ages acts on supplied spans; any policy keeps the response
            # shape a policy gives /v1/deidentify.
            request["policy"] = (
                None
                if policy is None
                else ({"ages": policy["ages"]} if "ages" in policy else {})
            )
        else:
            request["policy"] = policy
        if config.project_id is not None:
            request["project_id"] = config.project_id
    # Validate the policy, replacement, scope and site dictionary selection as
    # the service will apply them, on an empty note that skips the model.
    anchor = "outis-batch-check" if requires_patient else None
    try:
        if config.spans is not None:
            deid.replace("", [], patient_id=anchor, **request)
        else:
            detector.results = {"": []}
            if config.detect_only:
                deid.detect("", **request)
            else:
                deid.deidentify("", patient_id=anchor, **request)
    except ValueError as exc:
        raise BatchError(str(exc)) from None
    finally:
        if detector is not None:
            detector.results = {}
    if detector is None:
        batch_size = config.batch_size or 64
    else:
        batch_size = detector.batch_size or config.batch_size or 8
    device = getattr(detector, "device", None)
    version = SUPPLIED if config.spans is not None else deid.version

    settings = {
        "input_sha256": file_digest(config.input),
        "input_format": in_format,
        "output_format": out_format,
        "text_column": config.text_column,
        "id_column": config.id_column,
        "patient_column": config.patient_column,
        "pass_through": list(config.pass_through),
        "mode": config.mode,
        "policy": hashlib.sha256(
            json.dumps([policy, replacement], sort_keys=True).encode()
        ).hexdigest(),
        "include_originals": config.include_originals,
        "project_id": config.project_id,
        "site_dictionaries": getattr(dictionaries, "digest", None),
        "batch_size": batch_size,
        "model_version": version,
        "device": None if device is None else str(device),
        "precision": getattr(detector, "precision", None),
        # A keyed digest: identifies the secret without revealing it.
        "secret": hmac.new(
            secret.encode(), b"outis batch resume", "sha256"
        ).hexdigest()[:16]
        if requires_patient
        else None,
    }
    # Only the split stages add settings, so earlier progress files still match.
    if config.detect_only:
        settings["stage"] = "detect"
        settings["fingerprint"] = (
            hmac.new(
                fingerprint_secret.encode(), b"outis batch fingerprint", "sha256"
            ).hexdigest()[:16]
            if fingerprints
            else None
        )
    elif config.spans is not None:
        settings["stage"] = "replace"
        settings["spans_sha256"] = file_digest(config.spans)
        settings["skip_fingerprint_check"] = config.skip_fingerprint_check
    if resuming:
        state = read_state(progress_path)
        changed = sorted(
            key
            for key in set(settings) | set(state["settings"])
            if state["settings"].get(key) != settings.get(key)
        )
        if changed:
            raise BatchError(
                "--resume needs the interrupted run's input and settings; "
                "changed: " + ", ".join(changed)
            )
        if not journal.exists() or journal.stat().st_size < state["output_bytes"]:
            if not state["complete"]:
                raise BatchError("Output is shorter than its checkpoint; cannot resume")
    else:
        state = {
            "version": STATE_VERSION,
            "settings": settings,
            "rows_done": 0,
            "output_bytes": 0,
            "ok": 0,
            "failed": 0,
            "failure_codes": {},
            "complete": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        journal.touch()
        if config.detect_only:
            # Span files carry note and patient IDs: owner-only, like spans.py.
            os.chmod(journal, 0o600)
        write_state(progress_path, state)

    skipped = state["rows_done"]
    totals = {"processed": 0, "ok": 0, "failed": 0, "tokens": 0, "codes": {}}
    timing = {"detection": 0.0, "replacement": 0.0}
    progress = Progress(
        scan.rows,
        skipped,
        stderr,
        progress_interval
        if progress_interval is not None
        else (2.0 if getattr(stderr, "isatty", lambda: False)() else 30.0),
    )
    started = time.monotonic()
    if not state["complete"]:
        rows = (row for _, row in read_rows(config.input, in_format, _columns(config)))
        if config.spans is not None:
            # Records follow the input order, checked by scan_spans: stream them.
            from outis.spans import iter_span_file

            records = (record for _, record in iter_span_file(config.spans))

            def process(batch):
                pairs = [(row, next(records)) for row in batch]
                return _replace(pairs, config, deid, request, check, totals, timing)

        elif config.detect_only:

            def process(batch):
                return _detect(batch, config, deid, detector, request, totals, timing)

        else:

            def process(batch):
                return _process(batch, config, deid, detector, request, totals, timing)

        with open(journal, "r+b") as sink:
            sink.truncate(state["output_bytes"])
            sink.seek(state["output_bytes"])
            for _ in range(skipped):
                next(rows)
                if config.spans is not None:
                    next(records)
            batch = []
            for row in rows:
                batch.append(row)
                if len(batch) == batch_size:
                    _commit(process(batch), sink, state, progress_path, totals)
                    batch = []
                    progress.update(
                        totals["processed"], totals["failed"], totals["tokens"]
                    )
            if batch:
                _commit(process(batch), sink, state, progress_path, totals)
        if out_format == "parquet":
            pa = _pyarrow()
            schema = parquet_schema(
                pa,
                config,
                scan,
                bool(replacement and replacement.get("include_mapping")),
            )
            write_parquet(journal, output, schema, pa, config, scan)
        state["complete"] = True
        write_state(progress_path, state)
        if journal != output:
            journal.unlink()
    elapsed = time.monotonic() - started
    progress.update(totals["processed"], totals["failed"], totals["tokens"], final=True)

    def rate(seconds):
        return round(totals["tokens"] / seconds, 1) if seconds > 0 else None

    summary = {
        "notes_total": scan.rows,
        "processed": totals["processed"],
        "skipped": skipped,
        "failed": totals["failed"],
        "failure_codes": dict(sorted(totals["codes"].items())),
        "output": {
            "ok": state["ok"],
            "failed": state["failed"],
            "failure_codes": dict(sorted(state["failure_codes"].items())),
        },
        "content_tokens": totals["tokens"],
        "tokens_per_second": {
            "detection": rate(timing["detection"]),
            "replacement": rate(timing["replacement"]),
            "end_to_end": rate(elapsed),
        },
        "seconds": {
            "input_scan": round(scan_seconds, 3),
            "model_load": round(load_seconds, 3),
            "detection": round(timing["detection"], 3),
            "replacement": round(timing["replacement"], 3),
            "end_to_end": round(elapsed, 3),
        },
        "oom_retries": 0 if detector is None else detector.oom_retries,
        "batch_size": batch_size,
        "device": settings["device"],
        "precision": settings["precision"],
        "model_version": version,
        "mode": config.mode,
    }
    codes = ", ".join(
        f"{code} {count:,}" for code, count in summary["failure_codes"].items()
    )
    print(
        f"outis batch: done. processed {summary['processed']:,}, skipped "
        f"{skipped:,} (completed earlier), failed {summary['failed']:,}"
        + (f" ({codes})" if codes else "")
        + f"; output holds {state['ok']:,} ok and {state['failed']:,} failed rows",
        file=stderr,
    )
    rates = summary["tokens_per_second"]
    if totals["processed"]:
        print(
            f"outis batch: {totals['tokens']:,} content tokens in {elapsed:,.1f}s: "
            f"{rates['end_to_end'] or 0:,.0f} tok/s end to end, "
            f"{rates['detection'] or 0:,.0f} detection, "
            f"{rates['replacement'] or 0:,.0f} replacement",
            file=stderr,
        )
    if config.summary_json is not None:
        config.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _columns(config):
    columns = [config.id_column, config.text_column]
    if config.patient_column:
        columns.append(config.patient_column)
    return columns + list(config.pass_through)


def _prefetch(texts, detector, totals, timing):
    """Detect a batch's notes; overlong notes fail without reaching the model."""
    eligible = [i for i, text in enumerate(texts) if len(text) <= MAX_NOTE_CHARS]
    started = time.monotonic()
    found = detector.prefetch([texts[i] for i in eligible])
    timing["detection"] += time.monotonic() - started
    outcomes = [Failed("text_too_long")] * len(texts)
    for i, outcome in zip(eligible, found):
        outcomes[i] = outcome
    totals["tokens"] += count_tokens(detector, [texts[i] for i in eligible])
    return outcomes


def _patient(row, config):
    return row.get(config.patient_column) if config.patient_column else None


def _output_row(config, row, text, result, code, model_version, totals):
    """One de-identified output row, or a failed row carrying ``code``."""
    patient = _patient(row, config)
    out = {config.id_column: row[config.id_column]}
    if config.patient_column:
        out[config.patient_column] = patient
    for column in config.pass_through:
        out[column] = row[column]
    if result is None:
        out.update(
            {
                config.text_column: None,
                "status": "failed",
                "error": code,
                "entities": [],
                "warnings": [],
                "model_version": model_version,
            }
        )
        totals["codes"][code] = totals["codes"].get(code, 0) + 1
    else:
        out.update(
            {
                config.text_column: result["text"],
                "status": "ok",
                "error": None,
                "entities": _entities(result, text, config.include_originals),
                "warnings": result["warnings"],
                "model_version": model_version,
            }
        )
        if "mappings" in result:
            out["mappings"] = result["mappings"]
    return out


def _process(batch, config, deid, detector, request, totals, timing):
    """De-identify one batch of rows; return their output rows in input order."""
    texts = [row[config.text_column] for row in batch]
    outcomes = _prefetch(texts, detector, totals, timing)
    started = time.monotonic()
    rows = []
    for row, text, outcome in zip(batch, texts, outcomes):
        patient = _patient(row, config)
        result = None
        if not isinstance(outcome, Failed):
            try:
                result = deid.deidentify(
                    text,
                    patient_id=None if patient is None else str(patient),
                    **request,
                )
            except Exception as exc:
                outcome = Failed(_failure_code(exc))
        code = None if result is not None else outcome.code
        rows.append(_output_row(config, row, text, result, code, deid.version, totals))
    timing["replacement"] += time.monotonic() - started
    detector.results = {}
    return rows


def _detect(batch, config, deid, detector, request, totals, timing):
    """Detect one batch of rows; return their span records in input order."""
    from outis.spans import failed_record, span_record

    texts = [row[config.text_column] for row in batch]
    outcomes = _prefetch(texts, detector, totals, timing)
    started = time.monotonic()
    records = []
    for row, text, outcome in zip(batch, texts, outcomes):
        patient = _patient(row, config)
        patient = None if patient is None else str(patient)
        note_id = row[config.id_column]
        if not isinstance(outcome, Failed):
            try:
                records.append(
                    span_record(
                        note_id, deid.detect(text, **request), patient_id=patient
                    )
                )
                continue
            except Exception as exc:
                outcome = Failed(_failure_code(exc))
        totals["codes"][outcome.code] = totals["codes"].get(outcome.code, 0) + 1
        records.append(failed_record(note_id, outcome.code, patient_id=patient))
    timing["replacement"] += time.monotonic() - started
    detector.results = {}
    return records


def _replace(pairs, config, deid, request, check, totals, timing):
    """Replace one batch's supplied spans; return output rows in input order."""
    texts = [row[config.text_column] for row, _ in pairs]
    totals["tokens"] += count_tokens(None, texts)
    started = time.monotonic()
    rows = []
    for (row, record), text in zip(pairs, texts):
        patient = _patient(row, config)
        result, code = None, record.get("error")
        if code is None:
            extra = {}
            if check and record.get("text_fingerprint"):
                extra["text_fingerprint"] = record["text_fingerprint"]
            try:
                result = deid.replace(
                    text,
                    record["entities"],
                    patient_id=None if patient is None else str(patient),
                    **request,
                    **extra,
                )
            except Exception as exc:
                code = _failure_code(exc)
        version = record.get("model_version") or SUPPLIED
        rows.append(_output_row(config, row, text, result, code, version, totals))
    timing["replacement"] += time.monotonic() - started
    return rows


def _commit(rows, sink, state, progress_path, totals):
    """Append whole rows, flush them to disk, then advance the checkpoint."""
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    sink.write(payload)
    sink.flush()
    os.fsync(sink.fileno())
    # Failed rows and failed span records both carry an error code.
    failed = [row["error"] for row in rows if row.get("error")]
    state["rows_done"] += len(rows)
    state["output_bytes"] += len(payload)
    state["ok"] += len(rows) - len(failed)
    state["failed"] += len(failed)
    for code in failed:
        state["failure_codes"][code] = state["failure_codes"].get(code, 0) + 1
    write_state(progress_path, state)
    totals["processed"] += len(rows)
    totals["ok"] += len(rows) - len(failed)
    totals["failed"] += len(failed)
