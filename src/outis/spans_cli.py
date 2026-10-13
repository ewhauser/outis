"""``outis spans detect|replace`` and ``outis review``: detect, review, then apply.

Messages name input or span file lines, never note text, note IDs or patient IDs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from outis.spans import (
    SpanError,
    jsonl,
    pair_notes,
    private_write,
    read_notes,
    read_span_file,
    span_record,
)


def _note_columns(parser):
    parser.add_argument("--input", type=Path, required=True, help="JSONL notes")
    parser.add_argument("--id-column", default="note_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--patient-column")
    parser.add_argument(
        "--project-id",
        help="Project for project-scoped replacement (default: main)",
    )


def _replacement_options(parser):
    parser.add_argument("--spans", type=Path, required=True, help="Span file")
    parser.add_argument("--mode", choices=["redact", "synthetic"], default="redact")
    parser.add_argument(
        "--policy",
        type=Path,
        help='JSON file {"policy": {"ages": ...}, "replacement": {...}}; other '
        "detection policy fields already acted when the spans were detected",
    )
    parser.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="Apply span records without checking their text fingerprints, for "
        "example when the fingerprint key is unavailable. Only use this when the "
        "notes are known to be the detected text",
    )


def add_parsers(sub):
    spans = sub.add_parser(
        "spans", help="Detect spans into a span file, or apply a span file"
    )
    actions = spans.add_subparsers(dest="spans_command", required=True)
    detect = actions.add_parser("detect", help="Write a span file; no text output")
    _note_columns(detect)
    detect.add_argument("--output", type=Path, required=True, help="Span file")
    detect.add_argument(
        "--policy", type=Path, help='JSON file {"policy": {...detection policy}}'
    )
    detect.add_argument("--model-dir")
    detect.add_argument("--device", default="auto")
    detect.add_argument(
        "--site-dictionaries",
        default=os.environ.get("OUTIS_SITE_DICTIONARIES") or None,
    )
    replace = actions.add_parser(
        "replace", help="Replace the spans in a span file; no detection runs"
    )
    _note_columns(replace)
    _replacement_options(replace)
    replace.add_argument("--output", type=Path, required=True, help="JSONL output")
    review = sub.add_parser(
        "review",
        help="Write a local HTML report of original and replaced text (contains PHI)",
    )
    _note_columns(review)
    _replacement_options(review)
    review.add_argument("--output", type=Path, required=True, help="HTML report")


def _policy_file(path, allowed):
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise SpanError("--policy must be a readable JSON file") from None
    if not isinstance(value, dict) or set(value) - allowed:
        raise SpanError(f"--policy may contain only {', '.join(sorted(allowed))}")
    return value


def _notes(args):
    return read_notes(
        args.input,
        id_column=args.id_column,
        text_column=args.text_column,
        patient_column=args.patient_column,
    )


def _secret():
    return os.environ.get("OUTIS_SECRET")


def _fingerprint_secret():
    """The fingerprint key: OUTIS_FINGERPRINT_SECRET, else OUTIS_SECRET."""
    return os.environ.get("OUTIS_FINGERPRINT_SECRET") or _secret()


_ROTATION_HINT = (
    "; if OUTIS_SECRET changed since detection, set OUTIS_FINGERPRINT_SECRET to "
    "the key used then, or pass --skip-fingerprint-check"
)


def detect(args):
    from outis.service import Deidentifier

    policy = _policy_file(args.policy, {"policy"}).get("policy")
    notes = _notes(args)
    options = {}
    if args.site_dictionaries:
        options["site_dictionaries"] = args.site_dictionaries
    if os.environ.get("OUTIS_FINGERPRINT_SECRET"):
        options["fingerprint_secret"] = os.environ["OUTIS_FINGERPRINT_SECRET"]
    deid = Deidentifier(args.model_dir, secret=_secret(), device=args.device, **options)
    key = (_fingerprint_secret() or "").encode("utf-8", "surrogatepass")
    fingerprint = len(key) >= 32
    if not fingerprint:
        print(
            "Note: neither OUTIS_FINGERPRINT_SECRET nor OUTIS_SECRET is set, so "
            "span records carry no text fingerprint.",
            file=sys.stderr,
        )
    records = []
    for note in notes:
        try:
            result = deid.detect(
                note["text"],
                policy=policy,
                project_id=args.project_id or "main",
                include_fingerprint=fingerprint,
            )
        except ValueError:
            raise SpanError(
                f"Input line {note['line']}: the detection policy cannot process "
                "this note"
            ) from None
        records.append(
            span_record(note["note_id"], result, patient_id=note["patient_id"])
        )
    private_write(args.output, jsonl(records))
    print(f"Wrote {len(records)} span records.", file=sys.stderr)


def _replaced(args, *, extended):
    from outis.replacement.transform import legacy_options
    from outis.service import Deidentifier

    settings = _policy_file(args.policy, {"policy", "replacement"})
    policy = settings.get("policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise SpanError("--policy policy must be an object")
        policy = {"ages": policy["ages"]} if "ages" in policy else None
    replacement = settings.get("replacement")
    if replacement is None and extended:
        # The review needs output offsets; these options render legacy output.
        replacement = legacy_options(args.mode).model_dump()
    pairs = pair_notes(_notes(args), read_span_file(args.spans))
    fingerprinted = any(record.get("text_fingerprint") for _, record in pairs)
    check = fingerprinted and not args.skip_fingerprint_check
    if check and not _fingerprint_secret():
        raise SpanError(
            "Span records carry text fingerprints; set OUTIS_FINGERPRINT_SECRET "
            "or OUTIS_SECRET to the key used at detection, or pass "
            "--skip-fingerprint-check"
        )
    if fingerprinted and not check:
        print("Warning: text fingerprints were not checked.", file=sys.stderr)
    deid = Deidentifier.for_replacement(
        secret=_secret(), fingerprint_secret=_fingerprint_secret()
    )
    results = []
    for note, record in pairs:
        request = {
            "mode": args.mode,
            "patient_id": note["patient_id"],
            "policy": policy,
            "replacement": replacement,
        }
        if args.project_id:
            request["project_id"] = args.project_id
        if check and record.get("text_fingerprint"):
            request["text_fingerprint"] = record["text_fingerprint"]
        try:
            results.append(deid.replace(note["text"], record["entities"], **request))
        except SpanError as error:
            hint = (
                _ROTATION_HINT
                if "text_fingerprint does not match" in str(error)
                else ""
            )
            raise SpanError(f"Input line {note['line']}: {error}{hint}") from None
        except ValueError:
            raise SpanError(
                f"Input line {note['line']}: the replacement options cannot process "
                "this note"
            ) from None
    return pairs, results, settings


def replace(args):
    pairs, results, _ = _replaced(args, extended=False)
    private_write(
        args.output,
        jsonl(
            {"note_id": note["note_id"], **result}
            for (note, _), result in zip(pairs, results)
        ),
    )
    print(f"Wrote {len(results)} replaced notes.", file=sys.stderr)


def review(args):
    from outis.review import PHI_WARNING, build_report

    pairs, results, settings = _replaced(args, extended=True)
    replacement = settings.get("replacement") or {}
    summary = {
        "Mode": args.mode,
        "Replacement": replacement.get("type", "default for mode"),
        "Style": replacement.get("style", "obvious"),
        "Scope": replacement.get("scope", "patient"),
        "Ages policy": (results[0]["policy"]["ages"] if results else "over_89"),
    }
    private_write(args.output, build_report(pairs, results, settings=summary))
    print(f"WARNING: {PHI_WARNING}", file=sys.stderr)
    print(
        f"Wrote a review of {len(pairs)} notes, readable by the file owner only.",
        file=sys.stderr,
    )


def run(args):
    try:
        if args.command == "review":
            review(args)
        elif args.spans_command == "detect":
            detect(args)
        else:
            replace(args)
    except SpanError as error:
        print(f"outis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except OSError as error:
        # Paths are the caller's own arguments; never file contents.
        print(f"outis: {error.strerror or 'file error'}", file=sys.stderr)
        raise SystemExit(2) from None
