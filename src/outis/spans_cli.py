"""``outis spans replace`` and ``outis review``: review, then apply, a span file.

``outis batch --detect-only`` writes the span file.

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
        "spans",
        help="Apply a span file (write one with outis batch --detect-only)",
    )
    actions = spans.add_subparsers(dest="spans_command", required=True)
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


def _replaced(args, *, extended):
    from outis.replacement.transform import legacy_options
    from outis.service import Deidentifier

    settings = _policy_file(args.policy, {"policy", "replacement"})
    policy = settings.get("policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise SpanError("--policy policy must be an object")
        # Any policy keeps the response shape it gives /v1/deidentify.
        policy = {"ages": policy["ages"]} if "ages" in policy else {}
    replacement = settings.get("replacement")
    if replacement is None and extended:
        # The review needs output offsets; these options render legacy output.
        replacement = legacy_options(args.mode).model_dump()
    # The review shows notes whose detection failed; replacing them is an error.
    pairs = pair_notes(_notes(args), read_span_file(args.spans), allow_failed=extended)
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
        if record.get("error"):
            results.append(None)
            continue
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
        "Ages policy": next(
            (result["policy"]["ages"] for result in results if result), "over_89"
        ),
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
        else:
            replace(args)
    except SpanError as error:
        print(f"outis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except OSError as error:
        # Paths are the caller's own arguments; never file contents.
        print(f"outis: {error.strerror or 'file error'}", file=sys.stderr)
        raise SystemExit(2) from None
