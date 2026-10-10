#!/usr/bin/env python3
"""Audit only explicit development corpus paths; never discover data files."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outis.data.corpus_quality import audit_corpora  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help="Explicit train=path.jsonl or validation=path.jsonl; repeat as needed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON report path; existing files are never replaced",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Structural-only mode; does not certify v2 annotation metadata",
    )
    parser.add_argument(
        "--tokenizer", choices=("o200k_base", "none"), default="o200k_base"
    )
    args = parser.parse_args(argv)
    corpora = defaultdict(list)
    for argument in args.corpus:
        role, separator, path = argument.partition("=")
        if not separator or role not in {"train", "validation"} or not path:
            parser.error("--corpus must be train=PATH or validation=PATH")
        corpora[role].append(path)
    token_counter = None
    if args.tokenizer != "none":
        import tiktoken

        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))
        encoding = tiktoken.get_encoding(args.tokenizer)

        def token_counter(text):
            return len(encoding.encode(text, disallowed_special=()))

    result = audit_corpora(
        dict(corpora),
        strict_policy=not args.legacy,
        token_counter=token_counter,
        tokenizer_name=args.tokenizer if token_counter else None,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.suffix.lower() != ".json":
            parser.error("--output must name a new .json report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "errors": len(result["errors"]),
                    "warnings": len(result["warnings"]),
                    "output": str(args.output.resolve()),
                }
            )
        )
    else:
        print(serialized, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
