"""Filesystem locations and source provenance shared by the package."""

import ast
import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Checkout root for an editable install: holds data/, models/ and .cache/.
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
# Default detector: DeBERTa-v3-base deberta_mix9c_001 step 845, promoted on
# 2026-10-08 (reports/takeover/deberta_mix9c_001/promotion_v3/RESULT.md).
# Previously ModernBERT-base mix5 epoch 1. Relative to the working directory.
DEFAULT_MODEL_DIR = "models/takeover-deberta-mix9c-001/step-000845"


def source_manifest(*entry_points: str) -> dict[str, str]:
    """SHA-256 of each entry point and every package module it transitively imports.

    Entry points and keys are paths relative to the package, such as
    ``detection/rules.py``.
    """
    pending = [PACKAGE_ROOT / name for name in entry_points]
    result = {}
    while pending:
        path = pending.pop()
        name = path.relative_to(PACKAGE_ROOT).as_posix()
        if name in result:
            continue
        payload = path.read_bytes()
        result[name] = hashlib.sha256(payload).hexdigest()
        for node in ast.walk(ast.parse(payload.decode("utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "clinical_deid."
            ):
                module = PACKAGE_ROOT / node.module.removeprefix(
                    "clinical_deid."
                ).replace(".", "/")
                for candidate in (module.with_suffix(".py"), module / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
    return dict(sorted(result.items()))
