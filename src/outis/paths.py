"""Filesystem locations and source provenance shared by the package."""

import ast
import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Checkout root for an editable install: holds data/, models/ and .cache/.
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
# Default detector: DeBERTa-v3-base deberta_mix9_001 step 9288 (see MODEL_CARD.md).
# Relative to the working directory.
DEFAULT_MODEL_DIR = "models/deid-deberta-v3-base"


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
                "outis."
            ):
                module = PACKAGE_ROOT / node.module.removeprefix("outis.").replace(
                    ".", "/"
                )
                for candidate in (module.with_suffix(".py"), module / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
    return dict(sorted(result.items()))
