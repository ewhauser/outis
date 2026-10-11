"""Filesystem locations and source provenance shared by the package."""

import ast
import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Checkout root for an editable install: holds data/, models/ and .cache/.
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
# Default detector: fine-tuned DeBERTa-v3-base (see MODEL_CARD.md).
# Relative to the working directory.
DEFAULT_MODEL_DIR = "models/deid-deberta-v3-base"
# The same checkpoint on Hugging Face, fetched when DEFAULT_MODEL_DIR has no
# weights. Pinned so each release loads the weights it was tested with.
DEFAULT_MODEL_REPO = "ewhauser/outis-deberta-v3-base"
DEFAULT_MODEL_REVISION = "11019035c00e3354fffd9631d78ce8e8efadaa09"


def resolve_model_dir(model_dir=None):
    """Directory of the detector checkpoint to load.

    An explicit ``model_dir`` is used as given. Otherwise the checkout's
    DEFAULT_MODEL_DIR is used when it holds weights, and the pinned Hugging Face
    snapshot is downloaded into the Hugging Face cache when it does not. Only
    the model is downloaded; set ``HF_HUB_OFFLINE=1`` to load from the cache
    without network access.
    """
    if model_dir is not None:
        return Path(model_dir)
    local = Path(DEFAULT_MODEL_DIR)
    if (local / "hf" / "model.safetensors").is_file():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            DEFAULT_MODEL_REPO,
            revision=DEFAULT_MODEL_REVISION,
            allow_patterns=["config.json", "hf/*"],
        )
    )


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
