"""Fingerprint of the neural detection runtime, recorded with every evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neural_runtime_manifest(model_dir, encoding=None):
    """Pin the local tokenizer, decoder and encoder implementation, and versions."""
    import tokenizers
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    from clinical_deid import types
    from clinical_deid.detection import decoding, hf_encoder

    directory = Path(model_dir) / "hf"
    encoding = (
        encoding
        if encoding is not None
        else AutoTokenizer.from_pretrained(directory, local_files_only=True)
    )
    if not getattr(encoding, "is_fast", False):
        raise ValueError("Inference requires a fast tokenizer with character offsets")
    model_config = AutoConfig.from_pretrained(directory, local_files_only=True)
    model_class = AutoModelForTokenClassification._model_mapping[type(model_config)]
    sources = {
        "decoding.py": _sha256_file(decoding.__file__),
        "types.py": _sha256_file(types.__file__),
        "hf_encoder.py": _sha256_file(hf_encoder.__file__),
        "hf_model_implementation": _sha256_file(inspect.getfile(model_class)),
        "hf_tokenizer_implementation": _sha256_file(inspect.getfile(type(encoding))),
    }
    tokenizer_state = {
        "backend": json.loads(encoding.backend_tokenizer.to_str()),
        "special_tokens": list(
            zip(encoding.all_special_tokens, encoding.all_special_ids)
        ),
        "padding_side": encoding.padding_side,
        "truncation_side": encoding.truncation_side,
        "model_max_length": encoding.model_max_length,
    }
    return {
        "neural_sources": sources,
        "decoder": {
            "name": "decode_bioes",
            "source_sha256": sources["decoding.py"],
            "boundary_refinement_sha256": sources["hf_encoder.py"],
        },
        "encoder": {
            "kind": hf_encoder.ENCODER_KIND,
            "class": model_class.__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "tokenizer": {
            "name": type(encoding).__name__,
            "n_vocab": len(encoding),
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": transformers.__version__,
            "state_sha256": hashlib.sha256(
                json.dumps(tokenizer_state, sort_keys=True).encode()
            ).hexdigest(),
        },
    }
