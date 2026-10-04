"""Fully fine-tuned Hugging Face token-classification encoders.

The encoder emits one score per BIOES tag for every tokenizer token. Windows
overlap and each token's emissions come from the window that owns it, exactly
like :class:`clinical_deid.detection.adapter.Detector`; the whole document is then decoded
once with :func:`clinical_deid.detection.adapter.decode_bioes` and, for the hybrid backend,
passed through the same deterministic rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from clinical_deid.detection.adapter import TAGS, choose_device, decode_bioes
from clinical_deid.types import ENTITY_TYPES

ENCODER_KIND = "hf_token_classifier"
# Separators that subword tokenizers sometimes merge into a PHI token ("=(617)")
# but that never begin or end an annotated value.
LEADING_SEPARATORS = '=:;,|>"\u201c\u201d*'
TRAILING_SEPARATORS = '=:;,|<"\u201c\u201d*'


def trim_merged_punctuation(text, start, end):
    """Undo punctuation that subword tokens merge across a span edge.

    "((717) 555-0185" and "(+1 ..." carry an unbalanced opening parenthesis from
    the surrounding text; "Samira O.." carries the sentence period after an
    initial. Balanced parentheses and single periods are kept.
    """
    value = text[start:end]
    while value.startswith("(") and value.count("(") > value.count(")"):
        start += 1
        value = text[start:end]
    while value.endswith(")") and value.count(")") > value.count("("):
        end -= 1
        value = text[start:end]
    if value.endswith(".."):
        end -= 1
    return start, end


def refine_boundaries(spans, text):
    """Trim merged separator characters from decoded span edges."""
    from clinical_deid.types import Span

    refined = []
    for span in spans:
        start, end = span.start, span.end
        while start < end and (
            text[start] in LEADING_SEPARATORS or text[start].isspace()
        ):
            start += 1
        while end > start and (
            text[end - 1] in TRAILING_SEPARATORS or text[end - 1].isspace()
        ):
            end -= 1
        start, end = trim_merged_punctuation(text, start, end)
        if end > start:
            refined.append(
                span
                if (start, end) == (span.start, span.end)
                else Span(start, end, span.label, span.score, span.source)
            )
    return refined


def _parts(span):
    if isinstance(span, tuple):
        return span
    return span.start, span.end, span.label


def apply_label_modes(raw, hybrid, model_labels):
    """Use raw encoder spans for labels in model_labels and hybrid spans otherwise.

    Raw spans of a model-mode label replace every hybrid span of that label, and
    win any overlap with hybrid spans of other labels. Works on Span objects or
    (start, end, label) tuples.
    """
    model_labels = set(model_labels or ())
    if not model_labels:
        return list(hybrid)
    chosen = [span for span in raw if _parts(span)[2] in model_labels]

    def overlaps(span):
        a, b, _ = _parts(span)
        return any(a < _parts(c)[1] and _parts(c)[0] < b for c in chosen)

    kept = [
        span
        for span in hybrid
        if _parts(span)[2] not in model_labels and not overlaps(span)
    ]
    return sorted(
        kept + chosen,
        key=lambda span: (_parts(span)[0], _parts(span)[1], _parts(span)[2]),
    )


def window_jobs(n_tokens: int, content_window: int):
    """Yield (start, stop, owned_left, owned_right) windows covering n tokens."""
    if content_window < 1 or n_tokens < 0:
        raise ValueError("window size must be positive and token count nonnegative")
    overlap = min(128, content_window // 4)
    stride = content_window - 2 * overlap
    start = 0
    while True:
        stop = min(n_tokens, start + content_window)
        left = 0 if start == 0 else start + overlap
        right = n_tokens if stop == n_tokens else stop - overlap
        yield start, stop, left, right
        if stop >= n_tokens:
            break
        start += stride


def special_layout(tokenizer):
    """Return (prefix_ids, suffix_ids) the model expects around content tokens."""
    probe = "clinical note"
    inner = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    outer = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
    for position in range(len(outer) - len(inner) + 1):
        if outer[position : position + len(inner)] == inner:
            return outer[:position], outer[position + len(inner) :]
    raise ValueError("cannot locate content tokens inside the special-token layout")


def encode_text(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [
        tuple(pair) for pair in encoded["offset_mapping"]
    ]


class HFTokenDetector:
    """Detector-compatible wrapper around an AutoModelForTokenClassification."""

    def __init__(
        self,
        model_dir,
        *,
        backend="hybrid",
        device="auto",
        batch_size=8,
        model=None,
        tokenizer=None,
        config=None,
    ):
        if backend not in {"model", "hybrid"}:
            raise ValueError("backend must be model or hybrid")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.config = config or json.loads((self.model_dir / "config.json").read_text())
        if self.config.get("encoder") != ENCODER_KIND:
            raise ValueError("model directory is not an hf_token_classifier")
        if "tags" in self.config and self.config["tags"] != TAGS:
            raise ValueError("encoder tag ordering does not match the BIOES tag set")
        modes = self.config.get("model_labels", [])
        if not isinstance(modes, list) or any(
            label not in ENTITY_TYPES for label in modes
        ):
            raise ValueError("model_labels must be a list of supported PHI labels")
        self.device = choose_device(device)
        hf_dir = str(self.model_dir / "hf") if model is None else None
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            hf_dir, local_files_only=True
        )
        self.model = model or AutoModelForTokenClassification.from_pretrained(
            hf_dir, local_files_only=True
        )
        self.model.to(self.device).eval()
        if self.model.config.num_labels != len(TAGS):
            raise ValueError("encoder head does not match the BIOES tag set")
        self.backend = backend
        self.batch_size = batch_size
        self.max_length = int(self.config.get("max_length", 512))
        self.prefix, self.suffix = special_layout(self.tokenizer)
        self.content_window = self.max_length - len(self.prefix) - len(self.suffix)
        if self.content_window < 1:
            raise ValueError("max_length must leave room for content tokens")
        self.window = self.max_length
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        self.encoding = self.tokenizer  # runtime-manifest compatibility attribute

    def detect(self, text):
        return self.detect_batch([text])[0]

    def emissions(self, texts):
        """Return per-document (logits[n,57], offsets) using owned windows."""
        jobs, offsets_all, logits_all = [], [], []
        for doc, text in enumerate(texts):
            ids, offsets = encode_text(self.tokenizer, text)
            offsets_all.append(offsets)
            logits_all.append(torch.empty((len(ids), len(TAGS)), dtype=torch.float32))
            for start, stop, left, right in window_jobs(len(ids), self.content_window):
                if stop > start:
                    jobs.append((doc, ids[start:stop], left, right, start))
        with torch.inference_mode():
            for a in range(0, len(jobs), self.batch_size):
                batch = jobs[a : a + self.batch_size]
                rows = [self.prefix + j[1] + self.suffix for j in batch]
                size = max(len(r) for r in rows)
                tokens = torch.full((len(rows), size), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(rows), size), dtype=torch.long)
                for i, row in enumerate(rows):
                    tokens[i, : len(row)] = torch.tensor(row)
                    mask[i, : len(row)] = 1
                out = self.model(
                    input_ids=tokens.to(self.device),
                    attention_mask=mask.to(self.device),
                ).logits
                out = out.float().cpu()
                shift = len(self.prefix)
                for i, (doc, _, left, right, start) in enumerate(batch):
                    logits_all[doc][left:right] = out[
                        i, shift + left - start : shift + right - start
                    ]
        return list(zip(logits_all, offsets_all))

    def raw_batch(self, texts):
        return [
            refine_boundaries(decode_bioes(logits, offsets, text), text)
            if len(offsets)
            else []
            for (logits, offsets), text in zip(self.emissions(texts), texts)
        ]

    def detect_batch(self, texts):
        results = self.raw_batch(texts)
        if self.backend == "hybrid":
            from clinical_deid.detection.rules import augment

            model_labels = self.config.get("model_labels", [])
            results = [
                apply_label_modes(s, augment(t, s), model_labels)
                for t, s in zip(texts, results)
            ]
        return results


def load_detector(
    model_dir, *, backend="hybrid", device="auto", batch_size=8, base_dir=None
):
    """Construct the right detector class for a model directory."""
    directory = Path(model_dir)
    if (directory / "publication_in_progress.json").exists():
        raise ValueError("checkpoint publication is in progress")
    config = json.loads((directory / "config.json").read_text())
    if config.get("encoder") == ENCODER_KIND:
        return HFTokenDetector(
            model_dir, backend=backend, device=device, batch_size=batch_size
        )
    if "model_type" in config and "adapter" not in config:
        raise ValueError(
            "Pass the detector checkpoint directory containing hf/, not the nested Hugging Face directory"
        )
    if config.get("encoder") is not None:
        raise ValueError("Unsupported detector encoder kind")
    from clinical_deid.detection.adapter import Detector

    return Detector(
        model_dir,
        backend=backend,
        device=device,
        base_dir=base_dir,
        batch_size=batch_size,
    )
