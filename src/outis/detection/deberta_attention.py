"""Gather-free relative attention for DeBERTa-v2/v3 encoders.

transformers' ``DisentangledSelfAttention.disentangled_attention_bias`` scores
every token against the relative-position slots and then picks each
(query, key) pair's slot with ``torch.gather`` over a [batch*heads, L, L]
index, twice per layer. On Apple MPS that gather and the scatter-add in its
backward pass dominate training time.

A pair's slot depends only on the offset i - j. This version expands the small
per-head position table to the 2L-1 possible offsets (shared by the whole
batch), scores tokens against offsets with one batched matmul, and aligns the
offsets into the [L, L] grid with the Transformer-XL pad-and-reshape shift,
which is views only. Parameters, outputs and gradients match the stock code to
float32 rounding; the parity tests are in ``tests/test_deberta_attention.py``.

The replacement is bound per model instance, never globally, and only when the
stock transformers code is the version it was checked against.
"""

from __future__ import annotations

import hashlib
import inspect
import types
import warnings

import torch

# sha256 of inspect.getsource() for the stock code this replaces or relies on
# (transformers 5.18.0). build_relative_position is what makes positions a
# function of i - j. A different transformers version keeps stock code.
CHECKED_SOURCES = {
    "DisentangledSelfAttention.disentangled_attention_bias": "f4f4b7467256b601d9392c818be680ea96d80f676d71e30d91375de0d4fbc995",
    "DisentangledSelfAttention.forward": "4f0ce96b1027bd763187ec8400e4bf501a66515804ccf7539bace174985f5c29",
    "build_relative_position": "00558921462f21358afc8b09815c18dc053210482f2e664aba9556fa162d901e",
}


def _modeling():
    from transformers.models.deberta_v2 import modeling_deberta_v2

    return modeling_deberta_v2


def stock_code_matches() -> bool:
    def resolve(path):
        target = _modeling()
        for part in path.split("."):
            target = getattr(target, part)
        return target

    return all(
        hashlib.sha256(inspect.getsource(resolve(path)).encode()).hexdigest() == digest
        for path, digest in CHECKED_SOURCES.items()
    )


def _skew(x: torch.Tensor) -> torch.Tensor:
    """[N, L, 2L-1] by offset column c -> [N, L, L] with out[i, j] = x[i, j - i + L - 1]."""
    n, length, width = x.shape
    if length == 1:
        return x
    flat = x.reshape(n, length * width)
    start = length - 1
    return flat[:, start : start + length * (width - 1)].reshape(n, length, width - 1)[
        :, :, :length
    ]


def _offset_buckets(relative_pos: torch.Tensor) -> torch.Tensor:
    """Bucketed position for offset column c, where i - j = L - 1 - c."""
    return torch.cat([relative_pos[1:, 0].flip(0), relative_pos[0, :]])


def gather_free_attention_bias(
    self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
):
    stock = type(self).disentangled_attention_bias
    length = query_layer.size(1)
    while (
        relative_pos is not None
        and relative_pos.dim() > 2
        and relative_pos.size(0) == 1
    ):
        relative_pos = relative_pos[0]  # the encoder passes [1, L, L]
    if (
        relative_pos is None
        or relative_pos.dim() != 2
        or key_layer.size(1) != length
        or relative_pos.shape != (length, length)
    ):
        return stock(
            self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
        )
    buckets = _offset_buckets(
        relative_pos.to(device=query_layer.device, dtype=torch.long)
    )

    modeling = _modeling()
    heads = self.num_attention_heads
    batch_heads, _, head_dim = query_layer.shape
    batch = batch_heads // heads
    span = self.pos_ebd_size
    rel_embeddings = rel_embeddings[: span * 2, :]

    def table(projection):  # [heads, 2 * span, head_dim], shared by the batch
        return (
            projection(rel_embeddings).view(span * 2, heads, head_dim).transpose(0, 1)
        )

    def by_offset(layer, positions, index):
        offsets = positions.index_select(1, index)  # [heads, 2L - 1, head_dim]
        scores = torch.matmul(
            layer.view(batch, heads, length, head_dim),
            offsets.transpose(-1, -2).unsqueeze(0),
        )
        return _skew(scores.view(batch_heads, length, 2 * length - 1))

    score = 0
    if "c2p" in self.pos_att_type:
        pos_key = table(self.key_proj if self.share_att_key else self.pos_key_proj)
        index = torch.clamp(buckets + span, 0, span * 2 - 1)
        c2p = by_offset(query_layer, pos_key, index)
        scale = modeling.scaled_size_sqrt(pos_key, scale_factor)
        score = score + c2p / scale.to(dtype=c2p.dtype)
    if "p2c" in self.pos_att_type:
        pos_query = table(
            self.query_proj if self.share_att_key else self.pos_query_proj
        )
        index = torch.clamp(-buckets + span, 0, span * 2 - 1)
        p2c = by_offset(key_layer, pos_query, index).transpose(-1, -2)
        scale = modeling.scaled_size_sqrt(pos_query, scale_factor)
        score = score + p2c / scale.to(dtype=p2c.dtype)
    return score


def apply_gather_free_attention(model) -> bool:
    """Bind the gather-free bias on a DeBERTa-v2/v3 model's attention modules.

    Returns True when applied. Other model types, and DeBERTa under an
    unchecked transformers version, are left unchanged.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "deberta-v2":
        return False
    if not stock_code_matches():
        warnings.warn(
            "transformers DeBERTa attention differs from the checked version; "
            "using the stock implementation",
            stacklevel=2,
        )
        return False
    attention = _modeling().DisentangledSelfAttention
    modules = [module for module in model.modules() if isinstance(module, attention)]
    for module in modules:
        module.disentangled_attention_bias = types.MethodType(
            gather_free_attention_bias, module
        )
    return bool(modules)
