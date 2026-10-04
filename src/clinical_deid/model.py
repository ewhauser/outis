"""Privacy Filter backbone with a trained residual clinical encoder adapter + BIOES NER."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from pathlib import Path
from types import MethodType
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F
import tiktoken

from .types import ENTITY_TYPES, Span

BASE_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
BASE_WEIGHTS_SHA256 = "9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c"
TAGS = ["O"] + [f"{boundary}-{label}" for label in ENTITY_TYPES for boundary in "BIES"]


def get_encoding(base_dir=None):
    root = (
        Path(base_dir).resolve().parent.parent
        if base_dir
        else Path(__file__).resolve().parents[2]
    )
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root / ".cache" / "tiktoken"))
    return tiktoken.get_encoding("o200k_base")


BACKBONE_COMPUTATION_VERSION = "grouped-moe-v2-fp32-mps-rank-sum"
BACKBONE_COMPUTE_DTYPE = "float32"


def grouped_moe(self, x):
    """Same MoE computation as OPF, grouped into padded per-expert GEMMs.

    Power-of-two capacity buckets avoid per-token expert weight replication.
    This portable implementation supports Apple Metal and CPU without Triton.
    Frozen backbone only; parity is checked against the upstream reference.
    """
    from opf._model.model import swiglu

    t = self.norm(x).reshape(-1, x.shape[-1])
    gate = F.linear(t.float(), self.gate.weight.float(), self.gate.bias.float())
    values, indices = gate.topk(self.experts_per_token, dim=-1, sorted=True)
    weights = values.softmax(-1)
    expert_ids = indices.reshape(-1)
    order = expert_ids.argsort(stable=True)
    sorted_ids = expert_ids[order]
    token_ids = order // self.experts_per_token
    counts = torch.bincount(sorted_ids, minlength=self.num_experts).cpu().tolist()
    buckets = {}
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            capacity = 1 << (count - 1).bit_length()
            buckets.setdefault(capacity, []).append((expert, offset, count))
        offset += count
    stable_mps = x.device.type == "mps"
    # Capacity buckets depend on companion tokens. Accumulating in bucket order
    # changes floating-point summation order and can change later expert routes.
    # Store MPS contributions by each token's own top-k rank before reducing.
    output = (
        torch.empty((expert_ids.numel(), t.shape[-1]), device=x.device, dtype=torch.float32)
        if stable_mps else torch.zeros_like(t, dtype=torch.float32)
    )
    for capacity, experts in buckets.items():
        ids = torch.tensor([e[0] for e in experts], device=x.device)
        packed = t.new_zeros((len(experts), capacity, t.shape[-1]))
        for i, (_, start, count) in enumerate(experts):
            packed[i, :count] = t[token_ids[start : start + count]]
        w1 = self.mlp1_weight[ids]
        w2 = self.mlp2_weight[ids]
        b1, b2 = self.mlp1_bias[ids, None, :], self.mlp2_bias[ids, None, :]
        if stable_mps:
            # Match upstream reference arithmetic even for a half-stored module.
            w1, w2, b1, b2 = w1.float(), w2.float(), b1.float(), b2.float()
        h = torch.bmm(packed.to(w1.dtype), w1) + b1
        h = swiglu(h, limit=self.swiglu_limit, packed=self.packed_geglu)
        y = torch.bmm(h.to(w2.dtype), w2) + b2
        for i, (_, start, count) in enumerate(experts):
            positions = order[start : start + count]
            contribution = y[i, :count].float() * weights.reshape(-1)[positions, None]
            if stable_mps:
                output[positions] = contribution
            else:
                output.index_add_(0, token_ids[start : start + count], contribution)
    if stable_mps:
        ranked = output.view(t.shape[0], self.experts_per_token, t.shape[-1])
        # MPS sum(dim=1) also changes its reduction tree with the outer shape.
        # Separate additions fix the operation order for each token and channel.
        output = ranked[:, 0].clone()
        for rank in range(1, self.experts_per_token):
            output = output + ranked[:, rank]
    return x + output.reshape_as(x).to(x.dtype)


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(base_dir, device="auto"):
    from opf._model.model import Transformer

    device = choose_device(device)
    weights = Path(base_dir) / "original" / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != BASE_WEIGHTS_SHA256:
        raise ValueError(
            "Base encoder weights differ from the pinned upstream artifact"
        )
    torch.set_num_threads(4)
    model = Transformer.from_checkpoint(str(Path(base_dir) / "original"), device="cpu")
    # Half-precision residual/attention arithmetic amplified tiny batch-shape
    # differences into different PHI spans. Keep the complete backbone in fp32;
    # using fp32 only inside experts did not fix the full-document failures.
    model = model.to(dtype=getattr(torch, BACKBONE_COMPUTE_DTYPE), device=device)
    model.requires_grad_(False).eval()
    for block in model.block:
        block.mlp.forward = MethodType(grouped_moe, block.mlp)
    return model, device


def tokenize(text, encoding):
    from opf._core.spans import token_char_ranges_for_text

    ids = encoding.encode(text, disallowed_special=())
    starts, ends = token_char_ranges_for_text(ids, encoding, text)
    return ids, list(zip(starts, ends))


def encode_backbone(model, ids, mask):
    with torch.no_grad():
        x = model.embedding(ids)
        for block in model.block:
            x = block(x, attention_mask=mask)
        return model.norm(x).float()


class ClinicalAdapter(nn.Module):
    """Trainable bidirectional transformer domain adapter followed by NER head."""

    def __init__(self, hidden_size=640, bottleneck=160):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.encoder = nn.TransformerEncoderLayer(
            bottleneck,
            4,
            bottleneck * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.up = nn.Linear(bottleneck, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, len(TAGS))

    def forward(self, features, mask):
        x = self.down(features)
        # Add positional encoding so the adapter learns span boundary structure.
        pos = torch.arange(x.shape[1], device=x.device).float()[:, None]
        freq = torch.exp(
            torch.arange(0, x.shape[-1], 2, device=x.device).float()
            * (-math.log(10000.0) / x.shape[-1])
        )
        pe = x.new_zeros((x.shape[1], x.shape[-1]))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        x = self.encoder(x + pe[None], src_key_padding_mask=~mask)
        x = self.norm(features + self.up(x))
        return self.classifier(x)


def decode_bioes(logits, offsets, text):
    """Constrained Viterbi with valid BIOES transitions and exact character offsets."""
    scores = logits.detach().float().cpu()
    n, c = scores.shape
    if n == 0:
        return []
    if not torch.isfinite(scores).all():
        raise RuntimeError("Invalid model logits; no deidentified output released")
    legal = torch.full((c, c), float("-inf"))
    for prev in range(c):
        for nxt in range(c):
            p, q = TAGS[prev], TAGS[nxt]
            if p == "O" or p[0] in "ES":
                valid = q == "O" or q[0] in "BS"
            else:
                valid = q != "O" and q[0] in "IE" and p[2:] == q[2:]
            if valid:
                legal[prev, nxt] = 0
    initial = torch.tensor(
        [0 if t == "O" or t[0] in "BS" else float("-inf") for t in TAGS]
    )
    final = torch.tensor(
        [0 if t == "O" or t[0] in "ES" else float("-inf") for t in TAGS]
    )
    dp = scores[0] + initial
    back = []
    for i in range(1, n):
        best, bp = (dp[:, None] + legal).max(0)
        dp = best + scores[i]
        back.append(bp)
    last = int((dp + final).argmax())
    labels = [last]
    for bp in reversed(back):
        last = int(bp[last])
        labels.append(last)
    labels.reverse()
    spans = []
    start = None
    for i, label in enumerate(labels):
        tag = TAGS[label]
        if tag == "O":
            continue
        if tag[0] in "BS":
            start = i
        if tag[0] in "ES" and start is not None:
            a, b = offsets[start][0], offsets[i][1]
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            # Tokenizers can attach terminal punctuation to a PHI token.
            while b > a and text[b - 1] in ",;:":
                b -= 1
            # BPE can merge an email's first letter with a preceding contact
            # separator (e.g. ';m'). BIOES cannot express a subtoken boundary.
            # These characters cannot start an unquoted email local part.
            if tag[2:] == "EMAIL":
                while a < b and text[a] in ";,:[(<":
                    a += 1
            if tag[2:] in {"NAME", "FACILITY", "LOCATION"}:
                # NER must not disclose the remaining letters of a partially
                # predicted word. Include Unicode combining marks and internal
                # name apostrophes/hyphens, without crossing whitespace.
                def letter(char):
                    return unicodedata.category(char)[0] in {"L", "M"}

                left_limit = offsets[start - 1][1] if start > 0 and labels[start - 1] != 0 else 0
                right_limit = offsets[i + 1][0] if i + 1 < n and labels[i + 1] != 0 else len(text)
                while a > left_limit and a < b:
                    if letter(text[a - 1]) and letter(text[a]):
                        a -= 1
                    elif (
                        text[a] in "'’-" and a + 1 < b
                        and letter(text[a - 1]) and letter(text[a + 1])
                    ):
                        a -= 1
                    elif (
                        text[a - 1] in "'’-" and a - 2 >= left_limit
                        and letter(text[a - 2]) and letter(text[a])
                    ):
                        a -= 2
                    else:
                        break
                while 0 < b < right_limit:
                    if letter(text[b - 1]) and letter(text[b]):
                        b += 1
                    elif (
                        text[b - 1] in "'’-" and b - 2 >= a
                        and letter(text[b - 2]) and letter(text[b])
                    ):
                        b += 1
                    elif (
                        text[b] in "'’-" and b + 1 < right_limit
                        and letter(text[b - 1]) and letter(text[b + 1])
                    ):
                        b += 2
                    else:
                        break
                if tag[2:] == "NAME":
                    # An English possessive is grammar outside the person's
                    # name. Normalize after word completion: BPE may attach
                    # the suffix, or a partial prediction may stop inside it.
                    # A true lexical ending is required so O'Sullivan and
                    # apostrophe/hyphen compounds remain complete, including
                    # when another decoded entity prevents right expansion.
                    possessive_end = b
                    while possessive_end > a and text[possessive_end - 1] in ".!?":
                        possessive_end -= 1
                    terminal = (
                        possessive_end == len(text)
                        or not (
                            text[possessive_end].isalnum() or letter(text[possessive_end]) or text[possessive_end] == "_"
                            or (
                                text[possessive_end] in "'’-\u2010\u2011" and possessive_end + 1 < len(text)
                                and letter(text[possessive_end + 1])
                            )
                        )
                    )
                    def period_ends_name(value):
                        if re.search(r"\b(?:Jr|Sr)\.$", value, re.I):
                            return True
                        # A terminal personal initial can include Unicode
                        # combining marks. A period after a longer word is
                        # not enough evidence to remove following grammar.
                        if not value.endswith("."):
                            return False
                        index = len(value) - 2
                        while index >= 0 and unicodedata.category(value[index])[0] == "M":
                            index -= 1
                        return (
                            index >= 0 and unicodedata.category(value[index])[0] == "L"
                            and (index == 0 or value[index - 1].isspace() or value[index - 1] == ".")
                        )

                    suffix_before_possessive = period_ends_name(text[a:possessive_end - 2])
                    if terminal and possessive_end - a > 2 and text[possessive_end - 2] in "'’" and text[possessive_end - 1] in "sS" and (letter(text[possessive_end - 3]) or suffix_before_possessive):
                        b = possessive_end - 2
                    elif (
                        b > a and text[b - 1] in "'’"
                        and period_ends_name(text[a:b - 1])
                        and text[b:b + 1] in {"s", "S"}
                        and (b + 1 == len(text) or not (
                            letter(text[b + 1]) or text[b + 1].isalnum() or text[b + 1] == "_"
                            or (text[b + 1] in "'’\u2010\u2011-" and b + 2 < len(text) and letter(text[b + 2]))
                        ))
                    ):
                        # BPE can end immediately after the apostrophe in
                        # "Jr.'s" or "M.'s". Keep the period, excluding possessive
                        # grammar only when the complete following s is proven.
                        b -= 1
                    elif terminal and possessive_end - a > 1 and text[possessive_end - 1] in "'’" and text[possessive_end - 2] in "sS":
                        # Plural possessives such as James' retain the final s.
                        b = possessive_end - 1
                    # An edge apostrophe can be a quotation delimiter or part
                    # of a name. Require a paired quote in this paragraph and
                    # preserve ambiguous leading elisions such as 't Hooft.
                    opening = a if text[a:a + 1] in {"'", "‘"} else a - 1
                    if (
                        opening >= 0 and text[opening] in "'‘"
                        and (opening == 0 or not (
                            letter(text[opening - 1]) or text[opening - 1].isalnum()
                            or text[opening - 1] == "_"
                        ))
                    ):
                        close_char = "’" if text[opening] == "‘" else "'"
                        first = opening + 1
                        after_initial = first + 1
                        while after_initial < len(text) and unicodedata.category(text[after_initial])[0] == "M":
                            after_initial += 1
                        leading_elision = (
                            first < len(text) and unicodedata.category(text[first])[0] == "L"
                            and after_initial < len(text)
                            and (text[after_initial].isspace() or text[after_initial] in "-\u2010\u2011")
                        )
                        if not leading_elision:
                            for closing in range(first, len(text)):
                                char = text[closing]
                                if char in "\r\n\u0085\u2028\u2029":
                                    break
                                if char not in {close_char, text[opening]}:
                                    continue
                                before = text[closing - 1] if closing > first else " "
                                after = text[closing + 1:closing + 2]
                                after_is_word = bool(after) and (
                                    letter(after) or after.isalnum() or after == "_"
                                    or (after in "-\u2010\u2011" and closing + 2 < len(text) and letter(text[closing + 2]))
                                )
                                if char == close_char and not before.isspace() and not after_is_word:
                                    # A distant apostrophe after a word could
                                    # instead be a plural possessive. Only a
                                    # punctuated utterance establishes a pair
                                    # beyond the name's own closing edge.
                                    if closing in {b - 1, b} or (closing > b and before in ".,!?"):
                                        if a == opening:
                                            a += 1
                                        if b - 1 == closing:
                                            b -= 1
                                    break
                                if char == text[opening] and (before.isspace() or before in "([{=:;") and after_is_word:
                                    # A new opener means the original mark was
                                    # not demonstrably paired; keep it intact.
                                    break
            if b > a:
                confidence = (
                    scores[start : i + 1]
                    .softmax(-1)[
                        torch.arange(i - start + 1), torch.tensor(labels[start : i + 1])
                    ]
                    .mean()
                    .item()
                )
                spans.append(Span(a, b, tag[2:], confidence, "encoder"))
            start = None
    # A UTF-8 character can occupy several tokenizer tokens. Projecting legal
    # token spans to characters can therefore create duplicates/overlaps.
    merged = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if merged and span.start < merged[-1].end:
            prior = merged.pop()
            preferred = span if span.score > prior.score else prior
            merged.append(
                Span(
                    prior.start,
                    max(prior.end, span.end),
                    preferred.label,
                    max(prior.score, span.score),
                    "encoder",
                )
            )
        else:
            merged.append(span)
    return merged


class Detector:
    def __init__(
        self,
        model_dir="models/clinical-adapter",
        *,
        backend="hybrid",
        device="auto",
        base_dir=None,
        batch_size=8,
    ):
        if backend not in {"model", "hybrid"}:
            raise ValueError("backend must be model or hybrid")
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        if base_dir is None:
            base_dir = self.config["base_dir"]
            if not Path(base_dir).is_dir():
                base_dir = self.model_dir.resolve().parent / "privacy-filter-base"
        self.backbone, self.device = load_backbone(base_dir, device)
        self.adapter = ClinicalAdapter(**self.config["adapter"]).to(self.device)
        from safetensors.torch import load_file

        self.adapter.load_state_dict(
            load_file(str(self.model_dir / "adapter.safetensors"))
        )
        self.adapter.eval()
        self.encoding = get_encoding(base_dir)
        self.backend = backend
        self.window = self.config.get("window", 512)
        self.batch_size = batch_size

    def detect(self, text):
        return self.detect_batch([text])[0]

    def detect_batch(self, texts):
        jobs = []
        document_offsets = []
        document_logits = []
        for doc, text in enumerate(texts):
            ids, offsets = tokenize(text, self.encoding)
            document_offsets.append(offsets)
            document_logits.append(
                torch.empty((len(ids), len(TAGS)), dtype=torch.float32)
            )
            # Overlap and keep the center of each window; no silent truncation.
            overlap = min(128, self.window // 4)
            stride = self.window - 2 * overlap
            for start in range(0, len(ids), stride):
                stop = min(len(ids), start + self.window)
                left = 0 if start == 0 else start + overlap
                right = len(ids) if stop == len(ids) else stop - overlap
                jobs.append(
                    (doc, ids[start:stop], offsets[start:stop], left, right, start)
                )
                if stop == len(ids):
                    break
        with torch.inference_mode():
            for a in range(0, len(jobs), self.batch_size):
                batch = jobs[a : a + self.batch_size]
                size = max(len(j[1]) for j in batch)
                tokens = torch.full(
                    (len(batch), size), 199999, dtype=torch.long, device=self.device
                )
                mask = torch.zeros_like(tokens, dtype=torch.bool)
                for i, j in enumerate(batch):
                    tokens[i, : len(j[1])] = torch.tensor(j[1], device=self.device)
                    mask[i, : len(j[1])] = True
                features = encode_backbone(self.backbone, tokens, mask)
                logits = self.adapter(features, mask)
                for i, j in enumerate(batch):
                    doc, _, offsets, left, right, window_start = j
                    # Merge emissions by token ownership, then decode the entire
                    # document once. A span can cross any number of windows.
                    document_logits[doc][left:right] = (
                        logits[i, left - window_start : right - window_start]
                        .float()
                        .cpu()
                    )
        results = [
            decode_bioes(logits, offsets, text)
            for logits, offsets, text in zip(document_logits, document_offsets, texts)
        ]
        if self.backend == "hybrid":
            from .rules import augment

            results = [augment(t, s) for t, s in zip(texts, results)]
        return results
