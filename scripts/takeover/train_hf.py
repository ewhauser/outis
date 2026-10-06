"""Fully fine-tune a Hugging Face encoder as a 57-tag BIOES PHI tagger.

Training windows match inference: content windows of (max_length - specials)
tokens, overlap 128, and only the tokens a window owns carry labels. After each
epoch the detector is evaluated on development selection banks; the best epoch
is kept. Detection is model-only; there is no rule layer.

  PYTHONPATH=src python scripts/takeover/train_hf.py --model-name answerdotai/ModernBERT-base \
      --train data/v12_train.jsonl data/remediation_v1_train.jsonl \
      --select data/v12_validation.jsonl data/remediation_v1_validation.jsonl \
      --output models/takeover-modernbert-base-001 --epochs 3 --lr 5e-5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

import torch  # noqa: E402

from clinical_deid.data.annotation_policy import POLICY_VERSION  # noqa: E402
from clinical_deid.detection.deberta_attention import apply_gather_free_attention  # noqa: E402
from clinical_deid.detection.decoding import TAGS  # noqa: E402
from clinical_deid.detection.hf_encoder import (  # noqa: E402
    ENCODER_KIND,
    LEADING_SEPARATORS,
    TRAILING_SEPARATORS,
    HFTokenDetector,
    encode_text,
    special_layout,
    trim_merged_punctuation,
    window_jobs,
)
from clinical_deid.evaluation.evaluate import (  # noqa: E402
    evaluate,
    load_verified_corpus,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

TAG_INDEX = {tag: i for i, tag in enumerate(TAGS)}
PROTECTED = ("blind_qualification_v6", "blind_qualification_v7", "blind_qualification_v9",
             "qualification_v9", ".sealed", "controls/validation.jsonl", "proxyq_",
             "v15_name_next_step_analysis_001/recovery_data_001")



def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_corpus_path(path, role):
    """Reject protected names before any corpus or sidecar bytes are read."""
    path = Path(path)
    for candidate in (path, path.resolve()):
        name = str(candidate).lower()
        if any(word in name for word in PROTECTED):
            raise SystemExit(f"refusing protected or held-out bank for {role}: {path}")
        if role == "train" and "retired_v8" in name:
            raise SystemExit(f"refusing development-only bank for training: {path}")


def check_corpus_metadata(metadata, role):
    if not isinstance(metadata, dict):
        raise SystemExit("corpus metadata must be an object")
    if (any(metadata.get(key) not in (None, False) for key in ("independent_final", "independent", "is_proxy"))
            or metadata.get("proxy_role")):
        raise SystemExit(f"refusing final/independent/proxy metadata for {role}")
    for key in ("split", "suite", "role", "bank_role", "dataset_role"):
        tokens = set(re.split(r"[^a-z0-9]+", str(metadata.get(key, "")).lower()))
        forbidden = {"final", "test", "independent", "proxy", "blind", "sealed", "qualification"}
        if tokens & forbidden:
            raise SystemExit(f"refusing final/test/independent/proxy metadata for {role}")


def read_corpus(path, role):
    check_corpus_path(path, role)
    sidecar = Path(str(path) + ".manifest.json")
    check_corpus_path(sidecar, role)
    # The shared loader reads the corpus first: establish its allowed role here.
    declaration_bytes = sidecar.read_bytes()
    declaration = json.loads(declaration_bytes)
    if not isinstance(declaration, dict):
        raise SystemExit("corpus manifest must be an object")
    check_corpus_metadata(declaration, role)
    rows, manifest = load_verified_corpus(path)
    if manifest != declaration or sidecar.read_bytes() != declaration_bytes:
        raise SystemExit("corpus manifest changed after metadata preflight")
    for row in rows:
        check_corpus_metadata(row, role)
        if role == "train" and row.get("annotation_policy") != POLICY_VERSION:
            raise SystemExit(f"{path}: training requires annotation policy {POLICY_VERSION}")
        if role == "train" and row.get("split") != "train":
            raise SystemExit(f"{path}: only train-split, non-final rows may be used for training")
    return rows, {"path": str(Path(path).resolve()), "sha256": manifest["sha256"],
                  "manifest_sha256": hashlib.sha256(declaration_bytes).hexdigest(), "rows": len(rows),
                  "annotation_policies": dict(Counter(row.get("annotation_policy", "unspecified") for row in rows)),
                  "splits": dict(Counter(row.get("split", "unspecified") for row in rows)),
                  "suites": dict(Counter(row.get("suite", "unspecified") for row in rows))}


def rows_digest(rows):
    """Pin the effective rows and their order without writing their contents."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def resume_provenance(source):
    """A weights-only restart does not restore optimizer/scheduler/RNG state."""
    if source is None:
        return {"mode": "pretrained", "test_used_for_training": False}
    path = Path(source)
    check_corpus_path(path, "resume")
    if not path.is_dir():
        raise SystemExit("--resume-weights must name a local hf directory")
    if (path.parent / "publication_in_progress.json").exists():
        raise SystemExit("resume source has an interrupted publication; use its complete step checkpoint")
    summary_path = path.parent / "training_summary.json"
    summary = None
    if summary_path.is_file():
        check_corpus_path(summary_path, "resume")
        summary = json.loads(summary_path.read_text())
    audit = summary.get("corpus_audit", {}) if isinstance(summary, dict) else {}
    artifacts = {}
    for candidate in sorted(path.rglob("*")):
        check_corpus_path(candidate, "resume")
        if candidate.is_file():
            artifacts[str(candidate.relative_to(path))] = sha256_file(candidate)
    return {"mode": "weights_only", "path": str(path.resolve()), "artifacts": artifacts,
            "training_summary_path": str(summary_path.resolve()) if summary is not None else None,
            "training_summary_sha256": sha256_file(summary_path) if summary is not None else None,
            "test_used_for_training": audit.get("test_used_for_training"),
            "prior_corpus_audit": audit or None,
            "optimizer_restored": False, "scheduler_restored": False, "rng_restored": False}


def save_evaluation(output, model, tokenizer, config, summary, improved):
    """Preserve a recoverable checkpoint before publishing ordinary root artifacts.

    A process killed during root publication leaves publication_in_progress.json
    and a complete checkpoint-local summary. Such an output must be recovered
    from that checkpoint before qualification; never infer a completed publish.
    """
    step = summary["last_evaluated_step"]
    checkpoint = output / f"step-{step:06d}"
    staging = output / f".step-{step:06d}.incomplete"
    staging.mkdir()
    model.save_pretrained(staging / "hf", safe_serialization=True)
    tokenizer.save_pretrained(staging / "hf")
    atomic_json(staging / "config.json", config)
    local_summary = dict(summary, checkpoint={"step": step, "path": str(checkpoint.resolve())})
    atomic_json(staging / "training_summary.json", local_summary)
    os.replace(staging, checkpoint)
    if improved:
        pending = output / ".hf.pending"
        previous = output / ".hf.previous"
        shutil.copytree(checkpoint / "hf", pending)
        marker = output / "publication_in_progress.json"
        atomic_json(marker, {"checkpoint": str(checkpoint.resolve()), "step": step})
        if previous.exists():
            shutil.rmtree(previous)
        if (output / "hf").exists():
            os.replace(output / "hf", previous)
        os.replace(pending, output / "hf")
        atomic_json(output / "config.json", config)
        atomic_json(output / "training_summary.json", summary)
        marker.unlink()
        if previous.exists():
            shutil.rmtree(previous)
    else:
        atomic_json(output / "training_summary.json", summary)


def align(text, spans, offsets):
    """Return BIOES tag ids per token plus alignment diagnostics."""
    owner = [-1] * len(text)
    for k, span in enumerate(spans):
        for c in range(span["start"], span["end"]):
            owner[c] = k
    token_owner, conflicts = [], 0
    for a, b in offsets:
        owners = {owner[c] for c in range(a, b) if c < len(text) and owner[c] >= 0 and not text[c].isspace()}
        if not owners and text[a:b].isspace():
            # A whitespace-only token wholly inside one entity must retain that
            # entity's ownership, otherwise B/I, O, I/E becomes an illegal path.
            # Never extend PHI ownership into whitespace outside the gold span.
            whitespace_owners = {owner[c] for c in range(a, min(b, len(text)))}
            if len(whitespace_owners) == 1 and -1 not in whitespace_owners:
                owners = whitespace_owners
        if len(owners) > 1:
            conflicts += 1
        token_owner.append(min(owners) if owners else -1)
    tags = [TAG_INDEX["O"]] * len(offsets)
    by_span = {}
    for i, k in enumerate(token_owner):
        if k >= 0:
            by_span.setdefault(k, []).append(i)
    misaligned = 0
    for k, idx in by_span.items():
        label = spans[k]["label"]
        if len(idx) == 1:
            tags[idx[0]] = TAG_INDEX[f"S-{label}"]
        else:
            tags[idx[0]] = TAG_INDEX[f"B-{label}"]
            for i in idx[1:-1]:
                tags[i] = TAG_INDEX[f"I-{label}"]
            tags[idx[-1]] = TAG_INDEX[f"E-{label}"]
        a, b = offsets[idx[0]][0], offsets[idx[-1]][1]
        while a < b and (text[a].isspace() or text[a] in LEADING_SEPARATORS):
            a += 1
        while b > a and (text[b - 1].isspace() or text[b - 1] in TRAILING_SEPARATORS):
            b -= 1
        a, b = trim_merged_punctuation(text, a, b)
        if (a, b) != (spans[k]["start"], spans[k]["end"]):
            misaligned += 1
    missing = len(spans) - len(by_span)
    return tags, {"conflict_tokens": conflicts, "misaligned_spans": misaligned, "unrepresented_spans": missing}


def build_windows(rows, tokenizer, max_length, stats):
    prefix, suffix = special_layout(tokenizer)
    content = max_length - len(prefix) - len(suffix)
    windows = []
    for row in rows:
        text, spans = row["text"], sorted(row["spans"], key=lambda s: s["start"])
        ids, offsets = encode_text(tokenizer, text)
        if not ids:
            continue
        tags, diag = align(text, spans, offsets)
        stats.update(diag)
        stats["spans"] += len(spans)
        stats["documents"] += 1
        stats["tokens"] += len(ids)
        for start, stop, left, right in window_jobs(len(ids), content):
            labels = [-100] * (stop - start)
            for i in range(left, right):
                labels[i - start] = tags[i]
            windows.append((prefix + ids[start:stop] + suffix,
                            [-100] * len(prefix) + labels + [-100] * len(suffix)))
    stats["windows"] = len(windows)
    return windows


def batches(windows, batch_size, rng):
    order = list(range(len(windows)))
    rng.shuffle(order)
    chunk = batch_size * 64
    out = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: len(windows[i][0]))
        out.extend(part[b:b + batch_size] for b in range(0, len(part), batch_size))
    rng.shuffle(out)
    return out


def collate(windows, idx, pad_id):
    size = max(len(windows[i][0]) for i in idx)
    ids = torch.full((len(idx), size), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), size), -100, dtype=torch.long)
    mask = torch.zeros((len(idx), size), dtype=torch.long)
    for r, i in enumerate(idx):
        tokens, tags = windows[i]
        ids[r, :len(tokens)] = torch.tensor(tokens)
        labels[r, :len(tags)] = torch.tensor(tags)
        mask[r, :len(tokens)] = 1
    return ids, mask, labels


class Precomputed:
    def __init__(self, table):
        self.table = table

    def detect(self, text):
        return self.table[text]

    def detect_batch(self, texts):
        return [self.table[t] for t in texts]


def evaluate_banks(model, tokenizer, config, banks, device, eval_batch):
    model.eval()
    detector = HFTokenDetector(None, device=device, batch_size=eval_batch,
                               model=model, tokenizer=tokenizer, config=config)
    report = {}
    for name, rows in banks.items():
        texts = [row["text"] for row in rows]
        raw = {}
        for o in range(0, len(texts), 16):
            chunk = texts[o:o + 16]
            for text, spans in zip(chunk, detector.raw_batch(chunk)):
                raw[text] = spans
        entry = {}
        for backend, table in (("model", raw),):
            result = evaluate(rows, Precomputed(table), max_failures=0)
            entry[backend] = {
                "typed_span": result["typed_span"],
                "character_recall": result["character"]["recall"],
                "gates": result["goal"]["checks"],
                "passed": result["goal"]["passed"],
                "label_recall": {lab: result["per_label"][lab]["recall"] for lab in ENTITY_TYPES},
                "label_fp": {lab: result["per_label"][lab]["false_positives"] for lab in ENTITY_TYPES},
            }
        report[name] = entry
    model.train()
    return report


def selection_score(report):
    values = []
    for entry in report.values():
        ts = entry["model"]["typed_span"]
        values.append(min(ts["precision"], ts["recall"]))
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--select", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-select", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=float, default=1.0, help="evaluate every N epochs (fractional allowed)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume-weights", help="weights-only restart from a local hf/ directory; resets optimizer, schedule, RNG and step count")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    args = parser.parse_args()
    training_script_sha256 = sha256_file(Path(__file__))

    for name in ("epochs", "lr", "batch_size", "grad_accum", "max_length", "eval_batch", "eval_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit_train", "limit_select", "max_steps"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup < 1:
        parser.error("--warmup must be in [0, 1)")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output {output} already exists")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"

    def log(event):
        event["time"] = time.time()
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    initialization = resume_provenance(args.resume_weights)
    if args.resume_weights is None and Path(args.model_name).exists():
        # A local starting model may already have been fine-tuned.
        initialization = resume_provenance(args.model_name)
    train_rows, corpora, origins = [], [], {}
    for path in args.train:
        rows, corpus = read_corpus(path, "train")
        origins.update((id(row), len(corpora)) for row in rows)
        corpora.append(corpus)
        train_rows.extend(rows)
    rng.shuffle(train_rows)
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    used_counts = Counter(origins[id(row)] for row in train_rows)
    for i, corpus in enumerate(corpora):
        corpus["rows_used"] = used_counts[i]
    select_banks, select_corpora = {}, []
    for path in args.select:
        rows, corpus = read_corpus(path, "selection")
        if args.limit_select:
            rows = rows[:args.limit_select]
        name = Path(path).stem
        if name in select_banks:
            raise SystemExit(f"duplicate selection bank name: {name}")
        if not rows:
            raise SystemExit(f"selection bank has no rows: {path}")
        select_banks[name] = rows
        select_corpora.append(dict(corpus, rows_used=len(rows), effective_rows_sha256=rows_digest(rows)))
    train_texts = {row["text"] for row in train_rows}
    overlap = sum(1 for rows in select_banks.values() for row in rows if row["text"] in train_texts)
    if overlap:
        raise SystemExit(f"{overlap} selection texts also occur in training data")
    corpus_audit = {"test_used_for_training": initialization["test_used_for_training"],
                    "current_run_test_used_for_training": False,
                    "train": corpora, "selection": select_corpora,
                    "effective_train_rows": len(train_rows), "effective_train_rows_sha256": rows_digest(train_rows),
                    "selection_text_overlap": 0, "selection_overlap_scope": "current_run_training_rows",
                    "sealed_overlap_audit": "not_performed_by_this_script"}

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    source = args.resume_weights or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    # Pin fp32: transformers keeps a checkpoint's stored dtype, and some base
    # models (microsoft/deberta-v3-base) are stored in fp16.
    model = AutoModelForTokenClassification.from_pretrained(
        source, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id=TAG_INDEX,
        ignore_mismatched_sizes=bool(args.resume_weights is None), dtype=torch.float32)
    # DeBERTa-v2/v3 only: same outputs and gradients, without the slow gather.
    gather_free_attention = apply_gather_free_attention(model)
    device = torch.device(args.device)
    model.to(device)
    config = {"encoder": ENCODER_KIND, "base_model": args.model_name, "max_length": args.max_length,
              "tags": TAGS, "window_overlap": 128}

    stats = Counter()
    t0 = time.time()
    windows = build_windows(train_rows, tokenizer, args.max_length, stats)
    log({"event": "data", "train_rows": len(train_rows), "alignment": dict(stats),
         "misaligned_rate": stats["misaligned_spans"] / max(1, stats["spans"]),
         "seconds": time.time() - t0})

    if not windows:
        raise SystemExit("training corpus produced no token windows")

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(windows) / args.batch_size / args.grad_accum)
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    warmup = int(total_steps * args.warmup)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eval_interval = max(1, int(steps_per_epoch * args.eval_every))
    log({"event": "plan", "windows": len(windows), "steps_per_epoch": steps_per_epoch,
         "total_steps": total_steps, "warmup": warmup, "eval_interval": eval_interval})

    model.train()
    step, best, history = 0, None, []
    running, running_batches, seen_tokens, started = 0.0, 0, 0, time.time()
    epoch = 0

    def make_summary(status):
        return {"encoder": ENCODER_KIND, "base_model": args.model_name, "arguments": vars(args),
                "alignment": dict(stats), "best": best, "history": history,
                "training_script_sha256": training_script_sha256,
                "corpus_audit": corpus_audit, "initialization": initialization,
                "status": status, "last_evaluated_step": history[-1]["step"] if history else None,
                "completed_steps": step, "planned_steps": total_steps,
                "selection_metric": "mean_bank_min_model_precision_recall" if select_banks else "latest_checkpoint_no_selection",
                "device": args.device, "torch": torch.__version__,
                "deberta_gather_free_attention": gather_free_attention}

    atomic_json(output / "training_summary.json", make_summary("training"))
    done = False
    while not done:
        epoch += 1
        epoch_batches = batches(windows, args.batch_size, rng)
        for batch_index, idx in enumerate(epoch_batches):
            ids, mask, labels = collate(windows, idx, pad_id)
            if args.amp == "bf16":
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            else:
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            # Flush the final partial accumulation group at each epoch boundary.
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(epoch_batches) - group_start)
            loss = out.loss / group_size
            loss.backward()
            running += float(out.loss.detach())
            seen_tokens += int(mask.sum())
            running_batches += 1
            if (batch_index + 1) % args.grad_accum and batch_index + 1 < len(epoch_batches):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - started
                log({"event": "train", "step": step, "epoch": epoch, "loss": running / running_batches,
                     "lr": scheduler.get_last_lr()[0], "tokens_per_s": seen_tokens / elapsed, "elapsed": elapsed})
                running, running_batches = 0.0, 0
            if step % eval_interval == 0 or step >= total_steps:
                report = evaluate_banks(model, tokenizer, config, select_banks, args.device, args.eval_batch) if select_banks else {}
                score = selection_score(report)
                history.append({"step": step, "epoch_fraction": step / steps_per_epoch, "score": score, "report": report})
                log({"event": "eval", "step": step, "score": score,
                     "summary": {k: {b: {m: round(v["typed_span"][m], 5) for m in ("precision", "recall", "f1")}
                                     for b, v in e.items()} for k, e in report.items()}})
                improved = best is None or not select_banks or score > best["score"]
                if improved:
                    best = {"step": step, "score": score, "checkpoint": f"step-{step:06d}"}
                save_evaluation(output, model, tokenizer, config, make_summary("training"), improved)
            if step >= total_steps:
                done = True
                break
    atomic_json(output / "training_summary.json", make_summary("completed"))
    log({"event": "done", "best": best})


if __name__ == "__main__":
    main()
