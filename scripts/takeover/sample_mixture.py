"""Draw a small, diverse training sample from a list of training banks.

A uniform sample of the mixture is dominated by the largest banks and drops
rare templates, layouts and the small targeted curricula. This sampler:

1. gives each bank a quota proportional to the square root of its size
   (capped at the bank size), so small banks keep a real share;
2. inside a bank, greedily picks the unit (a row, or a contrastive pair sharing
   ``pair_id``) that adds the most not-yet-covered features: template family,
   layout, cohort, phenomena, entity labels and PHI-free status; once nothing
   new is left it fills the quota round-robin across template families;
3. writes one training corpus with a manifest, and a coverage report comparing
   the sample with the full mixture.

Deterministic for a given seed and input bytes.

  python scripts/takeover/sample_mixture.py --rows 10000 --seed 20261005 \
      --output data/mix6s10k_train.jsonl --report reports/takeover/mix6s10k_001/sample_report.json \
      data/v12_train__....jsonl data/tk_train_admin__....jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = (
    "NAME",
    "DATE",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "MRN",
    "SSN",
    "ACCOUNT",
    "ID",
    "URL",
    "IP",
    "AGE",
    "LOCATION",
    "FACILITY",
)
PROTECTED = (
    "blind_",
    "qualification",
    ".sealed",
    "proxy",
    "retired",
    "validation",
    "development",
    "heldout_",
    "_dev",
    "test",
)


def load_bank(path: Path) -> tuple[list[dict], str]:
    if (
        any(word in path.name.lower() for word in PROTECTED)
        or "_train" not in path.name
    ):
        raise ValueError(f"refusing non-training corpus: {path.name}")
    payload = path.read_bytes()
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    sha = hashlib.sha256(payload).hexdigest()
    if manifest.get("sha256") != sha:
        raise ValueError(f"{path.name} differs from its manifest")
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"{path.name}: non-training row {row.get('id')}")
    return rows, sha


def features(row: dict) -> set[tuple[str, str]]:
    out = {
        ("family", str(row.get("template_family") or row.get("family"))),
        ("layout", str(row.get("layout") or row.get("authoring_layout"))),
        ("cohort", str(row.get("cohort"))),
        ("phi_free", str(not row["spans"])),
    }
    out.update(("phenomenon", str(p)) for p in row.get("phenomena") or [])
    out.update(("label", s["label"]) for s in row["spans"])
    return out


def units(rows: list[dict]) -> list[list[dict]]:
    """Group contrastive pairs so both members are kept or dropped together."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row.get("pair_id") or f"row:{i}")].append(row)
    return list(groups.values())


def quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    if total >= sum(sizes.values()):
        return dict(sizes)
    weights = {name: math.sqrt(n) for name, n in sizes.items()}
    out = dict.fromkeys(sizes, 0)
    remaining, open_banks = total, set(sizes)
    while remaining and open_banks:
        scale = remaining / sum(weights[b] for b in open_banks)
        changed = False
        for bank in sorted(open_banks):
            share = min(
                sizes[bank] - out[bank], max(1, int(weights[bank] * scale)), remaining
            )
            out[bank] += share
            remaining -= share
            if out[bank] >= sizes[bank]:
                open_banks.discard(bank)
                changed = True
        if not changed and remaining:
            for bank in sorted(open_banks, key=lambda b: -weights[b])[:remaining]:
                out[bank] += 1
                remaining -= 1
    return out


def select(bank_rows: list[dict], quota: int, rng: random.Random) -> list[dict]:
    pool = units(bank_rows)
    rng.shuffle(pool)
    feats = [set().union(*(features(r) for r in unit)) for unit in pool]
    covered: set = set()
    chosen: list[int] = []
    taken = 0
    heap = [(-len(f), i) for i, f in enumerate(feats)]
    heapq.heapify(heap)
    used = [False] * len(pool)
    while heap and taken < quota:
        negative_gain, i = heapq.heappop(heap)
        gain = len(feats[i] - covered)
        if gain == 0:
            break
        if gain < -negative_gain:
            heapq.heappush(heap, (-gain, i))
            continue
        if taken + len(pool[i]) > quota and taken:
            used[i] = True
            continue
        chosen.append(i)
        used[i] = True
        covered |= feats[i]
        taken += len(pool[i])
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, unit in enumerate(pool):
        if not used[i]:
            by_family[
                str(unit[0].get("template_family") or unit[0].get("family"))
            ].append(i)
    families = list(by_family)
    rng.shuffle(families)
    while taken < quota and any(by_family.values()):
        for family in families:
            if taken >= quota:
                break
            if by_family[family]:
                i = by_family[family].pop()
                if taken + len(pool[i]) > quota:
                    continue
                chosen.append(i)
                taken += len(pool[i])
    return [row for i in chosen for row in pool[i]]


def profile(rows: list[dict]) -> dict:
    labels = Counter(s["label"] for r in rows for s in r["spans"])
    spans = sum(labels.values())
    feats = Counter(k for r in rows for k, _ in features(r))
    distinct = defaultdict(set)
    for r in rows:
        for k, v in features(r):
            distinct[k].add(v)
    return {
        "rows": len(rows),
        "spans": spans,
        "phi_free_rows": sum(not r["spans"] for r in rows),
        "label_support": {label: labels[label] for label in LABELS},
        "label_share": {label: labels[label] / spans for label in LABELS},
        "distinct": {k: len(v) for k, v in distinct.items()},
        "feature_rows": dict(feats),
    }, distinct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.output,
        args.output.with_suffix(".jsonl.manifest.json"),
        args.report,
    ):
        if path.exists():
            raise FileExistsError(path)
    if "_train" not in args.output.name or any(
        w in args.output.name.lower() for w in PROTECTED
    ):
        raise ValueError("output name must mark a training corpus")
    banks, sources = {}, []
    for path in args.corpora:
        rows, sha = load_bank(path)
        banks[path.name] = rows
        sources.append({"path": str(path), "sha256": sha, "rows": len(rows)})
    plan = quotas({name: len(rows) for name, rows in banks.items()}, args.rows)
    rng = random.Random(args.seed)
    sample, per_bank = [], {}
    for name, rows in banks.items():
        picked = select(rows, plan[name], rng)
        per_bank[name] = {
            "rows": len(rows),
            "quota": plan[name],
            "sampled": len(picked),
        }
        sample.extend(picked)
    ids = [r["id"] for r in sample]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row ids across banks")
    full_rows = [r for rows in banks.values() for r in rows]
    full, full_distinct = profile(full_rows)
    part, part_distinct = profile(sample)
    coverage = {k: len(part_distinct[k]) / len(v) for k, v in full_distinct.items()}
    missing = {
        k: sorted(v - part_distinct[k])[:50]
        for k, v in full_distinct.items()
        if v - part_distinct[k]
    }
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample).encode(
        "utf-8"
    )
    manifest = {
        "annotation_policy": "clinical-phi-v2",
        "split": "train",
        "corpus": str(args.output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(sample),
        "documents": len(sample),
        "spans": part["spans"],
        "label_support": part["label_support"],
        "sampling": {
            "script": "scripts/takeover/sample_mixture.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": args.seed,
            "target_rows": args.rows,
            "sources": sources,
            "per_bank": per_bank,
        },
    }
    args.output.write_bytes(payload)
    args.output.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "output": str(args.output),
        "sha256": manifest["sha256"],
        "seed": args.seed,
        "per_bank": per_bank,
        "full": full,
        "sample": part,
        "distinct_value_coverage": coverage,
        "values_missing_from_sample": missing,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(sample), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
