"""Write a seeded uniform replay sample of existing training banks.

For continuation runs: a weights-only restart trains on new data plus a
replay sample of the earlier mixture so the model keeps what it learned. The
sample is drawn uniformly over all rows of the listed banks (seeded), keeps
rows unchanged, and records every source bank's SHA-256 and the number of
rows drawn from it. Sources must already be audited training banks; the
sample adds no new text.

  python scripts/takeover/make_replay_sample.py --rows 20000 --seed 20261008 \
      --output data/replay_X.jsonl data/BANK_A.jsonl data/BANK_B.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROTECTED = ("blind_", "qualification", ".sealed", "proxy", "heldout_dev")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pool, sources = [], []
    for source in args.sources:
        if any(word in source.name.lower() for word in PROTECTED):
            raise ValueError(f"refusing protected bank: {source.name}")
        payload = source.read_bytes()
        manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
        digest = hashlib.sha256(payload).hexdigest()
        if manifest.get("sha256") != digest:
            raise ValueError(f"{source} differs from its manifest")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if any(r.get("split") != "train" for r in rows):
            raise ValueError(f"{source} has non-train rows")
        sources.append({"path": str(source), "sha256": digest, "rows": len(rows)})
        pool += [(str(source), r) for r in rows]
    if args.rows > len(pool):
        raise ValueError("sample larger than the pool")
    picked = random.Random(args.seed).sample(pool, args.rows)
    drawn = Counter(path for path, _ in picked)
    for entry in sources:
        entry["rows_drawn"] = drawn.get(entry["path"], 0)
    out = "".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row in picked).encode("utf-8")
    support = Counter(s["label"] for _, r in picked for s in r["spans"])
    manifest = {
        "corpus": str(args.output), "sha256": hashlib.sha256(out).hexdigest(),
        "rows": len(picked), "documents": len(picked), "split": "train",
        "annotation_policy": sorted({r.get("annotation_policy") for _, r in picked})[0]
        if len({r.get("annotation_policy") for _, r in picked}) == 1 else "mixed",
        "label_support": dict(sorted(support.items())), "entity_support": dict(sorted(support.items())),
        "independent_final": False, "role": "train",
        "replay_sample": {"script": "scripts/takeover/make_replay_sample.py",
                          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "seed": args.seed, "rows": args.rows, "pool_rows": len(pool), "sources": sources},
    }
    args.output.write_bytes(out)
    Path(str(args.output) + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(picked), "pool_rows": len(pool), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
