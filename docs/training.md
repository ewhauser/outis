# Training and reproducing the model

This guide covers training a detector candidate. To *use* the detector, see
the [project README](../README.md). The detector is a fully fine-tuned
DeBERTa-v3-base token classifier with a BIOES decoder; there is no rule layer.
All task-specific text is synthetic; no clinical dataset is used for training.

The default model is `models/deid-deberta-v3-base`
(epoch 2 of a three-epoch run). Its weights are on Hugging Face as
[ewhauser/outis-deberta-v3-base](https://huggingface.co/ewhauser/outis-deberta-v3-base).

## Environment

```sh
uv sync --group dev
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export HF_HOME="$PWD/.cache/huggingface"
```

Inference sends no text to any hosted service. On Apple silicon, set
`OUTIS_DEVICE=mps`; CPU and CUDA are supported.

## Training corpora

Training banks are JSONL files of synthetic notes with gold spans under the
[annotation policy](annotation-policy-v2.md). Each has a
`*.jsonl.manifest.json` with its SHA-256, row count and label support. The
corpora used for the default model are not stored in this repository; they
are published as [ewhauser/outis-synthetic-phi](https://huggingface.co/datasets/ewhauser/outis-synthetic-phi), with the 46
training and 5 selection banks as Parquet and as the original JSONL with
manifests. The evaluation banks are not published.

The generators in `scripts/takeover/author*.py` write the targeted banks
(name mentions, transcripts, lab and pathology, radiology, spoken PHI, FHIR,
care-site boundaries, word surnames, age contexts, named instruments and
others), and `outis.data.generators` holds the general generators.
Each generator is seeded and deterministic. The normalisers in
`scripts/takeover/normalize_*.py` rewrite older banks to the current policy,
for example `normalize_ages2.py` for the all-ages convention.

Generators that use real place names draw only from pinned public sources
(Census Gazetteer, CMS Hospital General Information and
`data/care_site_names.json`) and from the places addendum's `train` value
pool:

```sh
uv run -m outis.data.generators.place_sources fetch
uv run -m outis.data.generators.places generate
```

Byte-for-byte regeneration of the published corpora also needs the
evaluation banks' hash-only exclusion lists, which are not public, so
regenerated banks can differ slightly from the originals.

Before training, check every training bank for overlap with your evaluation
banks and remove matching rows. Never train on a bank you evaluate on.

## Training a candidate

`scripts/takeover/train_hf.py` fine-tunes the encoder, evaluates the selection
banks after each epoch and keeps the checkpoint chosen by the `privacy`
selection rule (direct-identifier recall first, then indirect recall, then
any-label precision, then exact typed-span scores). It writes the arguments,
corpus hashes, overlap check and per-epoch history to
`training_summary.json`. The default model was trained with:

```sh
uv run scripts/takeover/train_hf.py \
  --model-name microsoft/deberta-v3-base \
  --train data/TRAIN_1.jsonl data/TRAIN_2.jsonl ... \
  --select data/SELECT_1.jsonl data/SELECT_2.jsonl ... \
  --output models/NEW_RUN \
  --epochs 3 --lr 3e-05 --batch-size 32 --grad-accum 1 --max-length 512 \
  --warmup 0.06 --weight-decay 0.01 --eval-batch 8 --eval-every 1 \
  --seed 20261010 --device mps --amp none --selection-metric privacy
```

The full list of 46 training and 5 selection banks is in the model's
`training_summary.json`. `scripts/takeover/make_replay_sample.py` builds a
seeded replay sample for weights-only continuation runs (`--resume-weights`).

## Evaluation

```sh
uv run scripts/takeover/eval_bank.py \
  --data data/BANK.jsonl \
  --model-dir models/NEW_RUN/step-NNNNNN \
  --output reports/NEW_RUN/BANK.evaluation.json --device mps
```

Compare the candidate with the current model on a bank from an author the
training data never saw, using the rules in
[evaluation methodology](evaluation.md). Public benchmarks are scored with
the harness in `benchmarks/`; see
[external verification](external-verification.md).
