# External verification

External verification measures the detector on public benchmarks that this
project did not author, and compares it with other open de-identification
models under one fixed protocol. It complements the internal
[evaluation protocol](../reports/evaluation_protocol.md), which qualifies the
detector on independently authored synthetic banks.

The project's acceptance gate is the goal in [GOALS.md](../GOALS.md), which
measures whether identifiers stay masked under any label. External results are
evidence of how the detector transfers to someone else's text and annotation
conventions, and each scored result should include that any-label coverage.
They are never used to select, tune or train a model.

GOALS.md ultimately requires evaluation on real notes. That is deferred. Until
the owner reopens it, do not start access requests for benchmarks that contain
real clinical text, such as n2c2 2014.

The harness lives in
[`experiments/external_benchmarks_20261004/`](../experiments/external_benchmarks_20261004/README.md).
That README records the current results. This guide covers the process.

## Rules

These rules hold for every benchmark and every system.

1. **Evaluation only.** No external corpus, prediction or failure example may
   enter a training set, an augmentation generator prompt, a checkpoint-selection
   set or a threshold sweep. A benchmark whose failures have been used to guide
   a fix is development evidence from then on, and later reports must say so.
2. **Licence first.** A benchmark is added only after its licence is confirmed
   to allow this organisation's use. Record the licence and its source in the
   benchmark table below and in the converter.
3. **Declare before scoring.** Label mappings, thresholds, model revisions and
   any added configuration arms are fixed in code before predictions are scored.
   A changed mapping is a new run, not an edit of the old one.
4. **Pin everything.** Datasets are pinned by SHA-256 and models by revision.
   Each prediction file has a receipt, and the scorer rejects predictions whose
   receipt does not match the corpus and file hashes.
5. **Third-party data stays out of git.** Downloads, converted corpora and raw
   predictions are gitignored. Manifests, receipts and scored results are
   committed.
6. **Report honestly.** Report every declared system and arm, including the
   ones that look bad. State each benchmark's scope; a synthetic query set is
   not evidence of accuracy on real clinical notes.

## Benchmarks

| Benchmark | Content | Licence | Commercial use | Status |
|---|---|---|---|---|
| [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1) | 1,051 synthetic clinical search queries, 2,973 PHI values, 219 hard negatives | MIT | Yes | Scored; development evidence since 2026-10-04 (guided a hybrid fix) |
| [SPY](https://huggingface.co/datasets/mks-logic/SPY) medical | 4,491 synthetic medical consultations, 7 PII types | CC BY 4.0 | Yes, with attribution | Planned |
| [n2c2 2014](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/) de-identification | Real clinical notes with surrogate PHI | Harvard DBMI data use agreement | Evaluation only, under the commercial agreement | Planned; access required |
| [SHIELD](https://stanford.redivis.com/datasets/ee4n-6gv2f84et) | 1,381 Stanford notes, 9 PHI categories | Stanford DRUA | Unconfirmed | Blocked on licence review |
| PhysioNet deidentified medical text | 2,434 MIMIC-II nursing notes | PhysioNet credentialed licence | No; research only | Excluded |

Do not use n2c2 2014 until the agreement is signed. Its terms forbid putting
the data into a product, which rules 1 and 5 already respect. Check the
agreement's publication clause before quoting results that name it.

## Systems

| Key | Model | Licence | Runtime |
|---|---|---|---|
| `project-mix5` | default ModernBERT mix5 checkpoint, model-only | Apache-2.0 | project `.venv` |
| `project`, `project-model` | removed Privacy-Filter adapter (hybrid, model-only); history | Apache-2.0 | project `.venv` |
| `meddeid` | `stighellemans/meddeid-english-synth` | AGPL-3.0 with a private fine-tuning exception | `.venv-meddeid` |
| `gliner-trained`, `gliner-facility` | `nvidia/gliner-PII` | NVIDIA Open Model License | `.venv-gliner` |

External models run only locally, for evaluation. Their weights and code are
never vendored, modified or served. MedDeID's AGPL licence makes this
distinction important: running the model for measurement is fine, but
building on its weights needs a separate licence review.

## How the harness works

```
download ──▶ converter ──▶ corpora/<benchmark>.jsonl + manifest
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
 predict_project.py        predict_meddeid.py           predict_gliner.py
   (project .venv)          (.venv-meddeid)              (.venv-gliner)
        │                          │                           │
        └──────▶ predictions/<benchmark>.<system>.jsonl + receipt ◀──┘
                                   │
                                   ▼
                    score.py ──▶ results/<benchmark>.<system>.json
                                 results/<benchmark>.summary.md
```

Each external model has its own virtual environment so its dependencies can
never change the project runtime. Prediction and scoring are separate steps,
so a system is run once and can be rescored without inference.

| File | Role |
|---|---|
| `common.py` | Standard-library helpers shared by every runtime: hashing, JSONL, corpus loading, windowing, prediction receipts. Must not import the project or third-party packages. |
| `labels.py` | Every gold and prediction label mapping, and the coarse comparison groups. |
| `<benchmark>.py` | Download, verify and convert one benchmark. |
| `predict_<system>.py` | Run one system and write native-label predictions. |
| `score.py` | Map labels, compute metrics, write results and the summary table. |
| `test_external.py` | Tests for conversion, windowing, merging, age rules and label mapping. |

### Corpus format

A converted corpus is JSONL with one document per line, accepted by both the
project evaluator and `load_verified_corpus`:

```json
{"id": "asq-0001", "text": "...", "family": "phi",
 "spans": [{"start": 72, "end": 79, "label": "NAME", "source_label": "NAME", "text": "Anna S."}],
 "source_tags": [{"identifier_type": "NAME", "value": "Anna S."}]}
```

Offsets are Unicode character offsets into `text`. `label` must be a project
entity type; `source_label` keeps the benchmark's own label. `family` groups
documents for reporting, for example `phi` and `hard_negative`.

The adjacent `<benchmark>.jsonl.manifest.json` records the source URL, DOI,
licence and pinned hash; the corpus SHA-256; row count; support for every
project entity type; total, gold and non-gold character counts; alignment
statistics; and the converter's own hash.

### Prediction format and receipts

Prediction files are JSONL with one record per document, in corpus order, with
**native** labels:

```json
{"id": "asq-0001", "spans": [{"start": 72, "end": 76, "label": "first_name", "score": 0.91, "text": "Anna"}]}
```

The receipt beside each file records the corpus hash, predictions hash, model
repository and revision, every configuration value, runtime, platform and the
runner's hash. For the project detector it also records adapter artefact and
detector source hashes, and the run fails if either changes during inference.

## Metrics

Benchmarks and systems use different label schemes, so the headline metrics
ignore labels and measure what a system would actually remove. This follows the
MedDeID paper, which lets its published ASQ-PHI figures serve as a check on
the scorer.

| Metric | Definition |
|---|---|
| Character recall | Gold PHI characters covered by any prediction, over all gold PHI characters. |
| Non-PII redaction | Predicted characters outside gold, over all non-gold characters. Lower is better. |
| Age-adjusted non-PII redaction | As above, with explicit ages below 90 removed from the numerator, for benchmarks that leave those ages unannotated. The rule is `score.age_characters`. |
| Exact span P/R/F1 | Predicted boundaries equal to gold boundaries, ignoring labels. |
| Leak-free PHI documents | Documents with PHI in which every gold character is covered. |
| Clean hard negatives | Documents without PHI that receive no prediction at all. |
| Coarse typed P/R/F1 | Exact boundaries and coarse group, scored by `clinical_deid.evaluation.evaluate.evaluate`, with per-group recall and Wilson bounds. |

Every prediction counts towards the label-agnostic metrics, including labels
that map to no project type (for example MedDeID's `Profession`), because the
system would still remove that text. Only the typed metrics drop them, and the
result JSON lists how many were dropped.

### Coarse groups

Typed scoring maps both gold and predictions to these groups, defined in
`labels.COARSE`:

| Group | Project types | Why |
|---|---|---|
| NAME | NAME | |
| DATE | DATE | |
| LOCATION | ADDRESS, LOCATION, FACILITY | ASQ-PHI tags hospitals as geographic locations |
| ID | MRN, SSN, ACCOUNT, ID | MedDeID has one ID family; benchmarks differ on identifier subtypes |
| PHONE, EMAIL, URL, IP, AGE | themselves | |

### Normalisations applied to every system

- **Adjacent names merge.** NAME spans separated only by whitespace become one
  span, because some label sets split first and last names. Without this, the
  space between "Anna" and "S." would count as leaked PHI.
- **Duplicates collapse.** Identical (start, end, label) predictions keep the
  highest score.
- **Long texts are windowed.** Runners for models with input limits split text
  into overlapping character windows (`common.windows`) and map offsets back.

No other post-processing is allowed in a runner. A system-specific fix belongs
in a declared configuration arm.

## Running

Commands run from the experiment directory. The external environments are
created once:

```sh
cd experiments/external_benchmarks_20261004
uv venv --python 3.13 .venv-gliner
uv pip install --python .venv-gliner/bin/python gliner==0.2.29
uv venv --python 3.13 .venv-meddeid
uv pip install --python .venv-meddeid/bin/python meddeid==0.4.2
```

Then convert, predict and score:

```sh
../../.venv/bin/python asq_phi.py
../../.venv/bin/python predict_project.py --benchmark asq_phi
.venv-meddeid/bin/python predict_meddeid.py --benchmark asq_phi
.venv-gliner/bin/python predict_gliner.py --benchmark asq_phi --arm trained
.venv-gliner/bin/python predict_gliner.py --benchmark asq_phi --arm facility
../../.venv/bin/python score.py --benchmark asq_phi --systems project-mix5 meddeid gliner-trained gliner-facility
```

`predict_project.py` runs the default model-only checkpoint and writes the
`project-mix5` system. The `project`, `project-model` and
`project-facility-fix` prediction files came from the removed adapter detector
and are kept as history. Model weights are not in git; in a worktree, pass
`--model-dir` pointing at the main checkout's copy.

Set `HF_HOME` to a shared cache to avoid downloading external weights again.
On ASQ-PHI on Apple silicon CPU, MedDeID takes about 17 seconds, the project
detector about 3 minutes and each GLiNER arm about 5 minutes.

Run the tests after any change to the harness:

```sh
../../.venv/bin/python -m pytest -q test_external.py
```

## Adding a benchmark

1. Confirm the licence allows this organisation's use. Add a row to the
   benchmark table here, with a link to the licence source.
2. Write `<benchmark>.py` with a pinned URL and SHA-256. Downloads go to
   `downloads/`, which is gitignored.
3. Map every source label to a project type in `labels.py`. An unknown source
   label must raise an error, never fall through silently.
4. Convert to the corpus format and write the manifest with
   `label_support` for every project entity type. Check the converted counts
   against the benchmark's published statistics and fail if they differ.
5. Record alignment or conversion losses in the manifest. Keep a document
   whose annotation could not be aligned, and report the loss rather than
   dropping the document.
6. If the benchmark leaves some Safe Harbor-permitted text unannotated (bare
   years, ages below 90), say so in the experiment README and add an adjusted
   metric if it changes the comparison.
7. Add conversion tests to `test_external.py`, including at least one real
   edge case from the data.
8. Where an external system has published results on the benchmark, add them
   to `score.PUBLISHED` so the scorer is checked against them.

## Adding a system

1. Confirm the model licence allows local evaluation. Record it in the systems
   table.
2. Create a dedicated `.venv-<system>` and pin the package version.
3. Write `predict_<system>.py`. Pin the model revision as a constant, emit
   native labels and offsets, and pass every configuration value to
   `common.write_predictions` so it lands in the receipt.
4. Add the native label mapping to `labels.py` before scoring. Use `None` for
   labels that are not PHI under the project policy.
5. If the model card prescribes a configuration (a threshold, a label list),
   that is the primary arm. Additional arms get distinct system keys.

## Results: ASQ-PHI, 2026-10-04

ASQ-PHI v1 (MIT), 1,051 synthetic clinical search queries, 2,973 gold spans.
One run per system. The project and MedDeID have both now been looked at on
this set, so treat it as development evidence for both. The full tables are in
`experiments/external_benchmarks_20261004/results/asq_phi.summary.md`.

| System | Char recall | Non-PII redaction | Exact-span F1 | Leak-free PHI docs | Clean hard negatives |
|---|---:|---:|---:|---:|---:|
| `project-mix5` (default, model-only) | 97.49% | 0.33% | 92.13% | 79.57% | 73.06% |
| `project-model` (removed adapter, model-only) | 94.70% | 1.05% | 88.20% | 69.35% | 84.02% |
| `project` (removed adapter, hybrid) | 95.25% | 1.03% | 83.19% | 58.89% | 84.47% |
| `meddeid` | 99.14% | 6.33% | 79.79% | 93.27% | 14.61% |
| `gliner-facility` | 96.81% | 8.27% | 70.65% | 74.88% | 27.40% |

Any-label masking from the evaluator's `privacy` block, as scored, against
ASQ-PHI's own gold boundaries:

| System | Direct masked | NAME | Indirect masked | LOCATION | DATE | Any-label precision |
|---|---:|---:|---:|---:|---:|---:|
| `project-mix5` | 91.57% | 88.82% | 95.34% | 92.13% | 98.64% | 97.83% |
| `project-model` | 92.02% | 93.73% | 88.73% | 82.20% | 95.41% | 97.04% |
| `meddeid` | 97.24% | 97.30% | 98.59% | 98.55% | 98.64% | 74.66% |

Most of the mix5 shortfall is convention, not exposure:

- **NAME.** 90 of the 91 unmasked names miss only the honorific ("Dr.",
  "Mr."), which ASQ-PHI includes in gold and the project policy excludes.
  Counting those as masked, mix5 masks 99.88% of names, the adapter 98.40% and
  MedDeID 97.67%. The one real miss is a name glued to the next word.
- **ID and MRN.** Every alphanumeric character of every MRN is masked. The
  unmasked characters are "#" and field labels such as "Patient ID:", which
  ASQ-PHI puts inside the span.
- **DATE.** The misses are relative dates ("last week", "last month"), which
  the project policy does not annotate.
- **LOCATION is the real gap.** 17 of 826 locations get no prediction at all,
  almost all a city used as a modifier: "our Chicago clinic", "the NYC clinic",
  "Seattle clinic". A further 33 are partly covered, usually because gold
  joins a facility and city ("Mayo Clinic in Rochester, MN") and the connecting
  words are left.

Against the removed adapter, mix5 cuts non-PHI redaction by two thirds and
raises leak-free documents by ten points. It leaves more PHI-free hard
negatives with a span (73% clean against 84%).

### Policy-aligned view

`score_aligned.py` rescores the same predictions with the differences in
requirements neutralised. Where ASQ-PHI and the
[annotation policy](annotation-policy-v2.md) disagree about whether text must
be masked, those characters count neither as PHI nor as over-redaction, for
every system:

| Neutral text | Why |
|---|---|
| Spaces and punctuation | A leaked comma or "#" reveals nothing |
| Honorifics and credentials in NAME | The policy excludes them |
| Field labels before an identifier ("Patient ID:") | Not part of the value |
| Relative dates ("last week"), and "last" in "last December" | The policy does not annotate relative expressions |
| Lowercase and generic unit words in LOCATION ("clinic", "in", "ER") | The policy keeps prepositions and generic care-site words outside place spans |
| Ages up to 89 and bare years outside gold | Safe Harbor permits both; ASQ-PHI never annotates them, and the project masks a patient-specific year |

The rules read only the text and gold, never predictions, and are tested in
`test_external.py`. The output is `results/asq_phi.aligned.md`.

| System | Value-char recall | Non-PHI redaction | Direct masked | NAME (doc 95% lb) | Indirect masked | LOCATION | Leak-free PHI docs | Clean hard negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `project-mix5` | 99.50% | 0.19% | 99.85% | 99.88% (99.63%) | 98.64% | 97.33% | 97.12% | 94.06% |
| `project-model` (removed adapter) | 95.73% | 1.18% | 98.58% | 99.51% (98.90%) | 90.07% | 83.76% | 79.33% | 87.21% |
| `project` (removed adapter, hybrid) | 97.15% | 1.20% | 99.63% | 99.63% (99.14%) | 92.54% | 86.06% | 85.22% | 86.30% |
| `meddeid` | 99.66% | 0.91% | 99.55% | 99.51% (99.01%) | 99.08% | 98.55% | 97.60% | 88.13% |
| `gliner-facility` | 98.57% | 2.43% | 97.02% | 99.88% (99.51%) | 99.57% | 99.15% | 94.47% | 54.79% |

Under the project's requirements, mix5 and MedDeID mask about the same
(99.50% against 99.66% of value characters). Mix5 masks more direct identifiers
and leaves a fifth as much non-PHI text masked. MedDeID is ahead on LOCATION,
which is mix5's remaining gap. The one email miss shared by every system is a
gold error: ASQ-PHI tags the word "email". The neutral rules are the project's
own judgement, so quote this view next to the as-published one, never instead
of it.

## Restricted benchmark: n2c2 2014

The n2c2 (i2b2/UTHealth) 2014 de-identification dataset is real clinical text
with surrogate PHI, distributed under a data use agreement through the
[DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).
The owner approved its use for evaluation only on 2026-10-06 (AGENTS.md).

- Each person obtains it under their own agreement. Keep the raw files outside
  every git repository.
- `n2c2_2014.py --split test --raw <folder of XML files>` converts it into
  `$DEID_RESTRICTED_DIR/corpora/n2c2_2014_test.jsonl` (default
  `~/working/deid/restricted`). Predictions for restricted benchmarks are
  written under `$DEID_RESTRICTED_DIR/predictions/`. The harness refuses a
  restricted folder inside a git repository.
- `score.py` writes no failure examples for restricted benchmarks; results
  files hold aggregate scores only and may be committed.
- Native subtypes map to project types (`n2c2_2014.TYPE_MAP`); PROFESSION has
  no project type and is excluded from gold. The policy-aligned view
  (`score_aligned.py`) treats STATE, COUNTRY, DEPARTMENT, ORGANIZATION and ROOM
  spans, and gold ages of 89 or under, as neutral.
- Published n2c2 2014 results use specific matching rules (strict or relaxed
  spans, token or entity level, the full PHI set or a HIPAA subset). Match the
  metric exactly before comparing with any paper.

## Reading results

- **Lead with character recall and non-PII redaction together.** Either can be
  maximised alone; the pair shows the actual trade-off.
- **Check the scope.** ASQ-PHI and SPY are short synthetic texts. Only n2c2 2014
  (and SHIELD, if licensed) approach real clinical notes.
- **Expect convention differences.** The project policy excludes honorifics
  from NAME spans and treats ages up to 89 and bare years as hard negatives. A
  benchmark that includes "Dr." in gold names or never annotates years will
  move exact-span and hard-negative metrics without any PHI being exposed.
  Quantify a convention effect before attributing a gap to it.
- **Separate observation from action.** Failure analysis on an external set
  may motivate a new synthetic generator or policy change. Write that up as a
  development plan, and from then on report the benchmark as seen.

## Reporting

External results may be quoted with these qualifications:

- the benchmark name, version and licence;
- that the numbers are from a single run with the recorded revisions;
- the benchmark's scope (for example "synthetic clinical search queries");
- the label-agnostic metric definitions above, not the internal 99% gates.

Do not combine external results with the internal qualification results, and
do not describe external results as clinical or production accuracy.
