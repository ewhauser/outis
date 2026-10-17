# External verification

External verification measures the detector on public benchmarks that this
project did not author, and compares it with other open de-identification
models under one fixed protocol. It complements the internal evaluation on independently authored synthetic
banks described in [evaluation methodology](evaluation.md), which measures
whether identifiers stay masked under any label. External results are
evidence of how the detector transfers to someone else's text and annotation
conventions, and each scored result should include that any-label coverage.
They are never used to select, tune or train a model.

Evaluation on real notes is not part of this repository. Benchmarks that
contain real clinical text, such as n2c2 2014, are used only under their own
data use agreements and never committed.

The harness lives in
[`benchmarks/`](../benchmarks/README.md).
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
| [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1) | 1,051 synthetic clinical search queries, 2,973 PHI values, 219 hard negatives | MIT | Yes | Scored; development evidence (its failures were inspected) |
| [MedDeID English](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark) synthetic benchmark | 300 synthetic notes | CC BY 4.0; final evaluation only | Yes, with attribution | Scored; development evidence |
| [Technetium-I](https://huggingface.co/datasets/temlm-foundation/Technetium-I) test split | 74,700 synthetic notes; a fixed 5,000-note sample is scored | EUPL-1.2 | Yes | Scored; uninspected |
| [SPY](https://huggingface.co/datasets/mks-logic/SPY) medical | 4,491 synthetic medical consultations, 7 PII types | CC BY 4.0 | Yes, with attribution | Planned |
| [n2c2 2014](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/) de-identification | Real clinical notes with surrogate PHI | Harvard DBMI data use agreement | Evaluation only, under the commercial agreement | Planned; access required |
| [SHIELD](https://stanford.redivis.com/datasets/ee4n-6gv2f84et) | 1,381 Stanford notes, 9 PHI categories | Stanford DRUA | Unconfirmed | Blocked on licence review |
| [PhysioNet Deidentified Medical Text](https://physionet.org/content/deidentifiedmedicaltext/1.0/) v1.0 | 2,434 MIMIC-II nursing notes with surrogate PHI | PhysioNet Credentialed Health Data License 1.5.0 | No; research evaluation only | Scored; restricted, uninspected |

Do not use n2c2 2014 until the agreement is signed. Its terms forbid putting
the data into a product, which rules 1 and 5 already respect. Check the
agreement's publication clause before quoting results that name it.

## Systems

| Key | Model | Licence | Runtime |
|---|---|---|---|
| `project` | this project's default model (`models/deid-deberta-v3-base`) | Apache-2.0 | project environment (`uv run`) |
| `meddeid` | `stighellemans/meddeid-english-synth` | AGPL-3.0 with a private fine-tuning exception | `uv run --with meddeid==0.4.2` |
| `gliner-trained`, `gliner-facility` | `nvidia/gliner-PII` | NVIDIA Open Model License | `uv run --with gliner==0.2.29` |

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
   (project env)         (uv --with meddeid)          (uv --with gliner)
        │                          │                           │
        └──────▶ predictions/<benchmark>.<system>.jsonl + receipt ◀──┘
                                   │
                                   ▼
        score.py, score_aligned.py ──▶ results/<benchmark>.<system>.json
                                        results/<benchmark>.aligned3.json
                                               │
                                   compare.py ──▶ results/COMPARISON.md
```

Each external model runs in its own uv environment so its dependencies can
never change the project runtime. Prediction and scoring are separate steps,
so a system is run once and can be rescored without inference.

| File | Role |
|---|---|
| `common.py` | Standard-library helpers shared by every runtime: hashing, JSONL, corpus loading, windowing, prediction receipts. Must not import the project or third-party packages. |
| `labels.py` | Every gold and prediction label mapping, and the coarse comparison groups. |
| `<benchmark>.py` | Download, verify and convert one benchmark. |
| `predict_<system>.py` | Run one system and write native-label predictions. |
| `score.py` | Map labels and compute the as-published metrics. |
| `score_aligned.py` | Score every system under this project's annotation policy (view 3). |
| `compare.py` | Write `results/COMPARISON.md` from the result files. |
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
runner's hash. For the project detector it also records model artefact and
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
| Coarse typed P/R/F1 | Exact boundaries and coarse group, scored by `outis.evaluation.evaluate.evaluate`, with per-group recall and Wilson bounds. |

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

Commands run from the `benchmarks/` directory. Project steps use the project
environment (`uv run`). Each external model runs with `uv run --no-project
--with <package>==<version>`, which gives it its own cached environment that
never touches the project's dependencies.

```sh
cd benchmarks
uv run asq_phi.py
uv run predict_project.py --benchmark asq_phi
uv run --no-project --python 3.13 --with meddeid==0.4.2 predict_meddeid.py --benchmark asq_phi
uv run --no-project --python 3.13 --with gliner==0.2.29 predict_gliner.py --benchmark asq_phi --arm trained
uv run --no-project --python 3.13 --with gliner==0.2.29 predict_gliner.py --benchmark asq_phi --arm facility
uv run score.py --benchmark asq_phi --systems project meddeid gliner-trained gliner-facility
uv run score_aligned.py --benchmark asq_phi --systems project meddeid gliner-trained gliner-facility
uv run compare.py
```

`synthetic_benchmarks.py --benchmark meddeid_en` and
`--benchmark technetium_test5k` convert the other two benchmarks.
`predict_project.py` runs the default model; pass `--model-dir` for another
checkpoint and `--system` to give it its own key.

Set `HF_HOME` to a shared cache to avoid downloading external weights again.

Run the tests after any change to the harness:

```sh
uv run pytest -q test_external.py
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
2. Run it with `uv run --no-project --with <package>==<version>`, pinning the
   version, so it never shares the project's environment.
3. Write `predict_<system>.py`. Pin the model revision as a constant, emit
   native labels and offsets, and pass every configuration value to
   `common.write_predictions` so it lands in the receipt.
4. Add the native label mapping to `labels.py` before scoring. Use `None` for
   labels that are not PHI under the project policy.
5. If the model card prescribes a configuration (a threshold, a label list),
   that is the primary arm. Additional arms get distinct system keys.

## Results

Current results are in [`benchmarks/results/COMPARISON.md`](../benchmarks/results/COMPARISON.md),
one table per benchmark, with the policy-aligned view first and the
as-published numbers beside it. On the aligned view the default model masks
99.94% of PHI value characters on ASQ-PHI (MedDeID 99.66%), 99.27% on MedDeID
English (MedDeID 99.19%) and 99.65% on the Technetium-I sample (MedDeID
99.65%), with less non-PHI redaction than MedDeID on all three.

## Restricted benchmark: n2c2 2014

The n2c2 (i2b2/UTHealth) 2014 de-identification dataset is real clinical text
with surrogate PHI, distributed under a data use agreement through the
[DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).
It is used for evaluation only.

- Each person obtains it under their own agreement. Keep the raw files outside
  every git repository.
- `n2c2_2014.py --split test --raw <folder of XML files>` converts it into
  `$OUTIS_RESTRICTED_DIR/corpora/n2c2_2014_test.jsonl` (default
  `~/working/deid/restricted`). Predictions for restricted benchmarks are
  written under `$OUTIS_RESTRICTED_DIR/predictions/`. The harness refuses a
  restricted folder inside a git repository.
- `score.py` writes no failure examples for restricted benchmarks; results
  files hold aggregate scores only and may be committed.
- Native subtypes map to project types (`n2c2_2014.TYPE_MAP`); PROFESSION has
  no project type and is excluded from gold. The policy-aligned view
  (`score_aligned.py`) treats STATE, COUNTRY, DEPARTMENT, ORGANIZATION and ROOM
  spans as neutral; view 1 also treated gold ages of 89 or under as neutral.
- Published n2c2 2014 results use specific matching rules (strict or relaxed
  spans, token or entity level, the full PHI set or a HIPAA subset). Match the
  metric exactly before comparing with any paper.

## Restricted benchmark: PhysioNet Deidentified Medical Text

PhysioNet's Deidentified Medical Text v1.0 is 2,434 real MIMIC-II nursing notes
in which every gold PHI instance has been replaced by a realistic surrogate.
It is distributed under the PhysioNet Credentialed Health Data License 1.5.0,
which allows scientific research use only and forbids sharing access. It is
used for research evaluation only, never in a product, training set or tuning
decision.

- Each person obtains it under their own credentialed access and signed data
  use agreement. Keep the raw files outside every git repository.
- Process it only locally. Do not send note text, predictions or failure
  examples to an LLM or any online service unless that use complies with
  PhysioNet's guidance, [Use of MIMIC Data with Large Language Models and Online
  Services](https://physionet.org/news/post/llm-responsible-use/).
  Models run with `HF_HUB_OFFLINE=1` from cached, pinned revisions.
- `physionet_deid.py --raw <folder holding id.text and id.res>` writes
  `$OUTIS_RESTRICTED_DIR/corpora/physionet_deid.jsonl`. Gold spans come from
  aligning `id.text` (surrogates) with `id.res` (`[**category**]` tags);
  conversion fails unless every record aligns. Only counts are printed.
- Tag categories map to project types (`physionet_deid.CATEGORY_RULES`). Ward
  names, states and countries are treated as out of policy, like n2c2's
  DEPARTMENT, STATE and COUNTRY. One release tag has no category and is
  neutral.
- Scoring writes no failure examples; results files hold aggregate scores only.
  Before publishing results that name the corpus, check the licence's
  publication terms (clause 2) and cite the dataset.

## Reading results

- **Compare systems on `results/COMPARISON.md`.** Lead with the policy-aligned
  columns: they score every system under this project's policy. Quote the
  as-published columns beside them, never instead.
- **Rule out a convention gap before blaming a model.** Each
  `<benchmark>.<view>.json` describes every partly covered span by the position
  and shape of what was left uncovered (for example
  `trailing:caps_abbreviation`), without any text, and flags shapes that recur
  for most systems. A flagged shape usually means the neutral rules miss a
  convention: fix the rules in a new view, do not compare under the old one.
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
