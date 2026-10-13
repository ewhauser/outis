# Deployment and operational controls

The detector runs entirely on the host. No text is sent to OpenAI or any other
hosted inference service. The only download is the model itself: without a
configured checkpoint, the first run fetches the pinned default from Hugging
Face into the Hugging Face cache. Set `HF_HUB_OFFLINE=1` to forbid network
access once it is cached, or point `OUTIS_MODEL_DIR` at a local checkpoint.

## Local launcher

```sh
uv run scripts/serve_local.py --port 8000
```

The local launcher binds to `127.0.0.1` and creates an owner-readable replacement
key once in the ignored `.cache/local-api-secret` file. It reuses that key across
restarts. A configured `OUTIS_SECRET` takes precedence. Preserve the key to keep
patient mappings stable; do not commit it or share it with API clients.

`outis serve` is the packaged equivalent. It loads the checkpoint in
`OUTIS_MODEL_DIR` when set, and otherwise `models/deid-deberta-v3-base` if
its weights are present, or the pinned model from Hugging Face. It refuses to
start without a fine-tuned encoder checkpoint. On Apple silicon, set
`OUTIS_DEVICE=mps`; CPU is supported.

## Docker

The [Dockerfile](../Dockerfile) builds two images from `uv.lock`. Both bake in
the pinned Hugging Face checkpoint and set `HF_HUB_OFFLINE=1`, so the container
never reaches the network and can run with `--network none` for the CLI. They
run as an unprivileged user and serve on port 8000.

| Variant | Platforms | PyTorch |
| --- | --- | --- |
| `cpu` | `linux/amd64`, `linux/arm64` | CPU build of the locked version |
| `gpu` | `linux/amd64` | the locked PyPI build with CUDA 13 libraries |

The GPU image needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
and a driver that supports CUDA 13 (580 or newer). With `--gpus`, the default
`OUTIS_DEVICE=auto` selects the GPU and infers in bf16 where the GPU supports
it; without a GPU it falls back to the CPU.

The [Docker workflow](../.github/workflows/docker.yml) builds and smoke-tests
both images on every pull request and publishes them to
`ghcr.io/ewhauser/outis`:

| Tag | Published by |
| --- | --- |
| `X.Y.Z-cpu`, `X.Y-cpu`, `cpu`, `X.Y.Z`, `X.Y`, `latest` | each release (CPU) |
| `X.Y.Z-gpu`, `X.Y-gpu`, `gpu` | each release (GPU) |
| `main-cpu`, `main-gpu` | each push to `main` |

Pre-releases get only their `X.Y.Z-cpu` and `X.Y.Z-gpu` tags. Images carry
build provenance and an SBOM; pin a digest for reproducible deployments.

```sh
docker run -d -p 8000:8000 \
  -e OUTIS_API_KEY=change-me \
  -e OUTIS_SECRET=a-replacement-secret-of-at-least-32-bytes \
  ghcr.io/ewhauser/outis:cpu
```

The server binds to `0.0.0.0` inside the container, so it refuses to start
without `OUTIS_API_KEY`. Set `OUTIS_SECRET` to enable synthetic mode, and keep
it stable across container restarts to keep patient mappings stable. The other
`OUTIS_*` settings below apply unchanged. To run another checkpoint, mount it
and set `OUTIS_MODEL_DIR`. The CLI reads standard input:

```sh
docker run --rm -i --network none ghcr.io/ewhauser/outis:cpu text < note.txt
```

To build locally, pass the variant:

```sh
docker build --build-arg VARIANT=cpu -t outis:cpu .
```

## Detector

Detection is model-only: a fine-tuned DeBERTa-v3-base token classifier reads the
original text in overlapping windows and a constrained BIOES decoder returns
spans at exact original character offsets. There are no regex extractors or
context guards; the rule layer was removed on 2026-10-04 because it cost
precision and recall on unseen authors without masking more direct
identifiers. Explicit `custom_entities` patterns in a request policy are still
applied as requested, as are operator-supplied site dictionaries when
configured (below).

## Site dictionaries

Site dictionaries add a site's own vocabulary — facility and department
names, provider names, identifier prefixes and formats — without retraining.
They are off by default and are loaded once at startup from
`OUTIS_SITE_DICTIONARIES` (server and CLI), `outis text --site-dictionaries`,
or `Deidentifier(..., site_dictionaries=...)` in Python. Outis ships no
dictionaries, and **their effect on detection accuracy has not been
measured**. Unlike the generic rule layer removed on 2026-10-04, they contain
only vocabulary the site supplies and reviews. Matching, priority and request
selection are described in the [API reference](api.md#site-dictionaries).

The setting is a path to a JSON manifest, or to a directory whose `*.json` files
each define one dictionary (named after the file unless `name` is given):

```json
{
  "version": 1,
  "dictionaries": [
    {
      "name": "facilities",
      "label": "FACILITY",
      "terms_file": "facilities.txt",
      "terms": ["Quillfeather Memorial Infirmary"]
    },
    {
      "name": "local-ids",
      "label": "ID",
      "terms": ["QX-"],
      "case_sensitive": true,
      "word_boundaries": false,
      "patterns": ["\\bQFM-\\d{6}\\b"]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `name` | 1–64 lowercase letters, digits, `_` or `-`; unique. Requests select dictionaries by name. |
| `label` | A clinical or extended label, or a custom label (uppercase letters, digits and underscores, at most 48 characters). `CONDITION` and `MEDICATION` are not accepted. |
| `terms` | Literal terms, 1–256 characters each after trimming and collapsing internal whitespace; each must contain a letter or digit. |
| `terms_file` | UTF-8 text, one term per line; blank lines and lines starting with `#` are skipped. A relative path resolves against the file that names it. |
| `case_sensitive` | Default `false`. |
| `word_boundaries` | Default `true`. Set `false` for prefixes that occur inside longer tokens. |
| `patterns` | Up to 32 identifier-format expressions per dictionary, with the `custom_entities` limits. |

A configuration holds at most 64 dictionaries and 1,000,000 terms. An invalid
configuration stops startup with an error that names the dictionary and the
field, term number or file line, never a term. Restart the process to load a
changed configuration; `/ready` reports each dictionary's term count and
content fingerprint so a rollout can be checked.

Terms are indexed by folded text and their first four characters, so the
per-request cost is one pass over the text, independent of the number of
terms. On an Apple M3 Max, a 100,000-character note took about 10 ms with 10
or with 100,000 synthetic facility names; loading 100,000 names took well under
a second and roughly 40 MB. The work at each position grows with the number of
distinct term lengths sharing that position's four-character prefix, so a
dictionary with hundreds of lengths under one common prefix, matched against a
note that repeats it, took about 0.45 s. `tests/test_site_dictionaries.py`
includes a cost check.

Dictionaries can hold sensitive staff or partner names. Keep them in
access-controlled storage such as a read-only volume, mount it into the
container, and set `OUTIS_SITE_DICTIONARIES` to its path. Any evaluation of a
dictionary's effect must follow the project's evaluation protocol and must not
use sealed or blind evaluation banks.

## Batch limits and response contents

`POST /v1/batch` accepts `{"items":[...requests...]}`. There are at most 16
requests and 100,000 total characters per batch. To de-identify a whole note
table, use [`outis batch`](#batch-processing) instead. Responses include transformed
text, original **character offsets** of detected entities, entity types,
replacement strings, model confidence scores, warnings, and a model version.
Response metadata contains no original PHI, patient anchor, date offset, or key.
Scores are model probabilities, not calibrated safety estimates.

## Span review report

`outis review` writes a static HTML page that shows each note's original text,
with its spans highlighted and labelled, beside the replaced output. It has
per-label counts, a note index, and previous/next links. A reviewer can click a
span to remove it, select original text and pick a label to add one, and
download a corrected span file (`spans.reviewed.jsonl`) that
`outis batch --spans`, `outis spans replace` and `/v1/replace` accept. The output column is not
updated in the page; run `outis review` again on the corrected file to see it.
See [supplied spans](api.md#supplied-spans-detect-review-replace) for the
workflow and span file format.

```sh
outis review --input notes.jsonl --spans spans.jsonl --output review.html \
  --mode synthetic --patient-column patient_id
```

**The report contains PHI by design**: the full original text of every note,
plus note and patient IDs. Outis therefore:

- prints a warning on stderr and repeats it at the top of the page;
- creates the file readable and writable by its owner only (mode `0600`,
  including when it replaces an existing file);
- keeps the page self-contained, with no external scripts, styles, fonts,
  images or links. A Content-Security-Policy allows only the page's own inline
  style and script, by SHA-256 hash, and blocks every network request;
- builds the page from escaped text, so note contents cannot inject markup.

Handle the report and any downloaded span file like the notes themselves. Write
them only to approved, access-controlled storage, open them on a machine
allowed to display PHI, and do not email, upload or commit them. Browsers may
keep downloaded files and history, so delete the report and the corrected file
once they have been applied. Without `OUTIS_SECRET`, `outis review` still
shows redaction output. Fingerprinted span files then need
`OUTIS_FINGERPRINT_SECRET` or an explicit `--skip-fingerprint-check`.

The reviewer can add a span with any supported label, or any custom label the
span file uses, even when detection found none of that type. Added spans carry
source `review`, so the ages policy never drops them.

## Security controls

Request bodies and source PHI are not logged by the application. Validation and
runtime errors return sanitized messages, and responses use `Cache-Control:
no-store`. Set `OUTIS_API_KEY` for `X-API-Key` authentication. The CLI requires it
when binding beyond loopback. Configure TLS and network access controls in
your deployment; downstream proxies and caller logs must follow your data
handling policy. `GET /health` checks process liveness; `/ready` verifies a
loaded model and reports whether synthetic mode has a configured secret, plus
site dictionary names, counts and fingerprints when configured.

## Capacity

Model-backed routes (`/v1/deidentify`, `/v1/batch`, `/v1/detect`,
`/v1/detect/batch`, `/v1/analyze`, `/v1/structured`, `/v1/files` and the
WebSocket) and the replacement-only `/v1/replace` and `/v1/replace/batch` run
on a dedicated inference pool, separate from the server's request
threads. `OUTIS_INFERENCE_WORKERS` (default 2) sets its threads and
`OUTIS_INFERENCE_QUEUE` (default 32) how many further requests may wait. Beyond
that, requests fail fast with `503` and `Retry-After: 1` instead of queueing
without bound. Detector calls are serialized one model batch (16 texts) at a
time, so a large structured request interleaves with other work rather than
holding the model throughout. `/health` and `/ready` never wait on inference
capacity, so liveness probes stay accurate under load.

## Batch processing

`outis batch` de-identifies a table of notes in one process on one machine,
reading and writing as it goes so memory stays flat however large the table.
It loads the model once and uses the same detection, policy and replacement
code as the library.

```sh
export OUTIS_SECRET=...   # synthetic mode only; at least 32 bytes
outis batch --input notes.jsonl --output deid.jsonl \
  --id-column note_id --text-column text --patient-column patient_id \
  --mode synthetic --policy policy.json --pass-through encounter_id
```

### Input

JSONL input has one JSON object per line; blank lines are skipped. Parquet
input and output need the `parquet` extra (`pip install 'outis[parquet]'`);
without it the command stops and names the extra. Each file's format follows
its extension (`.parquet` and `.pq` are Parquet, anything else JSONL), or
`--format` sets both.

Every row is checked before any output is written. A note ID must be a
nonempty string or an integer, the same kind in every row, and unique. The
text must be a string. When synthetic replacement or keyed entity IDs are
used, every row also needs a patient ID: a nonempty string or an integer of at
most 256 characters (integers anchor as their decimal string). Errors name the
line or row number and the problem, never the value.

`--policy` reads a JSON object with an optional `policy` (the
[detection policy](api.md#detection-and-policies)) and an optional
`replacement` (the [replacement controls](api.md#replacement-controls-and-output-offsets)):

```json
{
  "policy": {"profile": "extended", "ages": "all"},
  "replacement": {"type": "synthetic", "entity_ids": "keyed"}
}
```

`--project-id` sets one `project_id` for every note; [project
scope](replacement.md#patient-and-project-scope) requires it. [Site
dictionaries](#site-dictionaries) load from `--site-dictionaries` or
`OUTIS_SITE_DICTIONARIES`, as for the server. The policy, replacement options, scope and dictionary selection are
checked before the first note, and an invalid combination stops the run with
the service's error.

Each note is processed as `Deidentifier.deidentify(text, mode=..., patient_id=...,
policy=..., replacement=..., project_id=...)` would process it, with its own
row's patient ID as the anchor. All notes of a patient therefore share one date shift and one
synthetic identity mapping, whatever batch they fall in. Batching only groups
encoder windows; tests check that each note's output equals the library's
per-note output.

### Output

The output has one row per input row, in input order:

| Field | Contents |
| --- | --- |
| ID column | The note ID, as read |
| patient column | The patient ID, as read, when `--patient-column` is given |
| `--pass-through` columns | Copied unchanged. They are not de-identified |
| text column | The de-identified text; null for a failed row |
| `status` | `ok` or `failed` |
| `error` | For a failed row, a code from the table below |
| `entities` | `start`, `end` (offsets in the original note), `label`, `score` and `source`; with a policy file also `entity_id`, `processed_start` and `processed_end`. `original` is added only with `--include-originals` |
| `warnings` | Replacement warnings, such as unsupported dates that were redacted |
| `model_version` | The loaded checkpoint's digest |
| `mappings` | Only when the policy file sets `replacement.include_mapping`; holds original values |

Original values reach the output only through `--include-originals`,
`include_mapping`, a `keep` replacement or a pass-through column. The text
column keeps its input name and holds de-identified text.

In Parquet output the ID column is `int64` or `string`. Patient and
pass-through columns are `bool`, `int64`, `float64` or `string`, as their
values allow; Parquet dates, timestamps and decimals become ISO or decimal
strings, and mixed or nested values become JSON text.

| Error code | Meaning |
| --- | --- |
| `text_too_long` | The note exceeds the 100,000-character per-text limit; it is not truncated |
| `out_of_memory` | The note did not fit in memory even alone (see below) |
| `detector_error` | The detector raised an error or returned invalid output for this note |
| `policy_budget_exceeded`, `too_many_entities` | A policy limit from the [API guide](api.md#detection-and-policies) |
| `output_too_large`, `marker_collision` | A replacement limit |
| `rejected`, `internal_error` | Any other rejection or failure |

### Batches and out-of-memory errors

`--batch-size N` sets both the notes sent to the detector at once and the
encoder windows in one forward pass (the encoder's `batch_size`; default 8).
Raise it on a GPU with memory to spare. On a CUDA or MPS out-of-memory error,
the runner releases cached memory, halves the notes and the window batch, and
retries each half, down to one note in one-window passes. A note that still
does not fit becomes a failed row with `out_of_memory`, and the run continues.
The next batch starts again at `N`. An error other than out-of-memory is
retried note by note, so it fails only the note that raises it.

### Resume

Rows are appended to the output a whole batch at a time and flushed to disk.
Only then is `<output>.progress.json` replaced, atomically, with the number of
input rows and output bytes now complete. When re-run with `--resume`, the
command truncates the output to the recorded length, so any partly written
batch is dropped, and continues with the next batch.

**Guarantee.** With the same input and settings, an interrupted run continued
with `--resume` produces exactly the output of an uninterrupted run: the same
JSONL bytes, or the same Parquet table. No row is lost, duplicated or torn,
whether the run was interrupted or killed or the machine went down (provided
the storage honours `fsync`). This assumes
deterministic detection, which holds on CPU; on a GPU the results are as
deterministic as its kernels, and an out-of-memory split changes how notes are
batched. `--resume` refuses to continue if the input's SHA-256, the columns,
mode, policy file contents, project ID, site dictionaries, batch size, model,
device, precision or replacement secret differ from the interrupted run's (the secret is compared through a
keyed digest, never stored).

Parquet cannot be appended to, so Parquet output is journalled to
`<output>.partial.jsonl` and converted to Parquet once every row is complete;
the journal is then removed. The progress file stays after a run completes, so
re-running with `--resume` does nothing. With nothing to resume, `--resume`
starts the run. Without it, the command never overwrites an existing output.

### Detect once, replace later

`--detect-only` runs detection and writes a span file instead of
de-identified rows. `--spans` then replaces those spans without loading the
model, as often as needed: under another key, project, style or replacement
policy, or after a reviewer corrected the file with
[`outis review`](#span-review-report). The
[span file format](api.md#span-files) holds offsets and labels, never note text.

```sh
export OUTIS_FINGERPRINT_SECRET=...   # optional; defaults to OUTIS_SECRET
outis batch --detect-only --input notes.parquet --output spans.jsonl \
  --id-column note_id --text-column note_text --patient-column patient_id
outis batch --spans spans.jsonl --input notes.parquet --output deid.parquet \
  --id-column note_id --text-column note_text --patient-column patient_id \
  --mode synthetic --policy policy.json --project-id release-1
```

- **`--detect-only`** takes the input, columns, `--policy` (its `policy` part),
  `--project-id`, `--site-dictionaries`, `--batch-size` and model options of a
  full run, with the same batching, out-of-memory handling and resume. It
  writes JSONL, one record per input row in input order, readable by its owner
  only. It takes neither `--pass-through` nor `--include-originals`. A note
  whose detection fails gets a record with an `error` code and no spans. With
  a fingerprint secret configured, each record carries a `text_fingerprint`.
- **`--spans SPAN_FILE`** writes the same output rows as a full run, in JSONL
  or Parquet, with each entity's `source` set to `supplied` and
  `model_version` taken from the span record. Only `policy.ages` of `--policy`
  applies; the other policy fields already acted during detection, and site
  dictionaries are not loaded. Records must follow the input's row order, so
  both files are streamed. Before writing anything, it checks every record
  against its row: the same note ID, the same patient ID if the record has
  one, valid nonoverlapping spans within the text, and a matching
  `text_fingerprint`. Errors name the span file line, never values. A record
  with an `error` fails its row with that code.
- Replacing the spans of a `--detect-only` run with the same settings gives the
  same output rows as one full run, apart from each entity's `source`.

If `OUTIS_SECRET` changes between the two runs and fingerprints were made
under it, set `OUTIS_FINGERPRINT_SECRET` to the key used at detection.
`--skip-fingerprint-check` replaces without checking fingerprints; use it only
when the input is known to be the detected text. `--resume` also refuses to
continue a run whose stage or span file changed.

### Progress and exit status

Progress goes to stderr: notes done, percentage, content tokens per second,
the estimated time remaining and the failed count, every 2 seconds on a
terminal and every 30 seconds otherwise. A summary follows with counts of
processed, skipped (completed by an earlier run) and failed notes, and
throughput per stage. `--summary-json` also writes the summary to a file.
Neither contains note text, note IDs or patient IDs.

The exit status is 0 when every row in the output succeeded, 1 when any row
failed, 2 for an input or usage error (nothing is processed), and 130 after an
interruption.

### Throughput

[`scripts/benchmark_batch.py`](../scripts/benchmark_batch.py) runs
`outis batch` on synthetic notes only and reports throughput with its
hardware, dtype, batch size and model revision. By default it joins the first
4,000 rows of the public
[`ewhauser/outis-synthetic-phi`](https://huggingface.co/datasets/ewhauser/outis-synthetic-phi)
validation split, four rows per note, into 1,000 notes; `--source generated`
uses fictional templates and needs no download. It never reads real or
restricted corpora.

A **content token** is one token of note text under the encoder's tokenizer
(the DeBERTa-v3 SentencePiece tokenizer for the default model), without
special tokens, counted once per note however many overlapping windows read
it. Each stage divides the run's content tokens by that stage's wall time:

- **Detection**: tokenization, encoder forward passes and BIOES decoding,
  for whole batches.
- **Replacement**: per-note policy, identity analysis and rendering in
  `Deidentifier.deidentify`.
- **End to end**: from reading the first row to the finished output,
  including token counting, writing and the per-batch `fsync`. It excludes
  loading the model (about 5 seconds) and the up-front input check.

Measured on 2026-10-12 on an Apple M3 Max laptop (12 performance and 4
efficiency cores, 64 GB, macOS 26.5), PyTorch 2.14.1 with 12 CPU threads,
model `ewhauser/outis-deberta-v3-base` at revision `11019035`, default batch
size 8, dataset revision `46b09a63`. The 1,000 notes average 1,465 content
tokens (1,462,474 in all); two joined notes exceed the 100,000-character limit
and are failed rows (`text_too_long`) in every run.

| Device | dtype | Mode | Detection | Replacement | End to end |
| --- | --- | --- | ---: | ---: | ---: |
| CPU | fp32 | redact | 1,764 | 676,811 | 1,758 |
| CPU | fp32 | synthetic | 1,778 | 602,025 | 1,771 |
| Apple MPS | bf16 | redact | 6,891 | 653,340 | 6,792 |
| Apple MPS | bf16 | synthetic | 6,941 | 620,100 | 6,838 |

Content tokens per second. **These are single runs and indicative only.**
Detection dominates; replacement takes about 1% of the time or less. On this laptop
the GPU slows under sustained load: the MPS rows were measured after the
machine had idled, while MPS runs started straight after another long run
measured as low as 3,459 tokens per second end to end. Throughput
also depends on note length, the batch size and other load on the machine.
These figures say nothing about detection accuracy, and other projects'
throughput figures use other models, hardware and token definitions.

No NVIDIA GPU figures have been measured yet. To measure one, run from a
checkout on the GPU host:

```sh
uv run --extra parquet scripts/benchmark_batch.py --device cuda --report gpu-bs8.json
uv run --extra parquet scripts/benchmark_batch.py --device cuda --batch-size 32 --report gpu-bs32.json
```

The report records the GPU name, dtype (bf16 where the GPU supports it),
batch size and timings per stage. Add `--mode synthetic` for synthetic
replacement.
