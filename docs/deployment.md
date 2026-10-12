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
requests and 100,000 total characters per batch. Responses include transformed
text, original **character offsets** of detected entities, entity types,
replacement strings, model confidence scores, warnings, and a model version.
Response metadata contains no original PHI, patient anchor, date offset, or key.
Scores are model probabilities, not calibrated safety estimates.

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
WebSocket) run on a dedicated inference pool, separate from the server's request
threads. `OUTIS_INFERENCE_WORKERS` (default 2) sets its threads and
`OUTIS_INFERENCE_QUEUE` (default 32) how many further requests may wait. Beyond
that, requests fail fast with `503` and `Retry-After: 1` instead of queueing
without bound. Detector calls are serialized one model batch (16 texts) at a
time, so a large structured request interleaves with other work rather than
holding the model throughout. `/health` and `/ready` never wait on inference
capacity, so liveness probes stay accurate under load.
