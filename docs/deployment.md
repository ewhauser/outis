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

## Detector

Detection is model-only: a fine-tuned DeBERTa-v3-base token classifier reads the
original text in overlapping windows and a constrained BIOES decoder returns
spans at exact original character offsets. There are no regex extractors or
context guards; the rule layer was removed on 2026-10-04 because it cost
precision and recall on unseen authors without masking more direct
identifiers. Explicit `custom_entities` patterns in a request policy are still
applied as requested.

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
loaded model and reports whether synthetic mode has a configured secret.

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
