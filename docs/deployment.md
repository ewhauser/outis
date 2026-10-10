# Deployment and operational controls

The detector runs entirely on the host. No text is sent to OpenAI or any other
hosted inference service; downloads happen only during setup.

## Local launcher

```sh
uv run scripts/serve_local.py --port 8000
```

The local launcher binds to `127.0.0.1` and creates an owner-readable replacement
key once in the ignored `.cache/local-api-secret` file. It reuses that key across
restarts. A configured `OUTIS_SECRET` takes precedence. Preserve the key to keep
patient mappings stable; do not commit it or share it with API clients.

`outis serve` is the packaged equivalent. It reads `OUTIS_MODEL_DIR`
(default `models/deid-deberta-v3-base`) and refuses to start
without a fine-tuned encoder checkpoint there. On Apple silicon, set
`OUTIS_DEVICE=mps`; CPU is supported.

## Detector

Detection is model-only: a fine-tuned ModernBERT token classifier reads the
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
