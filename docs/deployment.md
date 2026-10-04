# Deployment and operational controls

The detector runs entirely on the host. No text is sent to OpenAI or any other
hosted inference service; downloads happen only during setup.

## Local launcher

```sh
.venv/bin/python scripts/serve_local.py --port 8000
```

The local launcher binds to `127.0.0.1` and creates an owner-readable replacement
key once in the ignored `.cache/local-api-secret` file. It reuses that key across
restarts. A configured `DEID_SECRET` takes precedence. Preserve the key to keep
patient mappings stable; do not commit it or share it with API clients.

`clinical-deid serve` is the packaged equivalent. It reads `DEID_MODEL_DIR`
(default `models/clinical-adapter`) and refuses to start without a trained
adapter. On Apple silicon, set `DEID_DEVICE=mps`; CPU is supported.

## Detector backends

The default API uses the hybrid detector. NER reads the original clinical text;
structural extractors then scan that same complete text and merge their matches
with the learned spans. An email does not need a successful NER prediction to
be detected. The email recognizer supports ASCII aliases and punctuation,
Unicode mailbox text and domains, punycode domains, quoted mailbox names, and
IP address literals while preserving original character offsets. Regression
tests force NER to return no PHI and verify both redaction and synthetic
replacement through the public Python API. Obfuscated forms such as
`alice [at] example [dot] org` remain outside this deterministic grammar.
`backend="model"` is available for evaluating NER separately and skips the
structural extractors.

## Batch limits and response contents

`POST /v1/batch` accepts `{"items":[...requests...]}`. There are at most 16
requests and 100,000 total characters per batch. Responses include transformed
text, original **character offsets** of detected entities, entity types,
replacement strings, model/rule confidence scores, warnings, and a model version.
Response metadata contains no original PHI, patient anchor, date offset, or key.
Scores are model probabilities or rule scores, not calibrated safety estimates.

## Security controls

Request bodies and source PHI are not logged by the application. Validation and
runtime errors return sanitized messages, and responses use `Cache-Control:
no-store`. Set `DEID_API_KEY` for `X-API-Key` authentication. The CLI requires it
when binding beyond loopback. Configure TLS and network access controls in
your deployment; downstream proxies and caller logs must follow your data
handling policy. `GET /health` checks process liveness; `/ready` verifies a
loaded model and reports whether synthetic mode has a configured secret.
