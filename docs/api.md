# Clinical text API

The API detects and transforms English clinical text locally. It adds request
policies, configurable replacements, related-message context, structured inputs,
supported native text documents, entity analysis, and caller-controlled
restoration. OCR, image redaction, audio, and multilingual detection are outside
this implementation. These interfaces do not establish new accuracy or
regulatory qualification; see the [evaluation results](../README.md#results).

Start the server as described in the [README](../README.md#quickstart).
Interactive schemas are at `/docs`; the generated schema is at `/openapi.json`.
When `OUTIS_API_KEY` is configured, send `X-API-Key` on processing and operational
requests. `/health` is public. HTTP responses use `Cache-Control: no-store`.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/detect` | Return entity spans without transformed text. |
| `POST /v1/detect/batch` | Detect in up to 16 messages. |
| `POST /v1/deidentify` | Replace detected spans in one text. |
| `POST /v1/batch` | Replace spans in up to 16 messages. |
| `POST /v1/analyze` | Detect, parse supported entity values, and identify explicit relationship templates. |
| `POST /v1/reidentify` | Restore values from caller-supplied literal mappings. |
| `POST /v1/structured` | Transform selected JSON, CSV, or XML text fields. |
| `POST /v1/files` | Transform a supported file supplied as base64. |
| `GET /v1/policies` | List profiles, entity types, policy version, and principal limits. |
| `GET /v1/usage` | Read aggregate process-local usage. |
| `GET /metrics` | Read Prometheus counters and the active-request gauge. |
| `GET /diagnostics` | Read API/model version, device, platform, and supported language scope. |
| `GET /health`, `GET /ready` | Check liveness or loaded-model readiness. |
| WebSocket `/v1/ws` | Reuse a connection for individual detect, deidentify, analyze, and reidentify requests. |

Text inputs are at most 100,000 Unicode characters. A public batch has at most
16 items and 100,000 total characters; linked context includes the inserted
newline separators in that limit. HTTP bodies and WebSocket text envelopes are
limited to 16 MiB. Invalid inputs fail without partial transformed output.

## Detection and policies

`/v1/detect` accepts `text`, an optional `policy`, and an optional `project_id`.
The same policy object works with replacement, analysis, and structured/file
processing. Omitting it retains the clinical detector's existing defaults.

```json
{
  "text": "Study subject STUDY-482901 met Dr. Mara Calder.",
  "policy": {
    "profile": "clinical_default",
    "custom_entities": [
      {"label": "STUDY_ID", "pattern": "\\bSTUDY-\\d{6}\\b"}
    ]
  }
}
```

The response contains `entities`, `entity_count`, `model_version`, and policy
metadata. Each entity has `start`, `end`, `label`, `score`, `source`, and an
`entity_id`. The offsets are zero-based Unicode character offsets with an
exclusive end: Python `text[start:end]` selects the entity. They are not UTF-8
byte offsets or JavaScript UTF-16 code-unit offsets. Original values are not
copied into detection metadata. Scores are model probabilities or rule scores,
not calibrated safety estimates.

| Policy field | Behavior |
| --- | --- |
| `profile` | `clinical_default` uses the 14 trained clinical labels. `extended` also enables the finer identifier recognizers below. `identifiers_only` omits `DATE`, `AGE`, `LOCATION`, and `FACILITY`, adds extended identifiers, and omits `DOB`. |
| `include` | When supplied, replaces the enabled-label set. An empty list disables ordinary detections. |
| `exclude` | Removes labels after `include` or the profile is applied. |
| `allowlist` | Suppresses an ordinary detection only when its complete detected value exactly equals an entry, case-sensitively. It does not exempt arbitrary substrings of a larger detected span. |
| `blocklist` | Forces case-sensitive literal matches anywhere in the text as `ID`. These matches override the allowlist and label selection. |
| `custom_entities` | Up to 32 `{label, pattern}` rules. The entire regex match becomes the span. Custom labels do not change or retrain the NER classifier. |
| `medical_terms` | Explicit `CONDITION` and/or `MEDICATION` term lists, matched case-insensitively at word boundaries. These categories are opt-in dictionaries, not additional trained medical models. |
| `ages` | `over_89` (default) or `all`. The detector finds every age; `over_89` leaves ages of 89 and under in place, as the HIPAA Safe Harbor age rule permits, and generalises older ages; `all` removes every age. |

Profiles first choose enabled labels, then custom labels and supplied medical
dictionary labels are added, then `include` and `exclude` apply. Recognizers for
enabled extended labels also run when those labels are explicitly included.
An `include` list must therefore name every label the caller wants retained.

The clinical labels are `NAME`, `DATE`, `ADDRESS`, `PHONE`, `EMAIL`, `MRN`, `SSN`,
`ACCOUNT`, `ID`, `URL`, `IP`, `AGE`, `LOCATION`, and `FACILITY`. Extended labels
are `CREDIT_CARD`, `CVV`, `ROUTING_NUMBER`, `BANK_ACCOUNT`, `PASSPORT`,
`DRIVER_LICENSE`, `PASSWORD`, `USERNAME`, and `DOB`. Most extended identifiers
require an explicit field heading, such as `passport:` or `routing number:`;
credit-card candidates use a numeric grammar. These recognizers are bounded
rules and do not cover every institution's identifier formats.

Overlapping policy matches are merged into their complete connected coverage,
so choosing a label does not leave an overlapping suffix exposed. The label
priority is blocklist, custom regex, extended/medical dictionary, then ordinary
detector. Equal-priority matches prefer greater span length, then score and a
stable label ordering. Adjacent non-overlapping spans remain separate.

Custom labels use uppercase letters, digits, and underscores, begin with a
letter, and have at most 48 characters. Patterns have at most 512 characters;
empty matches and excessive repetition counts are rejected. Matching has a
25 ms per-expression timeout and a 250 ms cumulative policy budget, plus a
10,000-match bound. Budget failures reject the request. There are at most 128
allowlist entries, 128 blocklist entries, and 128 supplied medical terms; each
literal has 1–256 characters. Policy metadata reports counts and labels rather
than the submitted literals or regexes.

## Replacement controls and output offsets

`/v1/deidentify` accepts the detection fields plus `mode`, `patient_id`,
`date_shift_days`, and `replacement`. Existing requests with no replacement
object retain typed redaction such as `[NAME]`, or the existing synthetic
behavior when `mode` is `synthetic`. An explicitly supplied empty
`replacement: {}` selects configured defaults instead: numbered markers such
as `[NAME_e1]` and processed offsets.

```json
{
  "text": "Mara Calder can be reached at mara.calder@example.invalid.",
  "replacement": {
    "type": "marker",
    "pattern": "[{label}_{id}]",
    "per_type": {"EMAIL": "mask"},
    "mask_character": "*",
    "entity_ids": "numbered",
    "include_mapping": false
  }
}
```

| Replacement field | Behavior |
| --- | --- |
| `type` | Default action: `marker`, `mask`, or `synthetic`. When a replacement object is present, this field chooses the action independently of the legacy `mode`. |
| `pattern` | Marker template, default `[{label}_{id}]`. Only `{label}`, `{id}`, and `{index}` substitutions are accepted; `index` is the one-based occurrence in the request's identity scope. It continues across linked/shared messages or structured fields, and starts again for independently processed texts. |
| `mask_character` | One visible character, repeated once per source character. Default `*`. |
| `per_type` | Overrides the action by label with `marker`, `mask`, `synthetic`, or `keep`. `keep` deliberately retains the original detected value. |
| `entity_ids` | `numbered` gives request-local identities such as `e1`. `keyed` uses patient-scoped HMAC identities and requires `patient_id` plus the server's stable secret. |
| `include_mapping` | Default `false`. When true, returns sensitive `{marker, original, label}` mappings for changed values. |

Configured replacements include `processed_start` and `processed_end` alongside source
`start` and `end`. The processed offsets select the emitted replacement in
the returned `text`; they do not refer to the original text. Configured
replacements copy all characters outside detected spans exactly. This property
does not establish that the detector found every identifier or preserved all
clinical meaning.

The replacement engine can generate fictional values for its original clinical
labels. Extended and custom labels fall back to markers under a synthetic
action. `DOB` is an extended label and also uses this fallback; the patient date
shift applies to `DATE`. For configured synthetic replacement, set both
`mode: "synthetic"` and `replacement.type: "synthetic"`, and provide a stable
`patient_id`. A nonzero `date_shift_days` may range from -3650 to 3650. Without
an explicit shift, the server derives a stable nonzero patient shift from its
secret. Unsupported date forms are redacted with warnings. The analysis
endpoint's `locale` does not change date-shifting interpretation.
If an inferred day/month/year date shifts to a value that could mean a different
date under the month/day/year default, it is emitted as year/month/day with the
original separator and a warning. For example, `13/09/2024` shifted by 20 days
becomes `2024/10/03`.

The response's `preservation` fields report date-shift outcomes and replacement
consistency for the selected actions. Masking, marking, or keeping a detected
date does not establish date-interval preservation. Keeping values deliberately
reduces deidentification. See the README for existing synthetic alias, age,
date-format, and fictional-identity limitations.

## Related messages and identities

Both batch endpoints accept `{items: [...], link_batch: false,
shared_entities: false}` and return an array in input order. Their two optional
switches have distinct effects:

- `shared_entities: true` detects each text independently, then assigns shared
  identities and replacement context across the batch.
- `link_batch: true` detects the texts together, joined by newlines, and maps
  detections back to each input's local offsets. It also shares identities.

Shared requests must have identical policies, project, patient anchor, mode,
replacement options, and date-shift settings. Use a separate batch for a
different patient or processing policy. The service does not persist linked
context between requests.

```json
{
  "items": [
    {"text": "Patient Mara Calder arrived.", "replacement": {"type": "marker"}},
    {"text": "Mara returned for follow-up.", "replacement": {"type": "marker"}}
  ],
  "link_batch": true
}
```

Identity resolution uses repeated values and unambiguous literal first/last
name aliases of detected full names. Supported explicit nickname declarations
link the detected nickname occurrence and owner to the same ID and synthetic
identity. Shared messages and fields use their exact occurrence offsets; a
matching nickname alone does not establish ownership. If the generated nickname
would repeat the original nickname, replacement uses the owner's synthetic family
name or redacts it with a warning. Pronouns and arbitrary semantic references
remain unresolved. If two full names share a reduced name, ambiguous reduced
mentions remain separate. Opaque identifiers retain their case and spelling.
Synthetic replacements for `MRN`, `ACCOUNT`, `ID`, `EMAIL`, and `URL` likewise
use exact source values, including whitespace and Unicode spelling. This fixes
earlier normalization that merged distinct values; affected replacements change
on upgrade, so reprocess related notes to keep one mapping universe.
A detection crossing a message boundary produces local `context_fragment`
spans; synthetic replacement uses markers for these fragments.

## Explicit restoration

To obtain mappings, set `replacement.include_mapping: true`. Use distinct
markers, for example `"pattern": "[{label}_{id}_{index}]"`, when multiple
spellings of the same identity occur in one text. Ambiguous mappings, markers
already present in the original context, and overlapping markers are rejected.
Mappings contain original sensitive values. The server returns them to the
caller and does not maintain a persistent mapping store. Structured/file
responses also return a combined `mappings` collection when explicitly requested;
conflicting markers across their fields reject the request. A public text batch
returns mappings separately in each item. Linked/shared batches validate a
common marker namespace; keep independently processed items' mapping scopes
separate.

Pass mappings back to `/v1/reidentify` along with the processed text or an LLM
response that preserved the literal markers:

```json
{
  "text": "Follow up with [NAME_e1_1].",
  "mappings": [
    {"marker": "[NAME_e1_1]", "original": "Mara Calder", "label": "NAME"}
  ],
  "allowed_types": ["NAME"],
  "restore_sensitive": true
}
```

Restoration is off for identifiers by default: `restore_sensitive` defaults to
`false`. `allowed_types` narrows which labels may be restored; it does not by
itself authorize identifier restoration. Only explicitly allowed `CONDITION`
and `MEDICATION` mappings can restore without `restore_sensitive: true`.
With the default options, no supplied mapping is restored. The result contains
`text` and `restored_count`. Restoration uses exact literal matches once;
inserted originals are not rescanned. Duplicate, overlapping, or ambiguous
markers fail, and altered markers remain unchanged. The request text and
result are bounded to 1,000,000 characters, with at most 20,000 mappings and
2,000,000 combined marker/original characters.

## Structured data

`/v1/structured` takes `data`, `format` (`json`, `csv`, or `xml`), optional
`paths`, optional replacement `options`, and `link_records` (default `false`).
`options` contains the fields of a deidentification request except `text`.
JSON data is a JSON value, including root `null`, which is returned unchanged.
The `data` field is required; omitting it returns 422. CSV and XML data are strings. The response returns
transformed `data`, its `format`, located `entities`, counts, warnings, and
explicitly requested mappings.

```json
{
  "format": "json",
  "data": {"records": [{"note": "Mara Calder arrived.", "visit_count": 3}]},
  "paths": ["/records/*/note"],
  "options": {"replacement": {"type": "marker"}},
  "link_records": false
}
```

Selection uses JSON-pointer syntax: `~1` means `/`, `~0` means `~`, and `*`
matches one path component. A selected path selects its subtree; `""` selects
the entire document. A selector that matches nothing is an error. Explicit
selection leaves all unselected content unchanged.

| Format | Default processing and selectors |
| --- | --- |
| JSON | All string values and object keys. `/records/*/note` selects that member's key and value. JSON numbers, booleans, and null remain unchanged and are not analyzed. Key collisions after replacement fail the request. |
| CSV | Every cell, including headers in row `0`. `/1/0` selects row 1, column 0; `/*/2` selects column 2. Rows and columns are zero-based. Serialization normalizes quoting and uses CRLF line endings. |
| XML | Element text, tails, and attribute values. Elements are indexed in preorder from zero: `/0/text`, `/1/tail`, `/1/attributes/0`. Element/attribute names are retained. Comments and processing instructions are removed; namespace formatting may change. DTDs, entities, and external references are rejected. |

Entity offsets are local to each selected string. JSON locations use numeric
member/list indexes plus `part: "key"` or `"value"`; CSV uses `row` and
`column`; XML uses `element`, `part`, and an attribute index where relevant.
Original JSON key names, CSV headers, and XML attribute names are not copied
into entity location metadata.

Selected fields share identity/replacement context by default. Set
`link_records: true` to also perform detection using their joined context.
This option can change detections and should only join related records. A
document supports at most 4,096 selected text fields and 100,000 selected text
characters including shared-context newline separators, with a 10 MiB input
bound, 64 nesting levels, and 50,000 nodes/cells (each CSV row, including a
blank one, counts as at least one cell).

## Native text files

`/v1/files` accepts `{data, content_type, options, paths, link_records}`.
`data` is standard base64 of the file bytes; the response's `data` is base64
of the transformed file, with `content_type`, entities, and warnings. This is
a JSON endpoint, not a multipart upload or remote-URL fetch. Input files may
contain 1 byte through 10 MiB. Supported content types are:

| Type | `content_type` |
| --- | --- |
| UTF-8 text | `text/plain` |
| JSON | `application/json` |
| CSV | `text/csv` |
| XML | `application/xml` or `text/xml` |
| Word | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| Excel | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| PowerPoint | `application/vnd.openxmlformats-officedocument.presentationml.presentation` |
| Digital PDF | `application/pdf` |

Short type names `txt`, `json`, `csv`, `xml`, `docx`, `xlsx`, `pptx`, and `pdf`
are also accepted. Text and structured files must use UTF-8; a UTF-8 BOM is
accepted. `paths` applies only to JSON, CSV, and XML, using the selectors above.

Office support is deliberately limited to supported text content. Text split
across runs is processed together within a paragraph or cell, then normalized
into the first text run. Word includes supported headers, footers, footnotes,
and endnotes. PowerPoint processes supported paragraph text. Excel processes
strings, sheet names, and supported numeric/date cells; numeric/date output is
text and sheet names are sanitized and made unique. Existing formatting can
change. Document properties, thumbnails, custom XML parts, printer settings,
and nonessential extension trees are removed. Style IDs and bookmark references
are remapped. Package relationship IDs are renumbered (with their references
updated), relationship targets and content-type entries are rebuilt from the
validated part names, and nonstandard relationship or content types are
screened like other attributes. Word tabs and breaks and PowerPoint soft
breaks are analyzed as part of the paragraph text and written back as break
elements. Office locations use numeric `part` indexes with paragraph/cell
coordinates, not original package filenames.

Unsupported Office parts or features fail the request, including images,
embedded objects, macros, external relationships, comments/notes, bound content
controls, populated defined names, tracked changes, formulas, hidden Excel
content, Word hyperlinks and symbol glyphs, PowerPoint custom path geometry,
literal or escaped custom number formats, and OOXML `_xHHHH_` character
escapes. Noncontent XML attributes are screened; a detected identifier that
would need replacement in an unsupported attribute causes rejection instead
of rewriting schema values. The parser does not
silently pass through unsupported binary content. Archives have a 256-member
and 20 MiB expanded-size limit, with path and compression-ratio checks. XML
parts are limited to 50,000 nodes and 64 nesting levels.

PDF support requires the optional `pdf` extra (`pip install -e '.[pdf]'`).
Without it, PDF requests return HTTP 422, and the Python API raises a
`ValueError` naming the extra. PDF support extracts text from digital pages and builds a new PDF from the
transformed text. It does not redact the original PDF in place. Layout, fonts,
graphics, and source metadata are not preserved, and page count can change.
Entity locations refer to source page numbers (starting at one) and offsets
within extracted source-page text, not PDF coordinates. Text with characters
unsupported by the output font is rejected. Encrypted PDFs,
images/embedded graphical objects, annotations, forms, attachments/actions,
and pages without extractable text are rejected. The input page limit is 100,
with at most 256 content streams per page and 20 MiB of decoded content across
the document. Supported page-content encoding is uncompressed, Flate, ASCII85,
or ASCII85 followed by Flate; other encodings or decode parameters are rejected.

## Entity analysis

`/v1/analyze` accepts the detection fields plus `locale` (`en-US` by default,
or `en-GB`) and `include_sensitive` (`false` by default). It adds an `analysis`
object where a supported parser applies, a `relationships` list, and analysis
metadata. It does not rewrite text or expand the detector's coverage.

- `DATE` and `DOB`: calendar parsing with explicit precision and unsupported,
  ambiguous, or invalid status. Numeric day/month interpretation follows
  `locale`. Normalized dates require `include_sensitive: true`.
- `AGE`: supported numeric ages/ranges with units and plausibility checks.
  Normalized age values require `include_sensitive: true`.
- `ADDRESS`: a bounded US postal grammar, with street, optional unit, city,
  state, ZIP, and optional country. Components require `include_sensitive: true`.
- `CREDIT_CARD` and `ROUTING_NUMBER`: Luhn or ABA checksum results. Checksums
  do not verify issuance, ownership, or identity; invalid candidates remain
  detected and available for redaction.

For optional medical terms, supply a dictionary such as
`"medical_terms": {"CONDITION": ["asthma"], "MEDICATION": ["albuterol"]}`.
Those detections can be analyzed or given a per-type action, including `keep`
when the caller intends to preserve the clinical concepts.

Relationship extraction is experimental matching of explicit predicates
between adjacent detected entities on one line. Supported types are `born_on`,
`resides_at`, `contact_at`, `has_age`, `has_condition`, and `takes_medication`.
For example, the supported pattern “NAME can be reached at EMAIL” can produce
`contact_at`. `source_entity` and `target_entity` are zero-based indexes into
the response's `entities`. These rules do not establish clinical truth or
provide general relationship/coreference extraction.

## Usage, diagnostics, and WebSockets

`project_id` defaults to `main` and accepts 1–60 ASCII letters, digits,
underscores, colons, or hyphens. Use an opaque project identifier. Usage reports
contain aggregate requests, documents, characters, successes, failures, and
processing seconds. Projects are represented by the first 16 hexadecimal
characters of SHA-256 of the project ID, with excess distinct projects grouped
into `overflow` after 128 project buckets. No source text or patient anchors
are recorded in these counters. Hashing a project ID is not a secrecy guarantee.

Counts are process-local and reset on restart; separate workers have separate
counters. A batch counts as one processing request for each included project,
with each message or structured field counted as a document. Pre-processing
validation failures and restoration calls are not included in these counters.
Success means the service completed text processing, not that the entire HTTP
operation succeeded: later key-collision checks, native-file reconstruction, or
response serialization can still fail after a processing success is counted.
`/metrics` exposes `deid_active_requests` and counters
`deid_{requests,documents,characters,successes,failures,seconds}_total`.
Diagnostics report runtime/model details without source text or secret values.

Connect to `/v1/ws` with the same `X-API-Key` header. Browser-origin requests
must have an HTTP(S) origin matching the connection's scheme, hostname, and
normalized port (`http` for `ws`, `https` for `wss`); clients without an Origin
header are accepted. Send a JSON text envelope:

```json
{
  "operation": "detect",
  "payload": {"text": "Mara Calder arrived."}
}
```

The reply is `{operation, result}` using the corresponding HTTP result schema,
or `{error: {status, detail}}`. Supported operations are `detect`, `deidentify`,
`analyze`, and `reidentify`. Requests are processed sequentially per connection;
this interface does not stream tokens. Batch, structured, and file operations
use HTTP. Binary messages close with code 1003, oversized messages with 1009,
and rejected authentication/origin with 1008.

Validation and runtime errors omit submitted source values. Application logging
does not include request bodies. Explicit mappings, normalized analysis values,
`keep` actions, unselected structured fields, and undetected identifiers can
still leave sensitive content in results; callers control those options and
the storage or forwarding of returned data.

## Integration smoke check

Run the reusable check against the local trained checkpoint:

```sh
uv run scripts/smoke_extended_api.py --device cpu
```

It writes `reports/extended_api_smoke.json`
by default; use `--model-dir` or `--output` to select another checkpoint or
report path. The recorded run passed its 12 exercised HTTP/WebSocket routes
against checkpoint version `7350eb40e25b1a2b`. It checks invented-input detection,
policy/replacement agreement, processed offsets, restoration, structured/text
file processing, WebSocket results, and operational responses. This is an
integration check, not a clinical accuracy evaluation or validation of every
native document feature.
