"""A self-contained local HTML report for reviewing spans before release.

The report shows each note's original text with its supplied spans beside the
replaced output. It contains PHI by design: it is written owner-only, says so on
the page, and makes no network requests. A Content-Security-Policy that admits
only the page's own inline style and script (by hash) enforces that.

A reviewer can remove or add spans in the page and download a corrected span
file that ``outis batch --spans``, ``outis spans replace`` and ``/v1/replace``
accept. The output column
is not re-rendered in the browser; replace again with the corrected file.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections import Counter

PHI_WARNING = (
    "This report contains original note text (protected health information). "
    "Keep it on an approved, access-controlled machine; do not email, upload, "
    "or commit it, and delete it when the review is finished."
)
# Colour-blind-safe hues; labels cycle through them in sorted order.
_PALETTE = (
    ("#cfe2ff", "#1f4e99"),
    ("#ffe0b3", "#8a4b00"),
    ("#d4f0d4", "#1e6b1e"),
    ("#f7d1e3", "#8c1d55"),
    ("#e3d9f7", "#4b2a8c"),
    ("#d2f0ee", "#14625c"),
    ("#fff1a8", "#6b5a00"),
    ("#f1d9cf", "#7a3416"),
    ("#dde3ea", "#33475b"),
)
_CHANGE_FIELDS = ("start", "end", "label", "score", "source")

_STYLE = """
:root { color-scheme: light; --fg: #1b1f24; --muted: #5b6573; --line: #d6dbe1;
  --bg: #ffffff; --panel: #f6f8fa; --warn-bg: #fff4e5; --warn: #8a3b00; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--fg); background: var(--bg); }
header, main { max-width: 1400px; margin: 0 auto; padding: 0 16px; }
h1 { font-size: 1.4rem; margin: 20px 0 8px; }
h2 { font-size: 1.1rem; margin: 0; overflow-wrap: anywhere; }
.phi { background: var(--warn-bg); color: var(--warn); border: 1px solid #f0c08a;
  border-radius: 6px; padding: 10px 14px; margin: 12px 0; font-weight: 600; }
.meta { color: var(--muted); font-size: 0.9rem; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 0.9rem; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: left; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
nav.toc { max-height: 240px; overflow: auto; border: 1px solid var(--line);
  border-radius: 6px; margin-bottom: 16px; }
nav.toc table { margin: 0; width: 100%; }
.tools { position: sticky; top: 0; z-index: 2; background: var(--bg);
  border-bottom: 1px solid var(--line); padding: 8px 0; display: flex;
  flex-wrap: wrap; gap: 8px; align-items: center; }
.tools button, .tools select { font: inherit; padding: 4px 10px; }
#status { color: var(--muted); font-size: 0.9rem; }
section.note { border: 1px solid var(--line); border-radius: 8px; margin: 16px 0;
  padding: 12px 14px; scroll-margin-top: 56px; }
.note-head { display: flex; flex-wrap: wrap; justify-content: space-between;
  gap: 8px; align-items: baseline; }
.note-nav a { margin-left: 10px; }
.counts { margin: 6px 0 10px; }
.chip { display: inline-block; border-radius: 10px; padding: 0 8px; margin: 2px 4px 2px 0;
  font-size: 0.8rem; }
.panes { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 800px) { .panes { grid-template-columns: 1fr; } }
.pane h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted); margin: 0 0 4px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px; padding: 10px;
  font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
mark { border-radius: 3px; padding: 0 1px; }
mark::after { content: attr(data-label); font: 600 0.65rem/1 system-ui, sans-serif;
  vertical-align: super; margin-left: 2px; }
.original mark { cursor: pointer; }
mark.removed { text-decoration: line-through; opacity: 0.45; }
mark.added { outline: 2px dashed currentColor; }
mark.kept { opacity: 0.6; }
.failed { color: var(--warn); font-weight: 600; margin: 0; }
footer { color: var(--muted); font-size: 0.85rem; margin: 24px 0; }
"""

# The editing model is pure (no DOM), so tests can run it under Node. Every
# rendered character maps to exactly one source code point: runs never overlap,
# and each run records its source start offset.
MODEL_SCRIPT = """
const ReviewModel = (() => {
  "use strict";
  const points = (value) => Array.from(value).length;
  const overlaps = (a, start, end) => a.start < end && start < a.end;

  /* Display runs covering the text exactly once, in order. */
  function runs(note) {
    const chars = Array.from(note.text);
    const order = note.entities.map((entity, index) => [entity, index])
      .sort((a, b) => a[0].start - b[0].start || a[0].end - b[0].end);
    const result = [];
    let at = 0;
    const plain = (end) => {
      if (end > at) {
        result.push({ start: at, end, text: chars.slice(at, end).join(""), index: null });
      }
    };
    for (const [entity, index] of order) {
      if (entity.start < at) continue; /* never show a character twice */
      plain(entity.start);
      result.push({
        start: entity.start,
        end: entity.end,
        text: chars.slice(entity.start, entity.end).join(""),
        index,
      });
      at = entity.end;
    }
    plain(chars.length);
    return result;
  }

  /* A source offset from a run start and the run text before the boundary. */
  const offset = (runStart, textBefore) => runStart + points(textBefore);

  function add(note, start, end, label) {
    if (note.failed) {
      return "Detection failed for this note; detect it again before reviewing.";
    }
    const length = points(note.text);
    if (!Number.isInteger(start) || !Number.isInteger(end)
        || start < 0 || end > length || end <= start) {
      return "Select at least one character.";
    }
    if (note.entities.some((e) => !e.removed && overlaps(e, start, end))) {
      return "The selection overlaps a span; remove that span first.";
    }
    /* A removed span under the new one can no longer be restored. */
    note.entities = note.entities.filter((e) => !(e.removed && overlaps(e, start, end)));
    note.entities.push({ start, end, label, source: "review", added: true });
    note.changed = true;
    return null;
  }

  function toggle(note, index) {
    const entity = note.entities[index];
    if (entity.added) {
      note.entities.splice(index, 1);
      note.changed = true;
      return ["Removed an added span.", true];
    }
    if (entity.removed && note.entities.some(
      (e, i) => i !== index && !e.removed && overlaps(e, entity.start, entity.end))) {
      return ["Another span now covers this text.", false];
    }
    entity.removed = !entity.removed;
    note.changed = true;
    return [entity.removed ? "Span marked for removal." : "Span restored.", true];
  }

  function exportLines(notes) {
    return notes.map((note) => {
      const record = Object.assign({}, note.record);
      record.entities = note.entities.filter((e) => !e.removed)
        .sort((a, b) => a.start - b.start)
        .map((e) => {
          const entity = { start: e.start, end: e.end, label: e.label };
          if (e.score !== undefined) entity.score = e.score;
          if (e.source !== undefined) entity.source = e.source;
          return entity;
        });
      if (note.changed) record.edited = true;
      return JSON.stringify(record) + "\\n";
    }).join("");
  }

  return { runs, offset, add, toggle, exportLines };
})();
"""

_DOM_SCRIPT = """
(() => {
  "use strict";
  const model = ReviewModel;
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const status = document.getElementById("status");
  const labelSelect = document.getElementById("label");
  let edits = 0;
  const say = (message) => { status.textContent = message; };

  function render(n) {
    const note = data.notes[n];
    const box = document.getElementById("original-" + n);
    box.replaceChildren();
    for (const run of model.runs(note)) {
      let element;
      if (run.index === null) {
        element = document.createElement("span");
      } else {
        const entity = note.entities[run.index];
        element = document.createElement("mark");
        element.className = [
          "l" + (data.labels.indexOf(entity.label) % data.palette),
          entity.removed ? "removed" : "",
          entity.added ? "added" : "",
          entity.kept ? "kept" : "",
        ].join(" ").trim();
        element.dataset.i = String(run.index);
        element.dataset.label = entity.label;
        element.title = entity.label
          + (entity.kept ? " (left in place by the ages policy)" : "")
          + " - click to " + (entity.removed ? "restore" : "remove");
      }
      element.dataset.s = String(run.start);
      element.append(document.createTextNode(run.text));
      box.append(element);
    }
  }

  function changed(n) {
    edits += 1;
    document.getElementById("download").disabled = false;
    render(n);
  }

  document.addEventListener("click", (event) => {
    const mark = event.target.closest(".original mark");
    if (!mark) return;
    const n = Number(mark.parentElement.dataset.note);
    const [message, edited] = model.toggle(data.notes[n], Number(mark.dataset.i));
    say(message);
    if (edited) changed(n);
  });

  /* Source offset of a selection boundary, from the run that contains it. */
  function boundary(box, node, at) {
    if (node === box) {
      const run = box.childNodes[at];
      return run ? Number(run.dataset.s) : Array.from(data.notes[Number(box.dataset.note)].text).length;
    }
    const run = node.nodeType === 3 ? node.parentElement : node;
    const text = run.textContent;
    const before = node.nodeType === 3 ? node.data.slice(0, at) : at ? text : "";
    return model.offset(Number(run.dataset.s), before);
  }

  document.getElementById("add").addEventListener("click", () => {
    const selection = window.getSelection();
    if (!selection.rangeCount || selection.isCollapsed) {
      say("Select text in an original note first.");
      return;
    }
    const range = selection.getRangeAt(0);
    const container = (node) => (node.nodeType === 1 ? node : node.parentElement).closest(".original");
    const box = container(range.startContainer);
    if (!box || box !== container(range.endContainer)) {
      say("Select text inside one original note.");
      return;
    }
    const n = Number(box.dataset.note);
    const start = boundary(box, range.startContainer, range.startOffset);
    const end = boundary(box, range.endContainer, range.endOffset);
    const label = labelSelect.value;
    const error = model.add(data.notes[n], start, end, label);
    if (error) {
      say(error);
      return;
    }
    selection.removeAllRanges();
    say("Added a span labelled " + label + ".");
    changed(n);
  });

  document.getElementById("download").addEventListener("click", () => {
    const blob = new Blob([model.exportLines(data.notes)], { type: "application/x-ndjson" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "spans.reviewed.jsonl";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
    say("Downloaded a corrected span file with " + edits + " edit(s). It contains note and patient IDs.");
  });

  data.notes.forEach((_, n) => render(n));
})();
"""
_SCRIPT = MODEL_SCRIPT + _DOM_SCRIPT


def _csp_hash(source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def _text(value: str) -> str:
    # The HTML parser would turn CR into LF and NUL into U+FFFD; the script
    # re-renders exactly from the embedded data, this is the no-script view.
    return html.escape(value, quote=True).replace("\r", "&#13;")


def _marked(text: str, spans, label_class) -> str:
    """Escape text and wrap each (start, end, label, extra class) span."""
    chars, parts, cursor = list(text), [], 0
    for start, end, label, extra in sorted(spans):
        parts.append(_text("".join(chars[cursor:start])))
        classes = " ".join(filter(None, (label_class(label), extra)))
        parts.append(
            f'<mark class="{classes}" data-label="{_text(label)}" '
            f'title="{_text(label)}">{_text("".join(chars[start:end]))}</mark>'
        )
        cursor = end
    parts.append(_text("".join(chars[cursor:])))
    return "".join(parts)


def _json_script(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Keep "</script>" and comment openers out of the raw text element.
    return (
        encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )


def _chips(counts: Counter, label_class) -> str:
    return "".join(
        f'<span class="chip {label_class(label)}">{_text(label)} {count}</span>'
        for label, count in sorted(counts.items())
    )


def build_report(pairs, results, *, settings: dict | None = None) -> str:
    """Build the report from (note, span record) pairs and replace results.

    Results must use the extended response shape (with output offsets), which
    any request that sets ``replacement`` returns.
    """
    from outis.policy import EXTENDED_TYPES, KNOWN_LABELS, MEDICAL_TYPES
    from outis.types import ENTITY_TYPES

    # Every supported label, so a reviewer can add a type detection missed
    # entirely, then any custom labels the records use.
    labels = [*ENTITY_TYPES, *EXTENDED_TYPES, *MEDICAL_TYPES]
    labels += sorted(
        {entity["label"] for _, record in pairs for entity in record["entities"]}
        - KNOWN_LABELS
    )

    def label_class(label):
        return f"l{labels.index(label) % len(_PALETTE)}"

    totals, notes, sections, toc = Counter(), [], [], []
    for n, ((note, record), result) in enumerate(zip(pairs, results)):
        text = note["text"]
        failed = record.get("error") if result is None else None
        if failed:
            # No spans to review: show the text, refuse edits, keep the record.
            result = {"text": "", "entities": []}
        replaced = {(e["start"], e["end"]) for e in result["entities"]}
        entities = []
        for entity in record["entities"]:
            item = {key: entity[key] for key in _CHANGE_FIELDS if key in entity}
            if (entity["start"], entity["end"]) not in replaced:
                item["kept"] = True
            entities.append(item)
        notes.append(
            {
                "text": text,
                "entities": entities,
                "record": {k: v for k, v in record.items() if k != "entities"},
                **({"failed": failed} if failed else {}),
            }
        )
        counts = Counter(entity["label"] for entity in result["entities"])
        totals.update(counts)
        original = _marked(
            text,
            [
                (e["start"], e["end"], e["label"], "kept" if e.get("kept") else "")
                for e in entities
            ],
            label_class,
        )
        output = _marked(
            result["text"],
            [
                (e["processed_start"], e["processed_end"], e["label"], "")
                for e in result["entities"]
            ],
            label_class,
        )
        note_id = _text(str(note["note_id"]))
        if failed:
            output = (
                '<p class="failed">Detection failed for this note '
                f"({_text(failed)}), so it has no spans to review. Detect it "
                "again before release.</p>"
            )
        previous = f'<a href="#note-{n - 1}">Previous</a>' if n else ""
        following = f'<a href="#note-{n + 1}">Next</a>' if n + 1 < len(pairs) else ""
        kept = len(entities) - len(result["entities"])
        sections.append(
            f'<section class="note" id="note-{n}">'
            f'<div class="note-head"><h2>{note_id}</h2>'
            f'<span class="note-nav">{previous}{following}<a href="#top">Top</a></span>'
            f"</div>"
            f'<div class="counts">{_chips(counts, label_class)}'
            + (
                '<span class="meta">detection failed'
                if failed
                else f'<span class="meta">{len(result["entities"])} replaced'
            )
            + (f", {kept} left in place by the ages policy" if kept else "")
            + "</span></div>"
            f'<div class="panes"><div class="pane"><h3>Original (PHI)</h3>'
            f'<div class="text original" id="original-{n}" data-note="{n}">'
            f"{original}</div></div>"
            f'<div class="pane"><h3>Replaced output</h3>'
            f'<div class="text">{output}</div></div></div></section>'
        )
        toc.append(
            f'<tr><td><a href="#note-{n}">{note_id}</a></td>'
            f'<td class="n">{"failed" if failed else len(result["entities"])}</td></tr>'
        )

    palette = "".join(
        f".l{i} {{ background: {bg}; color: {fg}; }}\n"
        for i, (bg, fg) in enumerate(_PALETTE)
    )
    style = _STYLE + palette
    script = _SCRIPT
    csp = (
        "default-src 'none'; "
        f"style-src {_csp_hash(style)}; script-src {_csp_hash(script)}; "
        "img-src 'none'; connect-src 'none'; base-uri 'none'; form-action 'none'"
    )
    data = {"notes": notes, "labels": labels, "palette": len(_PALETTE)}
    summary = "".join(
        f'<tr><td><span class="chip {label_class(label)}">{_text(label)}</span></td>'
        f'<td class="n">{count}</td></tr>'
        for label, count in sorted(totals.items())
    )
    versions = sorted(
        {record.get("model_version") for _, record in pairs} - {None}
    ) or ["not recorded"]
    setting_rows = "".join(
        f"<tr><th>{_text(str(key))}</th><td>{_text(str(value))}</td></tr>"
        for key, value in (settings or {}).items()
    )
    options = "".join(
        f'<option value="{_text(label)}">{_text(label)}</option>' for label in labels
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<meta name="referrer" content="no-referrer">
<meta name="robots" content="noindex, nofollow">
<title>Outis span review</title>
<style>{style}</style>
</head>
<body>
<header id="top">
<h1>Outis span review</h1>
<p class="phi" role="alert">{_text(PHI_WARNING)}</p>
<p class="meta">{len(pairs)} notes, {sum(totals.values())} replaced spans.
Detector version: {_text(", ".join(versions))}.</p>
<table>{setting_rows}</table>
<table><thead><tr><th>Label</th><th>Replaced</th></tr></thead>
<tbody>{summary}</tbody></table>
<nav class="toc"><table><thead><tr><th>Note</th><th>Spans</th></tr></thead>
<tbody>{"".join(toc)}</tbody></table></nav>
</header>
<main>
<div class="tools">
<label for="label">Label</label><select id="label">{options}</select>
<button id="add" type="button">Add span from selection</button>
<button id="download" type="button" disabled>Download corrected spans</button>
<span id="status" aria-live="polite">Click a highlighted span to remove it;
select original text and choose a label to add one.</span>
</div>
{"".join(sections)}
<footer>Generated locally by Outis. This file makes no network requests.
Corrected span files contain note and patient IDs; store them like the notes.</footer>
</main>
<script type="application/json" id="review-data">{_json_script(data)}</script>
<script>{script}</script>
</body>
</html>
"""
