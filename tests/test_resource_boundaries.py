"""Exact transport and resource limits, one unit either side of each boundary.

HTTP bodies are driven through the ASGI app with hand-built ``receive``
frames, so multi-frame uploads and mid-upload disconnects are exercised as a
server delivers them rather than as a test client chooses to buffer them.
"""

import asyncio
import io
import json
import re

import pytest
from test_files import StubService, invoke, make_office, make_pdf

import outis.api as api
from outis.api import create_app
from outis.formats.files import MAX_PDF_PAGES, MAX_ZIP_MEMBERS, W
from outis.formats.structured import MAX_NODES, process_structured
from outis.service import Deidentifier
from outis.types import Span

SECRET = b"test-only-secret-never-use-in-production-12345"


class RecordingDetector:
    def __init__(self):
        self.texts = []

    def detect_batch(self, texts):
        self.texts.extend(texts)
        return [
            [Span(m.start(), m.end(), "NAME") for m in re.finditer("Zoë Núñez", t)]
            for t in texts
        ]


def asgi_post(app, frames, *, content_length=None, path="/v1/detect"):
    """POST ``frames`` (bytes, or "disconnect") and return (status, body, read)."""
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "state": {},
    }
    pending = list(frames)
    read = 0
    sent = []

    async def receive():
        nonlocal read
        if not pending:
            # After the body, a server reports disconnect only when the
            # client goes away; block like a live connection would.
            await asyncio.sleep(3600)
        frame = pending.pop(0)
        read += 1
        if frame == "disconnect":
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": frame, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    async def run():
        await asyncio.wait_for(app(scope, receive, send), timeout=10)

    asyncio.run(run())
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return (start["status"] if start else None), body, read


@pytest.fixture
def detector():
    return RecordingDetector()


@pytest.fixture
def service(detector):
    return Deidentifier(detector=detector, secret=SECRET)


@pytest.fixture
def app(service):
    app = create_app(service)
    # The lifespan sets this; the raw ASGI driver sends HTTP scopes only.
    app.state.service = service
    return app


def split(body, sizes):
    frames, start = [], 0
    for size in sizes:
        frames.append(body[start : start + size])
        start += size
    return frames + [body[start:]]


def test_multi_frame_body_is_reassembled_across_a_split_character(app, detector):
    text = "Seen: Zoë Núñez."
    body = json.dumps({"text": text}, ensure_ascii=False).encode()
    cut = body.index("ë".encode()) + 1  # inside the two-byte UTF-8 sequence
    status, response, read = asgi_post(app, [body[:cut], b"", body[cut:]])
    assert status == 200 and read == 3
    assert detector.texts == [text]
    ((entity),) = json.loads(response)["entities"]
    assert text[entity["start"] : entity["end"]] == "Zoë Núñez"


def test_oversized_multi_frame_body_stops_reading_at_the_limit(
    app, detector, monkeypatch
):
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 100)
    frames = [b"x" * 40, b"x" * 40, b"x" * 21, b"x" * 40, b"x" * 40]
    status, response, read = asgi_post(app, frames)
    assert status == 413
    assert json.loads(response) == {"detail": "Request body too large"}
    assert read == 3  # the 101st byte arrives in frame three; nothing after
    assert detector.texts == []


@pytest.mark.parametrize("declared", [True, False])
def test_body_of_exactly_the_limit_is_accepted(app, detector, monkeypatch, declared):
    body = json.dumps({"text": "Zoë Núñez."}, ensure_ascii=False).encode()
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", len(body))
    status, _, _ = asgi_post(
        app, split(body, [7, 5]), content_length=len(body) if declared else None
    )
    assert status == 200
    assert detector.texts == ["Zoë Núñez."]


@pytest.mark.parametrize("declared", [True, False])
def test_body_one_byte_over_the_limit_is_rejected(app, detector, monkeypatch, declared):
    body = json.dumps({"text": "Zoë Núñez."}, ensure_ascii=False).encode()
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", len(body) - 1)
    status, _, read = asgi_post(
        app, split(body, [7, 5]), content_length=len(body) if declared else None
    )
    assert status == 413
    # A declared length is refused before any of the body is read.
    assert read == (0 if declared else 3)
    assert detector.texts == []


@pytest.mark.parametrize("length", ["-1", "12abc", "99999999999999999999999"])
def test_invalid_declared_length_is_rejected_unread(app, detector, length):
    status, _, read = asgi_post(app, [b"{}"], content_length=length)
    assert status in {400, 413} and read == 0
    assert detector.texts == []


@pytest.mark.parametrize("at", [0, 1, 2])
def test_disconnect_during_upload_sends_nothing_and_runs_nothing(
    app, service, detector, at
):
    body = json.dumps({"text": "Zoë Núñez."}).encode()
    frames = split(body, [5, 5])[:at] + ["disconnect"]
    status, response, read = asgi_post(app, frames)
    assert (status, response, read) == (None, b"", at + 1)
    assert detector.texts == []
    assert service.usage.snapshot()["projects"] == {}


# CSV ----------------------------------------------------------------------


@pytest.mark.parametrize("row", ["", "a", "a,b"])
def test_csv_exact_cell_budget_is_accepted_and_one_more_is_rejected(row):
    # A blank row holds no cells but is charged one.
    per_row = len(row.split(",")) if row else 1
    rows = MAX_NODES // per_row
    assert rows * per_row == MAX_NODES
    # One leading "a" keeps a selectable field; field-count limits are separate.
    lines = ["a" + "," * (per_row - 1)] + [row] * (rows - 1)
    accepted = "\n".join(lines) + "\n"
    result = process_structured(StubService(), accepted, format="csv", paths=["/0/0"])
    assert result["data"].count("\r\n") == rows
    with pytest.raises(ValueError, match="row or cell limits"):
        process_structured(
            StubService(), accepted + row + "\n", format="csv", paths=["/0/0"]
        )


def test_csv_rejects_on_budget_before_reading_the_rest():
    # Past the budget the input becomes malformed. Incremental enforcement
    # reports the limit; parsing everything first would report the quote.
    source = "\n".join(["a"] * (MAX_NODES + 1)) + '\n"unterminated'
    with pytest.raises(ValueError, match="row or cell limits"):
        process_structured(StubService(), source, format="csv")


# Native files --------------------------------------------------------------


def docx_with_members(count):
    # Unreferenced footer parts are allowed members; pad to an exact count.
    base = 4  # [Content_Types].xml, _rels/.rels, docProps/core.xml, document
    footer = f'<w:ftr xmlns:w="{W}"><w:p><w:r><w:t>x</w:t></w:r></w:p></w:ftr>'
    parts = {f"word/footer{index}.xml": footer for index in range(count - base)}
    return make_office("docx", parts)


def test_archive_member_count_boundary():
    import zipfile

    raw = docx_with_members(MAX_ZIP_MEMBERS)
    assert len(zipfile.ZipFile(io.BytesIO(raw)).infolist()) == MAX_ZIP_MEMBERS
    invoke(raw, "docx")
    with pytest.raises(ValueError, match="too many archive members"):
        invoke(docx_with_members(MAX_ZIP_MEMBERS + 1), "docx")


def pdf_with_pages(count):
    from reportlab.pdfgen.canvas import Canvas

    buf = io.BytesIO()
    canvas = Canvas(buf)
    for index in range(count):
        canvas.drawString(20, 750, f"Page {index} for Jane Doe.")
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def test_pdf_page_count_boundary():
    pytest.importorskip("pypdf")
    result = invoke(pdf_with_pages(MAX_PDF_PAGES), "pdf")
    assert result["entity_count"] == MAX_PDF_PAGES
    with pytest.raises(ValueError, match="between 1 and 100 pages"):
        invoke(pdf_with_pages(MAX_PDF_PAGES + 1), "pdf")


def pdf_with_content_streams(count):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject

    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(io.BytesIO(make_pdf())))
    page = writer.pages[0]
    original = page["/Contents"].get_object().get_data()
    streams = []
    for index in range(count):
        stream = DecodedStreamObject()
        stream.set_data(original if index == 0 else b" ")
        streams.append(writer._add_object(stream))
    page[NameObject("/Contents")] = ArrayObject(streams)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_content_stream_count_boundary():
    pytest.importorskip("pypdf")
    result = invoke(pdf_with_content_streams(256), "pdf")
    assert result["entity_count"] == 1
    with pytest.raises(ValueError, match="too many content streams"):
        invoke(pdf_with_content_streams(257), "pdf")
