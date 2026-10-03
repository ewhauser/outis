"""Bounded native file processing; no fetching, OCR, or binary pass-through.

OOXML support is deliberately conservative. Unsupported parts/features are rejected,
metadata is removed, and text split across runs is detected as one logical string.
PDF output is a newly generated text document and does not preserve page layout.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import posixpath
import re
import zipfile
import zlib
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET

from .structured import _entity_result, _process_texts, process_structured

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_ZIP_MEMBERS = 256
MAX_PDF_PAGES = 100
_FONT_LOCK = Lock()
MIME_TYPES = {
    "txt": "text/plain",
    "json": "application/json",
    "csv": "text/csv",
    "xml": "application/xml",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "pdf": "application/pdf",
}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
R = "http://schemas.openxmlformats.org/package/2006/relationships"
C = "http://schemas.openxmlformats.org/package/2006/content-types"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _decode(data):
    if not isinstance(data, str) or len(data) > ((MAX_FILE_BYTES + 2) // 3) * 4:
        raise ValueError("File must be base64 text encoding at most 10 MiB")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("File data must be valid base64") from None
    if not decoded or len(decoded) > MAX_FILE_BYTES:
        raise ValueError("File must contain between 1 byte and 10 MiB")
    return decoded


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique")
        result[key] = value
    return result


def _xml(data):
    try:
        parser = SafeET.iterparse(io.BytesIO(data), events=("start", "end"), forbid_dtd=True, forbid_entities=True, forbid_external=True)
        depth = count = 0
        for event, _ in parser:
            if event == "start":
                count += 1
                depth += 1
                if count > 50_000 or depth > 64:
                    raise ValueError("XML exceeds node or depth limits")
            else:
                depth -= 1
        return parser.root
    except Exception:
        raise ValueError("File contains invalid or unsafe XML") from None


def _tag(element):
    return element.tag.rsplit("}", 1)[-1] if isinstance(element.tag, str) else ""


def _zip_members(raw):
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
        if not 1 <= len(infos) <= MAX_ZIP_MEMBERS:
            raise ValueError("Office file contains too many archive members")
        names = set()
        total = 0
        files = {}
        for info in infos:
            name = info.filename
            if (name in names or name.startswith("/") or "\\" in name or
                    "\x00" in name or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))):
                raise ValueError("Office file contains an unsafe or duplicate archive path")
            names.add(name)
            if info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Encrypted archives and symbolic links are unsupported")
            total += info.file_size
            if total > MAX_ZIP_BYTES or info.file_size > MAX_ZIP_BYTES:
                raise ValueError("Office file expands beyond the 20 MiB limit")
            if info.file_size > 1024 * 1024 and info.file_size > max(1, info.compress_size) * 200:
                raise ValueError("Office file has an unsafe compression ratio")
            if not info.is_dir():
                files[name] = archive.read(info)
        archive.close()
        return files
    except ValueError:
        raise
    except Exception:
        raise ValueError("File must be a valid unencrypted Office archive") from None


def _allowed_part(name, kind):
    common = {"[Content_Types].xml", "_rels/.rels", "docProps/core.xml", "docProps/app.xml", "docProps/thumbnail.jpeg", "docProps/thumbnail.png"}
    if name in common:
        return True
    if kind == "docx":
        return bool(re.fullmatch(r"word/(?:document|styles|stylesWithEffects|settings|webSettings|fontTable|numbering|footnotes|endnotes|header\d+|footer\d+)\.xml|word/theme/theme\d+\.xml|word/_rels/(?:document|header\d+|footer\d+|footnotes|endnotes)\.xml\.rels", name))
    if kind == "xlsx":
        return bool(re.fullmatch(r"xl/(?:workbook|styles|sharedStrings)\.xml|xl/theme/theme\d+\.xml|xl/worksheets/sheet\d+\.xml|xl/_rels/workbook\.xml\.rels", name))
    return bool(re.fullmatch(r"ppt/(?:presentation|presProps|viewProps|tableStyles)\.xml|ppt/(?:slides/slide|slideLayouts/slideLayout|slideMasters/slideMaster|theme/theme)\d+\.xml|ppt/_rels/presentation\.xml\.rels|ppt/(?:slides|slideLayouts|slideMasters)/_rels/(?:slide|slideLayout|slideMaster)\d+\.xml\.rels", name))


def _relationship_target(name, target):
    if target.startswith("/"):
        return target.lstrip("/")
    # Relationships live in a _rels directory alongside their source part.
    return posixpath.normpath(posixpath.join(posixpath.dirname(posixpath.dirname(name)), target))


def _sanitize_package(files, kind):
    required = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "pptx": "ppt/presentation.xml"}[kind]
    if required not in files or "[Content_Types].xml" not in files or "_rels/.rels" not in files:
        raise ValueError("Office archive does not match the requested content type")
    roots = {}
    removed = {name for name in files if name.startswith(("docProps/", "customXml/", "ppt/printerSettings/"))}
    for name, payload in files.items():
        if name in removed:
            continue
        if not _allowed_part(name, kind):
            raise ValueError("Office file contains unsupported parts; remove media, embedded objects, notes, comments, custom XML, and macros first")
        root = _xml(payload)
        roots[name] = root
        if name.endswith(".rels"):
            if root.tag != f"{{{R}}}Relationships":
                raise ValueError("Invalid Office relationships")
            for rel in list(root):
                target = rel.get("Target", "")
                if rel.get("TargetMode", "").lower() == "external" or ":" in target or "\\" in target:
                    raise ValueError("External Office relationships are unsupported")
                resolved = _relationship_target(name, target)
                if resolved in removed:
                    root.remove(rel)
                elif resolved not in files:
                    raise ValueError("Office file contains an unresolved relationship")
        elif name == "[Content_Types].xml":
            for item in list(root):
                if item.get("PartName", "").lstrip("/") in removed:
                    root.remove(item)
                if any(x in item.get("ContentType", "").lower() for x in ("macro", "vba", "activex", "oleobject")):
                    raise ValueError("Executable Office content is unsupported")
        for parent in root.iter():
            for child in list(parent):
                if _tag(child) in {"extLst", "prnPr"} or (_tag(child) == "definedNames" and not len(child)):
                    parent.remove(child)
        for item in root.iter():
            tag = _tag(item)
            if any(re.search(r"_x[0-9a-fA-F]{4}_", value) for value in [item.text or "", item.tail or "", *item.attrib.values()]):
                raise ValueError("Office escaped character encodings are unsupported")
            if tag in {"custGeom", "sym", "hyperlink", "instrText", "fldSimple", "altChunk", "object", "pict", "drawing", "del", "ins", "delText", "moveFrom", "moveTo", "docVars", "dataBinding", "externalData", "f", "formula", "definedNames", "extLst", "oleObj", "graphicFrame", "pic", "audio", "video", "sdt", "AlternateContent", "customXml", "autoFilter", "sortState", "dataValidations", "conditionalFormatting", "hyperlinks", "headerFooter"}:
                raise ValueError("Office file contains unsupported fields, formulas, changes, embedded content, or extensions")
            if kind == "xlsx" and tag == "numFmt" and any(char in item.get("formatCode", "") for char in ('"', "\\")):
                raise ValueError("Spreadsheet contains unsupported attributes in a literal or escaped custom number format")
            if kind == "xlsx" and (item.get("state") in {"hidden", "veryHidden"} or item.get("hidden") in {"1", "true"}):
                raise ValueError("Hidden worksheets, rows, and columns are unsupported")
            # Non-content descriptions are metadata, not safe text-preserving parts.
            for attr in list(item.attrib):
                if attr.rsplit("}", 1)[-1] in {"descr", "title", "author", "initials", "tooltip", "Ignorable"}:
                    del item.attrib[attr]
            if tag in {"cNvPr", "cSld", "theme", "clrScheme", "fontScheme", "fmtScheme"} and "name" in item.attrib:
                item.set("name", "Deidentified")
    expected = {"docx": f"{{{W}}}document", "xlsx": f"{{{S}}}workbook", "pptx": f"{{{P}}}presentation"}[kind]
    if roots[required].tag != expected:
        raise ValueError("Office document root does not match the requested content type")
    return roots


def _date_styles(roots):
    root = roots.get("xl/styles.xml")
    if root is None:
        return set()
    custom = {}
    for item in root.iter(f"{{{S}}}numFmt"):
        fmt = re.sub(r'"[^\"]*"|\\\\.|\[[^]]*\]', "", item.get("formatCode", "")).lower()
        if re.search(r"[ydhs]|m[/\-]", fmt):
            custom[item.get("numFmtId")] = True
    cell_xfs = root.find(f"{{{S}}}cellXfs")
    if cell_xfs is None:
        return set()
    builtins = set(range(14, 23)) | set(range(27, 37)) | {45, 46, 47} | set(range(50, 59))
    return {str(i) for i, item in enumerate(cell_xfs) if int(item.get("numFmtId", "0")) in builtins or item.get("numFmtId") in custom}


def _office(service, raw, kind, options, link_records):
    roots = _sanitize_package(_zip_members(raw), kind)
    groups = []
    locations = []
    setters = []
    claimed_text = set()
    claimed_attributes = set()
    warnings = ["Document properties, thumbnails, custom XML, printer settings, and nonessential extensions were removed. Text split across runs is normalized within each paragraph or cell."]

    def add(nodes, location, value=None, setter=None):
        claimed_text.update(id(node) for node in nodes)
        text = "".join(node.text or "" for node in nodes) if value is None else value
        if not text:
            return
        groups.append(text)
        locations.append(location)
        if setter is not None:
            setters.append(setter)
        else:
            def set_text(new, ns=nodes):
                ns[0].text = new
                ns[0].set(XML_SPACE, "preserve")
                for node in ns[1:]:
                    node.text = ""
            setters.append(set_text)

    date_styles = _date_styles(roots) if kind == "xlsx" else set()
    workbook = roots.get("xl/workbook.xml")
    is_1904 = workbook is not None and any(x.get("date1904") in {"1", "true"} for x in workbook.iter(f"{{{S}}}workbookPr"))
    style_ids = {}
    bookmark_ids = {}
    for root in roots.values():
        for item in root.iter():
            if item.tag == f"{{{W}}}style" and item.get(f"{{{W}}}styleId"):
                key = item.get(f"{{{W}}}styleId")
                style_ids.setdefault(key, f"style_{len(style_ids)}")
            if item.tag == f"{{{W}}}bookmarkStart":
                key = item.get(f"{{{W}}}name", "")
                bookmark_ids.setdefault(key, f"bookmark_{len(bookmark_ids)}")
            if item.tag == f"{{{W}}}hyperlink" and item.get(f"{{{W}}}anchor"):
                key = item.get(f"{{{W}}}anchor")
                bookmark_ids.setdefault(key, f"bookmark_{len(bookmark_ids)}")
    for part_index, (part, root) in enumerate(roots.items()):
        if part.endswith(".rels") or part == "[Content_Types].xml":
            continue
        if kind == "docx":
            for item in root.iter():
                if item.tag == f"{{{W}}}style" and item.get(f"{{{W}}}styleId") in style_ids:
                    item.set(f"{{{W}}}styleId", style_ids[item.get(f"{{{W}}}styleId")])
                if _tag(item) in {"pStyle", "rStyle", "tblStyle", "basedOn", "next", "link"} and item.get(f"{{{W}}}val") in style_ids:
                    item.set(f"{{{W}}}val", style_ids[item.get(f"{{{W}}}val")])
                if item.tag == f"{{{W}}}bookmarkStart":
                    item.set(f"{{{W}}}name", bookmark_ids[item.get(f"{{{W}}}name", "")])
                if item.tag == f"{{{W}}}hyperlink" and item.get(f"{{{W}}}anchor") in bookmark_ids:
                    item.set(f"{{{W}}}anchor", bookmark_ids[item.get(f"{{{W}}}anchor")])
            if part.startswith("word/"):
                # Collect every w:t, including headers, footnotes and text boxes.
                used = set()
                for i, paragraph in enumerate(root.iter(f"{{{W}}}p")):
                    nodes = list(paragraph.iter(f"{{{W}}}t"))
                    if any(id(node) in used for node in nodes):
                        raise ValueError("Nested Word paragraphs are unsupported")
                    used.update(id(node) for node in nodes)
                    value = "".join((node.text or "") if node.tag == f"{{{W}}}t" else ("\t" if _tag(node) == "tab" else "\n") for node in paragraph.iter() if node.tag == f"{{{W}}}t" or _tag(node) in {"tab", "br", "cr"})
                    if nodes:
                        add(nodes, {"part": part_index, "paragraph": i}, value)
                        for parent in paragraph.iter():
                            for child in list(parent):
                                if _tag(child) in {"tab", "br", "cr"}:
                                    parent.remove(child)
                if any(id(node) not in used for node in root.iter(f"{{{W}}}t")):
                    raise ValueError("Word file contains text outside supported paragraphs")
                for i, item in enumerate(root.iter(f"{{{W}}}lvlText")):
                    add([], {"part": part_index, "numbering_index": i}, item.get(f"{{{W}}}val", ""), lambda new, item=item: item.set(f"{{{W}}}val", new))
                    claimed_attributes.add((id(item), f"{{{W}}}val"))
        elif kind == "pptx":
            for i, paragraph in enumerate(root.iter(f"{{{A}}}p")):
                add(list(paragraph.iter(f"{{{A}}}t")), {"part": part_index, "paragraph": i})
        else:
            for i, item in enumerate(root.iter(f"{{{S}}}si")):
                add(list(item.iter(f"{{{S}}}t")), {"part": part_index, "shared_string": i})
            for i, cell in enumerate(root.iter(f"{{{S}}}c")):
                kind_attr = cell.get("t", "n")
                if kind_attr == "s":
                    shared_index_nodes = list(cell.iter(f"{{{S}}}v"))
                    claimed_text.update(id(node) for node in shared_index_nodes)
                    shared_root = roots.get("xl/sharedStrings.xml")
                    string_count = len(list(shared_root.iter(f"{{{S}}}si"))) if shared_root is not None else 0
                    if len(shared_index_nodes) != 1 or not re.fullmatch(r"[0-9]+", shared_index_nodes[0].text or "") or int(shared_index_nodes[0].text) >= string_count:
                        raise ValueError("Spreadsheet contains an invalid shared string reference")
                    continue
                nodes = list(cell.iter(f"{{{S}}}t")) if kind_attr == "inlineStr" else list(cell.iter(f"{{{S}}}v"))
                value = "".join(node.text or "" for node in nodes)
                if kind_attr in {"e", "b"} or not value:
                    if kind_attr == "b" and value not in {"0", "1"}:
                        raise ValueError("Spreadsheet contains an invalid boolean cell")
                    if kind_attr == "e" and value not in {"#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A", "#GETTING_DATA"}:
                        raise ValueError("Spreadsheet contains an unsupported error cell")
                    claimed_text.update(id(node) for node in nodes)
                    continue
                if cell.get("s") in date_styles and kind_attr == "n":
                    try:
                        serial = float(value)
                        epoch = datetime(1904, 1, 1) if is_1904 else datetime(1899, 12, 31) if serial < 60 else datetime(1899, 12, 30)
                        value = (epoch + timedelta(days=serial)).isoformat(sep=" ").removesuffix(" 00:00:00")
                    except (ValueError, OverflowError):
                        raise ValueError("Spreadsheet contains an invalid date cell") from None
                def set_cell(new, node=cell, original=value, original_kind=kind_attr):
                    if new == original and original_kind not in {"n", "d"}:
                        # Existing string nodes can be collapsed safely.
                        ns = list(node.iter(f"{{{S}}}t")) or list(node.iter(f"{{{S}}}v"))
                        if ns:
                            ns[0].text = new
                            for other in ns[1:]:
                                other.text = ""
                        return
                    for child in list(node):
                        node.remove(child)
                    node.set("t", "inlineStr")
                    node.attrib.pop("s", None)
                    text_node = ET.SubElement(ET.SubElement(node, f"{{{S}}}is"), f"{{{S}}}t")
                    text_node.set(XML_SPACE, "preserve")
                    text_node.text = new
                add(nodes, {"part": part_index, "cell_index": i}, value, set_cell)
            for i, sheet in enumerate(root.iter(f"{{{S}}}sheet")):
                def set_sheet(new, item=sheet, index=i):
                    new = re.sub(r"[\\/*?:\[\]]", "_", new).strip("'")[:25] or "Sheet"
                    item.set("name", f"{new}_{index + 1}")
                add([], {"part": part_index, "sheet_index": i}, sheet.get("name", ""), set_sheet)
                claimed_attributes.add((id(sheet), "name"))
    # Inspect every remaining textual payload. Schema attributes cannot be
    # arbitrarily rewritten without corrupting the document, so detected entities
    # in unsupported attributes fail closed. Text-bearing content is transformed.
    for part_index, (part, root) in enumerate(roots.items()):
        if part.endswith(".rels") or part == "[Content_Types].xml":
            continue
        attributes = []
        for item_index, item in enumerate(root.iter()):
            if id(item) not in claimed_text and item.text and item.text.strip():
                add([item], {"part": part_index, "element": item_index})
            if item.tail and item.tail.strip():
                add([], {"part": part_index, "element": item_index, "tail": True}, item.tail,
                    lambda new, item=item: setattr(item, "tail", new))
            for attr, value in item.attrib.items():
                if (id(item), attr) not in claimed_attributes and value:
                    # Well-known relationship IDs and machine-generated run IDs
                    # are schema linkage, not text. All other values are screened.
                    local = attr.rsplit("}", 1)[-1]
                    if local in {"Ignorable", "space"} or local.startswith("rsid") or local in {"paraId", "textId"}:
                        continue
                    attributes.append(value)
        if attributes:
            original = "\n".join(dict.fromkeys(attributes))
            def check_attributes(new, old=original):
                if new != old:
                    raise ValueError("Office file contains detected identifiers in unsupported attributes; remove custom styles or metadata first")
            add([], {"part": part_index, "attributes": True}, original, check_attributes)
    results = _process_texts(service, groups, options=options, link_records=link_records)
    for result, setter in zip(results, setters):
        setter(result["text"])
    metadata = _entity_result(results, locations)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for part, root in roots.items():
            archive.writestr(part, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    if kind == "xlsx":
        warnings.append("Numeric and date cells are returned as text; sheet names are sanitized and made unique.")
    return output.getvalue(), metadata, warnings


def _bounded_pdf_content(page, remaining):
    """Preflight supported page stream filters before pypdf builds text objects."""
    streams = page.get("/Contents")
    if streams is None:
        return 0
    streams = streams.get_object()
    if not isinstance(streams, list):
        streams = [streams]
    if len(streams) > 256:
        raise ValueError("PDF contains too many content streams")
    used = 0
    for ref in streams:
        stream = ref.get_object()
        payload = getattr(stream, "_data", None)
        if not isinstance(payload, bytes) or stream.get("/DecodeParms"):
            raise ValueError("PDF content stream uses unsupported encoding parameters")
        filters = stream.get("/Filter", [])
        filters = filters.get_object() if hasattr(filters, "get_object") else filters
        filters = [filters] if isinstance(filters, str) else list(filters)
        if filters not in ([], ["/FlateDecode"], ["/ASCII85Decode"], ["/ASCII85Decode", "/FlateDecode"]):
            raise ValueError("PDF content stream encoding is unsupported")
        try:
            for codec in filters:
                if codec == "/ASCII85Decode":
                    payload = base64.a85decode(payload, adobe=True)
                else:
                    decoder = zlib.decompressobj()
                    payload = decoder.decompress(payload, max(1, remaining - used + 1))
                    if not decoder.eof or decoder.unconsumed_tail:
                        raise ValueError("PDF decoded content exceeds the 20 MiB limit")
            used += len(payload)
            if used > remaining:
                raise ValueError("PDF decoded content exceeds the 20 MiB limit")
        except (zlib.error, binascii.Error):
            raise ValueError("PDF content stream encoding is invalid") from None
    return used


def _pdf(service, raw, options, link_records):
    from pypdf import PdfReader, apply_configuration
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from xml.sax.saxutils import escape

    try:
        with apply_configuration(
            maximum_declared_stream_length=MAX_ZIP_BYTES,
            array_based_stream_maximum_output_length=MAX_ZIP_BYTES,
            zlib_maximum_output_length=MAX_ZIP_BYTES,
            lzw_maximum_output_length=MAX_ZIP_BYTES,
            run_length_maximum_output_length=MAX_ZIP_BYTES,
            page_tree_maximum_entries=MAX_PDF_PAGES * 4,
            page_tree_maximum_depth=32,
        ):
            reader = PdfReader(io.BytesIO(raw), strict=True)
            if reader.is_encrypted:
                raise ValueError("Encrypted PDF files are unsupported")
            root = reader.trailer["/Root"]
            if any(key in root for key in ("/AcroForm", "/Names", "/OpenAction", "/AA")):
                raise ValueError("PDF forms, attachments, actions, and named content are unsupported")
            if not 1 <= len(reader.pages) <= MAX_PDF_PAGES:
                raise ValueError("PDF must contain between 1 and 100 pages")
            texts = []
            total = 0
            decoded_bytes = 0
            for page in reader.pages:
                resources = page.get("/Resources", {})
                resources = resources.get_object() if hasattr(resources, "get_object") else resources
                if page.get("/Annots") or resources.get("/XObject"):
                    raise ValueError("PDF images, annotations, and embedded graphical objects are unsupported; OCR is not available")
                decoded_bytes += _bounded_pdf_content(page, MAX_ZIP_BYTES - decoded_bytes)
                text = page.extract_text() or ""
                if not text.strip():
                    raise ValueError("PDF contains a page without extractable text; OCR is not available")
                total += len(text)
                if total > 100_000:
                    raise ValueError("PDF text exceeds the 100000 character limit")
                texts.append(text)
    except ValueError:
        raise
    except Exception:
        raise ValueError("File must be a valid supported digital PDF") from None
    results = _process_texts(service, texts, options=options, link_records=link_records)
    out = io.BytesIO()
    story = []
    with _FONT_LOCK:
        if "ClinicalDeidVera" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ClinicalDeidVera", str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf")))
    font = pdfmetrics.getFont("ClinicalDeidVera")
    if any(not char.isspace() and ord(char) not in font.face.charToGlyph for result in results for char in result["text"]):
        raise ValueError("PDF text contains characters unsupported by the output font")
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "ClinicalDeidVera"
    for i, result in enumerate(results):
        if i:
            story.append(PageBreak())
        for line in result["text"].splitlines():
            story.append(Paragraph(escape(line) or "&#160;", styles["BodyText"]))
            story.append(Spacer(1, 3))

    # Rebuild from transformed text. No original PDF objects/metadata are copied.
    SimpleDocTemplate(out, pagesize=A4, title="Deidentified document", author="", subject="").build(story)
    return out.getvalue(), _entity_result(results, [{"page": i + 1} for i in range(len(results))]), ["PDF was rebuilt from extracted text. Original layout, fonts, graphics, and metadata are not preserved; page count may change."]


def process_file(service, *, data, content_type, options=None, paths=None, link_records=False):
    """Return a base64 redacted native file. Offsets identify extracted source text."""
    if not isinstance(content_type, str):
        raise ValueError("content_type must identify a supported file type")
    mime = content_type.lower().split(";", 1)[0].strip()
    mime = {"text/xml": "application/xml", **MIME_TYPES}.get(mime, mime)
    kind = next((key for key, value in MIME_TYPES.items() if value == mime), None)
    if kind is None:
        raise ValueError("Unsupported content type; supported types are txt, json, csv, xml, docx, xlsx, pptx, and digital pdf")
    if paths is not None and kind not in {"json", "csv", "xml"}:
        raise ValueError("Field selection is supported only for JSON, CSV, and XML files")
    raw = _decode(data)
    if kind in {"txt", "json", "csv", "xml"}:
        try:
            value = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("Text and structured files must use UTF-8 encoding") from None
        if kind == "txt":
            result = _process_texts(service, [value], options=options, link_records=link_records)[0]
            output = result["text"].encode("utf-8")
            result = _entity_result([result], [{"format": "text", "segment": 0}])
        else:
            if kind == "json":
                try:
                    value = json.loads(value, object_pairs_hook=_json_object, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("JSON constants must be finite")))
                except (ValueError, RecursionError):
                    raise ValueError("File must contain valid JSON with unique keys and finite numbers") from None
            result = process_structured(service, value, format=kind, paths=paths, options=options, link_records=link_records)
            output_value = json.dumps(result["data"], ensure_ascii=False) if kind == "json" else result["data"]
            output = output_value.encode("utf-8")
        metadata = {key: result[key] for key in ("entities", "entity_count", "mappings", "model_version") if key in result}
        warnings = result.get("warnings", [])
    elif kind == "pdf":
        output, metadata, warnings = _pdf(service, raw, options, link_records)
    else:
        output, metadata, warnings = _office(service, raw, kind, options, link_records)
    return {
        "data": base64.b64encode(output).decode("ascii"),
        "content_type": mime,
        **metadata,
        "warnings": list(dict.fromkeys(warnings + metadata.get("warnings", []))),
        "model_version": metadata.get("model_version", getattr(service, "version", "unknown")),
    }
