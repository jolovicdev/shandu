from __future__ import annotations

import csv
import io
import json
import re
import threading
from collections.abc import Iterable
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlsplit

import lxml.etree
import lxml.html
from bs4 import BeautifulSoup

from .constants import (
    _CAPTCHA_PATTERNS,
    _ERROR_PAGE_MAX_WORDS,
    _ERROR_PAGE_TITLE,
    _MAX_EXTRACTED_CHARS,
    _MAX_PAGE_LINKS,
    _MAX_XLSX_ROWS,
    _MIN_ARTICLE_WORDS,
    _PAYWALL_PATTERNS,
)
from .models import ContentBlock, PageLink, _ExtractionResult, _ParseError


def _cap_text(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_EXTRACTED_CHARS:
        return text
    return text[:_MAX_EXTRACTED_CHARS].rstrip()


def _cap_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    capped: list[ContentBlock] = []
    current_len = 0
    for block in blocks:
        block_len = len(block.text) + 1
        if current_len + block_len > _MAX_EXTRACTED_CHARS:
            if not capped:
                capped.append(
                    ContentBlock(type=block.type, text=block.text[:_MAX_EXTRACTED_CHARS].rstrip())
                )
            break
        capped.append(block)
        current_len += block_len
    return capped


def _cap_extraction_result(result: _ExtractionResult) -> _ExtractionResult:
    if len(result.text) <= _MAX_EXTRACTED_CHARS:
        return result
    blocks = _cap_blocks(result.blocks) if result.blocks else []
    return _ExtractionResult(
        title=result.title,
        text=_cap_text(result.text),
        blocks=blocks,
        published_at=result.published_at,
        site_name=result.site_name,
    )


_XML_CAPTURED_TAGS = {"head", "p", "item", "quote", "code", "table", "figcaption"}
_HTML_CAPTURED_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "pre",
    "code",
    "table",
    "figcaption",
}


def _trafilatura_xml_to_blocks(xml_str: str) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    if not xml_str:
        return blocks
    try:
        soup = BeautifulSoup(xml_str, "lxml-xml")
        for elem in soup.find_all(True):
            # A node nested in an already-captured ancestor (p in quote,
            # p in table cell) would emit its text twice.
            if any(parent.name in _XML_CAPTURED_TAGS for parent in elem.parents):
                continue
            tag = elem.name
            text = elem.get_text(" ", strip=True)
            if not text:
                continue
            if tag == "head":
                blocks.append(ContentBlock(type="heading", text=text))
            elif tag == "p":
                blocks.append(ContentBlock(type="paragraph", text=text))
            elif tag == "item":
                blocks.append(ContentBlock(type="list_item", text=text))
            elif tag == "quote":
                blocks.append(ContentBlock(type="blockquote", text=text))
            elif tag == "code":
                blocks.append(ContentBlock(type="code", text=text))
            elif tag == "table":
                rows: list[str] = []
                for row in elem.find_all("row"):
                    cells = [cell.get_text(" ", strip=True) for cell in row.find_all("cell")]
                    rows.append(" | ".join(cells))
                blocks.append(ContentBlock(type="table", text="\n".join(rows)))
            elif tag == "figcaption":
                blocks.append(ContentBlock(type="caption", text=text))
    except Exception:
        pass
    return blocks


def _soup_to_blocks(soup: BeautifulSoup) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for node in soup.find_all(sorted(_HTML_CAPTURED_TAGS)):
        if any(parent.name in _HTML_CAPTURED_TAGS for parent in node.parents):
            continue
        tag_name = node.name
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            blocks.append(ContentBlock(type="heading", text=text))
        elif tag_name == "li":
            blocks.append(ContentBlock(type="list_item", text=text))
        elif tag_name == "blockquote":
            blocks.append(ContentBlock(type="blockquote", text=text))
        elif tag_name in ("pre", "code"):
            blocks.append(ContentBlock(type="code", text=text))
        elif tag_name == "table":
            rows: list[str] = []
            for tr in node.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(" | ".join(cells))
            blocks.append(ContentBlock(type="table", text="\n".join(rows)))
        elif tag_name == "figcaption":
            blocks.append(ContentBlock(type="caption", text=text))
        else:
            blocks.append(ContentBlock(type="paragraph", text=text))
    return blocks


def _extract_with_trafilatura(html: str) -> _ExtractionResult | None:
    try:
        import trafilatura

        xml_result = trafilatura.extract(
            html,
            output_format="xml",
            with_metadata=True,
            include_comments=False,
            include_tables=True,
            include_images=False,
        )
        if not xml_result:
            return None
        # Title and date are attributes of <doc>; the body carries the blocks.
        doc = BeautifulSoup(xml_result, "lxml-xml").find("doc")
        title = (doc.get("title", "") if doc is not None else "").strip()
        published_at = (doc.get("date", "") if doc is not None else "").strip() or None
        blocks = _trafilatura_xml_to_blocks(xml_result)
        text = "\n".join(block.text for block in blocks).strip()
        if len(text.split()) < _MIN_ARTICLE_WORDS:
            return None
        return _cap_extraction_result(
            _ExtractionResult(title=title, text=text, blocks=blocks, published_at=published_at)
        )
    except Exception:
        return None


def _extract_with_readability(html: str) -> _ExtractionResult | None:
    try:
        from readability import Document

        doc = Document(html)
        title = (doc.title() or "").strip()
        summary = doc.summary()
        if not summary or len(summary) < 100:
            return None
        soup = BeautifulSoup(summary, "lxml")
        blocks = _soup_to_blocks(soup)
        text = "\n".join(b.text for b in blocks).strip()
        if len(text.split()) < _MIN_ARTICLE_WORDS:
            return None
        return _cap_extraction_result(_ExtractionResult(title=title, text=text, blocks=blocks))
    except Exception:
        return None


def _extract_with_bs4(html: str) -> _ExtractionResult:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("script,style,noscript,header,footer,nav,aside,form,iframe,svg"):
        tag.decompose()

    title = _extract_title_from_soup(soup)
    section = (
        soup.find("article")
        or soup.find("main")
        or soup.select_one("[role='main']")
        or soup.find("body")
        or soup
    )

    blocks = _soup_to_blocks(section)
    total_words = len(" ".join(b.text for b in blocks).split())
    if total_words < 120:
        blocks = []
        for line in _clean_blocks(line for line in section.get_text("\n").splitlines()):
            blocks.append(ContentBlock(type="paragraph", text=line))

    text = "\n".join(b.text for b in blocks).strip()
    if len(text) > _MAX_EXTRACTED_CHARS:
        truncated = _cap_blocks(blocks)
        text = "\n".join(b.text for b in truncated).strip()
        blocks = truncated

    return _ExtractionResult(title=title, text=text, blocks=blocks)


_ARXIV_STAMP = re.compile(
    r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[[\w.-]+\]\s*(\d{1,2} [A-Z][a-z]{2} \d{4})"
)


def _arxiv_stamp_date(html: str) -> str | None:
    # arXiv HTML papers carry no date metadata, only the version stamp
    # "arXiv:2309.17453v4 [cs.CL] 07 Apr 2024"; without it the date falls to
    # a guess from body text, which lands on dates inside the references.
    match = _ARXIV_STAMP.search(html)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%d %b %Y").date().isoformat()
    except ValueError:
        return None


def _extract_html(html: str) -> _ExtractionResult:
    result = _extract_with_trafilatura(html)
    if result is None:
        result = _extract_with_readability(html)
    if result is None:
        result = _extract_with_bs4(html)
    result.site_name = _extract_site_name(html)
    result.published_at = _arxiv_stamp_date(html) or result.published_at
    return result


_FITZ_BOLD_FLAG = 16


def _inked_size(spans: list[dict[str, Any]]) -> float:
    return max((span["size"] for span in spans if span.get("text", "").strip()), default=0.0)


def _horizontal_lines(block: dict[str, Any]) -> list[list[dict[str, Any]]]:
    # Preprint servers stamp a rotated identifier in the margin in larger type
    # than the title.
    return [
        line.get("spans", [])
        for line in block.get("lines", [])
        if tuple(line.get("dir", (1, 0))) == (1, 0) and _inked_size(line.get("spans", []))
    ]


def _title_text(lines: list[list[dict[str, Any]]]) -> str:
    # Word spaces can be spans of their own, so spans join with nothing between.
    title = " ".join(
        " ".join("".join(span.get("text", "") for span in spans) for spans in lines).split()
    )
    return title if 10 < len(title) < 300 else ""


def _pdf_layout_title(page: Any) -> str:
    """Title from the first-page block set in the largest horizontal type.

    Within that block, every line set near the largest size is kept whole,
    since small caps put most letters of a title in smaller spans than its
    capitals.
    """
    best_size = 0.0
    best_lines: list[list[dict[str, Any]]] = []
    for block in page.get_text("dict").get("blocks", []):
        lines = _horizontal_lines(block)
        if not lines:
            continue
        size = max(_inked_size(spans) for spans in lines)
        if size > best_size:
            best_size = size
            best_lines = lines
    return _title_text([spans for spans in best_lines if _inked_size(spans) >= 0.9 * best_size])


def _pdf_bold_title(page: Any) -> str:
    """Title from the first mostly bold block in the top third of the page.

    Some journal styles set the title in bold at body size, so the largest
    type on the page belongs to the body text.
    """
    for block in page.get_text("dict").get("blocks", []):
        if block.get("bbox", (0, 0, 0, 0))[1] > page.rect.height / 3:
            continue
        lines = _horizontal_lines(block)
        inked = [span for spans in lines for span in spans if span.get("text", "").strip()]
        bold = sum(len(span["text"]) for span in inked if span.get("flags", 0) & _FITZ_BOLD_FLAG)
        if 2 * bold > sum(len(span["text"]) for span in inked):
            title = _title_text(lines)
            if title:
                return title
    return ""


# Document templates leave their placeholders in the PDF title field: the bare
# word "Title", a style hint, a font note, or only the document date.
_PDF_PLACEHOLDER_TITLE = re.compile(
    r"^title$|style here\W*\)$|\(\s*[\w ]+\d+(?:\.\d+)?\s*pt\s*\)$|^[a-z]+ \d{1,2}, \d{4}$",
    re.IGNORECASE,
)


def _pdf_creation_date(metadata: dict[str, Any]) -> str | None:
    # PDF dates look like D:20251029105853-04'00'.
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", metadata.get("creationDate") or "")
    return "-".join(match.groups()) if match else None


# MuPDF is not thread-safe and pages are parsed in worker threads, so one
# document is read at a time. Reading several at once needs a process each.
_PDF_LOCK = threading.Lock()


def _parse_pdf(data: bytes) -> _ExtractionResult:
    try:
        import fitz

        with _PDF_LOCK:
            doc = fitz.open(stream=data, filetype="pdf")
            try:
                metadata = doc.metadata or {}
                title = (metadata.get("title") or "").strip()
                if _PDF_PLACEHOLDER_TITLE.search(title):
                    title = ""
                if not title and len(doc) > 0:
                    title = _pdf_layout_title(doc[0]) or _pdf_bold_title(doc[0])
                # A long report is read from the front until the text cap is
                # full, which keeps its summary and opening chapters.
                full_text: list[str] = []
                length = 0
                for page in doc:
                    page_text = page.get_text()
                    full_text.append(page_text)
                    length += len(page_text)
                    if length >= _MAX_EXTRACTED_CHARS:
                        break
                text = _cap_text("\n".join(full_text))
                if not text:
                    raise _ParseError("pdf_parse_failed", "Empty PDF text")
                return _ExtractionResult(
                    title=title, text=text, published_at=_pdf_creation_date(metadata)
                )
            finally:
                doc.close()
    except _ParseError:
        raise
    except Exception as exc:
        raise _ParseError("pdf_parse_failed", str(exc)) from exc


def _parse_docx(data: bytes) -> _ExtractionResult:
    try:
        from docx import Document
        from docx.oxml.ns import qn

        doc = Document(io.BytesIO(data))
        title = (doc.core_properties.title or "").strip()
        lines: list[str] = []
        chars = 0
        for child in doc.element.body:
            if chars >= _MAX_EXTRACTED_CHARS:
                break
            if child.tag == qn("w:p"):
                pending = [
                    "".join(
                        node.text or "" for node in child.iter(qn("w:t"))
                    ).strip()
                ]
            elif child.tag == qn("w:tbl"):
                pending = []
                for row in child.iter(qn("w:tr")):
                    cells = [
                        "".join(
                            node.text or "" for node in cell.iter(qn("w:t"))
                        ).strip()
                        for cell in row.iter(qn("w:tc"))
                    ]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        pending.append(" | ".join(cells))
            else:
                continue
            for line in pending:
                if not line or chars >= _MAX_EXTRACTED_CHARS:
                    continue
                lines.append(line)
                chars += len(line) + 1
        if not title and lines:
            for line in lines:
                if len(line) > 5:
                    title = line
                    break
        text = _cap_text("\n".join(lines))
        if not text:
            raise _ParseError("non_text_content", "Empty DOCX")
        return _ExtractionResult(title=title, text=text)
    except _ParseError:
        raise
    except Exception as exc:
        raise _ParseError("non_text_content", str(exc)) from exc


def _parse_xlsx(data: bytes) -> _ExtractionResult:
    try:
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        try:
            rows: list[str] = []
            row_count = 0
            for sheet in wb:
                for row in sheet.iter_rows(values_only=True):
                    cells = [str(c) for c in row if c is not None]
                    if cells:
                        rows.append(" | ".join(cells))
                    row_count += 1
                    if row_count >= _MAX_XLSX_ROWS:
                        break
                if row_count >= _MAX_XLSX_ROWS:
                    break
            text = _cap_text("\n".join(rows))
            if not text:
                raise _ParseError("non_text_content", "Empty XLSX")
            title = wb.properties.title or ""
        finally:
            wb.close()
        return _ExtractionResult(title=title, text=text)
    except _ParseError:
        raise
    except Exception as exc:
        raise _ParseError("non_text_content", str(exc)) from exc


def _parse_csv(data: bytes) -> _ExtractionResult:
    try:
        text = data.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        rows: list[str] = []
        for row in reader:
            if row:
                rows.append(" | ".join(row))
        text = _cap_text("\n".join(rows))
        if not text:
            raise _ParseError("non_text_content", "Empty CSV")
        return _ExtractionResult(title="", text=text)
    except _ParseError:
        raise
    except Exception as exc:
        raise _ParseError("non_text_content", str(exc)) from exc


def _parse_plaintext(data: bytes) -> _ExtractionResult:
    text = _cap_text(data.decode("utf-8", errors="ignore"))
    if not text:
        raise _ParseError("non_text_content", "Empty text file")
    return _ExtractionResult(title="", text=text)


_UTF8_HTML_PARSER = lxml.html.HTMLParser(encoding="utf-8")


def _extract_links(html: str, base_url: str) -> list[PageLink]:
    """Outbound links with their anchor text, links to other hosts first.

    Links to other hosts are where a page cites its sources, so they lead;
    site chrome is dropped before the scan so menus do not fill the list.
    """
    # Plain lxml: a second BeautifulSoup tree per page costs about 20 times
    # more. UTF-8 bytes, because lxml rejects a str that carries an XML
    # encoding declaration.
    try:
        document = lxml.html.fromstring(html.encode("utf-8"), parser=_UTF8_HTML_PARSER)
    except lxml.etree.ParserError:
        return []
    for tag in document.xpath("//nav|//header|//footer|//aside|//script|//style"):
        tag.drop_tree()
    base_host = (urlsplit(base_url).hostname or "").removeprefix("www.")
    external: list[PageLink] = []
    internal: list[PageLink] = []
    seen = {base_url}
    for anchor in document.iter("a"):
        href = anchor.get("href")
        if not href:
            continue
        url = urljoin(base_url, href.strip()).split("#", 1)[0]
        text = " ".join(anchor.text_content().split())
        if not url.startswith(("http://", "https://")) or url in seen or len(text) < 3:
            continue
        seen.add(url)
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        target = internal if host == base_host else external
        target.append(PageLink(url=url, text=text[:120]))
    return (external + internal)[:_MAX_PAGE_LINKS]


def _extract_published_at(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")

    og_time = soup.find("meta", attrs={"property": "article:published_time"})
    if og_time and og_time.get("content"):
        return str(og_time["content"]).strip()

    for script in soup.select("script[type='application/ld+json']"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for key in ("datePublished", "dateCreated", "dateModified", "uploadDate"):
                pub = data.get(key)
                if pub:
                    return str(pub).strip()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key in ("datePublished", "dateCreated", "dateModified", "uploadDate"):
                        pub = item.get(key)
                        if pub:
                            return str(pub).strip()

    meta_names = (
        "date",
        "pubdate",
        "dc.date",
        "dc.date.issued",
        "prism.publicationDate",
        "sailthru.date",
        "parsely-pub-date",
        "datePublished",
        "dateModified",
        "uploadDate",
    )
    for meta_name in meta_names:
        tag = soup.find("meta", attrs={"name": meta_name})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return str(time_tag["datetime"]).strip()

    return None


def _detect_fetch_error(html: str | None, text: str) -> str | None:
    if html:
        if len(html) > 8000 and len(text.split()) < 40:
            return "empty_js_shell"

        weak_text = len(text.split()) < 80
        if weak_text:
            html_lower = html.lower()
            if any(p.search(html_lower) for p in _CAPTCHA_PATTERNS):
                return "captcha"
            if any(p.search(html_lower) for p in _PAYWALL_PATTERNS):
                return "paywall"

        title = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if (
            title
            and len(text.split()) < _ERROR_PAGE_MAX_WORDS
            and _ERROR_PAGE_TITLE.search(unescape(title.group(1)))
        ):
            return "error_page"

    return None


def _error_for_status(status: int | None) -> str:
    if status == 403:
        return "blocked_403"
    if status == 401:
        return "login_required"
    if status == 404:
        return "scrape_failed"
    if status == 410:
        return "scrape_failed"
    if status is not None and status >= 500:
        return "scrape_failed"
    return "scrape_failed"


def _guess_format(url: str, content_type: str) -> str:
    ct = content_type.lower()
    if "html" in ct or "xhtml" in ct:
        return "html"
    if "pdf" in ct:
        return "pdf"
    if "wordprocessingml.document" in ct or "msword" in ct:
        return "docx"
    if "spreadsheetml.sheet" in ct or "excel" in ct:
        return "xlsx"
    if "csv" in ct:
        return "csv"
    if "text/plain" in ct:
        return "txt"
    if "text/markdown" in ct or "text/x-markdown" in ct:
        return "md"

    path = urlsplit(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith(".docx"):
        return "docx"
    if path.endswith(".xlsx"):
        return "xlsx"
    if path.endswith(".csv"):
        return "csv"
    if path.endswith(".txt"):
        return "txt"
    if path.endswith(".md"):
        return "md"

    if ct.startswith("text/"):
        return "html"

    if not ct:
        return "html"

    return ""


def _extract_site_name(html: str) -> str | None:
    tag = BeautifulSoup(html, "lxml").find("meta", attrs={"property": "og:site_name"})
    content = tag.get("content") if tag is not None else None
    cleaned = " ".join(str(content).split()) if content else ""
    return cleaned[:160] or None


def _extract_title_from_soup(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return str(og_title["content"]).strip()
    title_tag = soup.find("title")
    if title_tag and title_tag.text:
        return title_tag.text.strip()
    h1 = soup.find("h1")
    if h1 and h1.text:
        return h1.text.strip()
    return ""


def _clean_blocks(lines: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for raw in lines:
        line = " ".join(raw.split()).strip()
        if len(line) < 35:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        yield line
