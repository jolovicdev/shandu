from __future__ import annotations

import asyncio
import io

import aiohttp
import fitz
import time
from bs4 import BeautifulSoup
from docx import Document
from openpyxl import Workbook
from unittest import mock

import pytest

from shandu.services.scrape import ScrapeService
from shandu.services.scrape.constants import _MAX_DOWNLOAD_BYTES
from shandu.services.scrape.extraction import (
    _detect_fetch_error,
    _extract_html,
    _extract_published_at,
    _extract_with_bs4,
    _extract_with_readability,
    _extract_with_trafilatura,
    _parse_csv,
    _parse_docx,
    _parse_pdf,
    _parse_plaintext,
    _parse_xlsx,
    _soup_to_blocks,
    _trafilatura_xml_to_blocks,
)
from shandu.services.scrape.scheduler import _DomainScheduler


def _long_html(title: str = "Title", words: int = 100) -> str:
    text = " ".join(["word"] * words)
    return f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1><p>{text}</p><p>{text}</p></article></body></html>"



@pytest.mark.parametrize(
    "extract,title",
    [
        (_extract_with_trafilatura, "Trafilatura Article"),
        (_extract_with_readability, "Readability Article"),
        (_extract_with_bs4, "BS4 Article"),
    ],
)
def test_extract_backends_return_title_text_and_blocks(extract, title) -> None:
    result = extract(_long_html(title, words=100))
    assert result is not None
    assert result.title == title
    assert len(result.text.split()) >= 100
    assert len(result.blocks) > 0


def test_extract_html_captures_og_site_name() -> None:
    html = (
        '<html><head><title>Article</title>'
        '<meta property="og:site_name" content="Example News" />'
        "</head><body><article><h1>Article</h1>"
        f"<p>{' '.join(['word'] * 120)}</p></article></body></html>"
    )

    assert _extract_html(html).site_name == "Example News"
    assert _extract_html(_long_html("No Site Name", words=120)).site_name is None


def test_extract_cascade_prefers_trafilatura_then_readability_then_bs4() -> None:
    result = _extract_html(_long_html("Cascade", words=120))
    assert result.title == "Cascade"
    assert len(result.text.split()) >= 120


def test_trafilatura_returns_none_for_short_content() -> None:
    html = "<html><body><article><p>Short text.</p></article></body></html>"
    assert _extract_with_trafilatura(html) is None


def test_readability_returns_none_for_short_content() -> None:
    html = "<html><body><article><p>Short text.</p></article></body></html>"
    assert _extract_with_readability(html) is None


def test_soup_to_blocks_emits_nested_text_once() -> None:
    soup = BeautifulSoup(
        "<article>"
        "<ul><li><p>list item text</p></li></ul>"
        "<blockquote><p>quoted text</p></blockquote>"
        "<pre><code>code text</code></pre>"
        "<table><tr><td><p>cell text</p></td></tr></table>"
        "</article>",
        "lxml",
    )

    texts = [block.text for block in _soup_to_blocks(soup)]

    assert texts == [
        "list item text",
        "quoted text",
        "code text",
        "cell text",
    ]


def test_trafilatura_xml_to_blocks_skips_nested_nodes() -> None:
    xml_str = (
        "<doc><main>"
        "<quote><p>quoted text</p></quote>"
        "<table><row><cell><p>cell text</p></cell></row></table>"
        "</main></doc>"
    )

    blocks = _trafilatura_xml_to_blocks(xml_str)

    assert [(block.type, block.text) for block in blocks] == [
        ("blockquote", "quoted text"),
        ("table", "cell text"),
    ]


def test_trafilatura_extracts_title_date_and_blocks_in_one_pass() -> None:
    filler = " ".join(["word"] * 80)
    xml_str = (
        '<doc title="Mocked Title" date="2026-03-01"><main>'
        "<head>Mocked Title</head>"
        f"<p>{filler}</p>"
        "</main></doc>"
    )
    with mock.patch("trafilatura.extract", return_value=xml_str) as extract:
        result = _extract_with_trafilatura("<html></html>")

    assert extract.call_count == 1
    assert result is not None
    assert result.title == "Mocked Title"
    assert result.published_at == "2026-03-01"
    assert [block.type for block in result.blocks] == ["heading", "paragraph"]
    assert len(result.text.split()) >= 80



def test_extract_html_takes_a_latexml_title_without_its_author_footnote() -> None:
    body = " ".join(["The outage began when one content update reached every sensor."] * 12)
    html = (
        "<html><head><title>Channel File 291: An Engineering Analysis</title></head><body>"
        '<article class="ltx_document"><h1 class="ltx_title ltx_title_document">'
        "Channel File 291: An Engineering Analysis"
        '<span class="ltx_pubnotes"><span class="ltx_role_thanks">'
        "Thanks: The author is with Purdue University.</span></span></h1>"
        f"<p>{body}</p></article></body></html>"
    )
    assert _extract_html(html).title == "Channel File 291: An Engineering Analysis"


def test_extract_html_keeps_the_spaces_where_a_latexml_title_breaks_lines() -> None:
    body = " ".join(["Query visibility changes how compression methods rank."] * 12)
    html = (
        "<html><head><title>How Query Visibility ChangesKV-Cache Rankings:A Matched Audit"
        '</title></head><body><article class="ltx_document">'
        '<h1 class="ltx_title ltx_title_document">How Query Visibility Changes\n'
        '<br class="ltx_break">KV-Cache Rankings:<br class="ltx_break">A Matched\N{ZERO WIDTH SPACE} Audit'
        '<span class="ltx_note ltx_role_thanks"><sup>1</sup>Work done at a lab.</span>'
        ' at <math alttext="z=10"><semantics><mi>z</mi><mo>=</mo><mn>10</mn>'
        '<annotation encoding="application/x-tex">z=10</annotation></semantics></math></h1>'
        f"<p>{body}</p></article></body></html>"
    )
    assert _extract_html(html).title == (
        "How Query Visibility Changes KV-Cache Rankings: A Matched Audit at z=10"
    )


def test_parse_pdf_extracts_title_and_text() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF Document Title")
    page.insert_text((72, 100), "This is the body text of the PDF document.")
    data = doc.tobytes()
    result = _parse_pdf(data)
    assert result.title == "PDF Document Title"
    assert "body text" in result.text


def test_parse_pdf_reads_a_long_report_from_the_front() -> None:
    doc = fitz.open()
    for number in range(260):
        doc.new_page().insert_text((72, 72), f"Chapter text on page {number}.")

    result = _parse_pdf(doc.tobytes())

    assert "Chapter text on page 0." in result.text
    assert "Chapter text on page 259." in result.text


def test_parse_pdf_prefers_largest_type_for_title_and_reads_creation_date() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Proceedings of the 2025 Conference, pages 1-17", fontsize=8)
    page.insert_text((72, 110), "Lookahead Caching for Long Contexts", fontsize=18)
    page.insert_text((300, 500), "preprint:2405.14366v2 7 Sep 2024", fontsize=24, rotate=90)
    page.insert_text((72, 150), "This is the body text of the paper.")
    doc.set_metadata({"creationDate": "D:20251029105853-04'00'"})
    result = _parse_pdf(doc.tobytes())
    assert result.title == "Lookahead Caching for Long Contexts"
    assert result.published_at == "2025-10-29"


def test_parse_pdf_reads_small_caps_titles_whole() -> None:
    doc = fitz.open()
    page = doc.new_page()
    # Small caps: capitals at 18 pt, the rest at 14 pt. The word space is set
    # in another font, so it comes out as a span of its own.
    for baseline, spans in [
        (100, [("SCB", 18), ("ENCH", 14), (": KV", 18), (" ", 18), ("C", 18), ("ACHE", 14)]),
        (121, [("L", 17.6), ("ONG", 14), ("-C", 17.6), ("ONTEXT", 14)]),
    ]:
        x = 72.0
        for text, size in spans:
            fontname = "tiro" if text == " " else "helv"
            page.insert_text((x, baseline), text, fontsize=size, fontname=fontname)
            x += fitz.get_text_length(text, fontsize=size, fontname=fontname)
    page.insert_text((72, 136), "Yucheng Li, Huiqiang Jiang", fontsize=10)
    result = _parse_pdf(doc.tobytes())
    assert result.title == "SCBENCH: KV CACHE LONG-CONTEXT"


@pytest.mark.parametrize(
    ("first", "second", "body", "title"),
    [
        (
            "Cache Compression by Pyra-",
            "midal Funneling",
            "Attention narrows in a pyramidal shape across layers.",
            "Cache Compression by Pyramidal Funneling",
        ),
        (
            "Retrieval-",
            "augmented Generation at Scale",
            "Retrieval-augmented generation grounds answers in documents.",
            "Retrieval-augmented Generation at Scale",
        ),
        (
            "Cache Compression: A Mini-",
            "Review of Methods",
            "This is the body text of the paper.",
            "Cache Compression: A Mini-Review of Methods",
        ),
    ],
)
def test_parse_pdf_joins_title_lines_broken_at_a_hyphen(
    first: str, second: str, body: str, title: str
) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), first, fontsize=18)
    page.insert_text((72, 121), second, fontsize=18)
    page.insert_text((72, 160), body)
    assert _parse_pdf(doc.tobytes()).title == title


def test_parse_pdf_skips_a_small_caps_heading_whose_capitals_outsize_the_title() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 88), "External Technical Root Cause Analysis", fontsize=16)
    x = 72.0
    for text, size in [("I", 18), ("NTRODUCTION", 14.5)]:
        page.insert_text((x, 200), text, fontsize=size)
        x += fitz.get_text_length(text, fontsize=size)
    page.insert_text((72, 230), "This report elaborates on the information shared earlier.", fontsize=11)
    assert _parse_pdf(doc.tobytes()).title == "External Technical Root Cause Analysis"


def test_parse_pdf_reads_bold_titles_set_at_body_size() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 50), "Draft version October 29, 2024", fontsize=8)
    page.insert_text((72, 110), "JWST Validates HST Distance Measurements", fontsize=10, fontname="hebo")
    page.insert_text((72, 140), "Adam G. Riess, Dan Scolnic", fontsize=9)
    body = "The distance ladder calibrates type Ia supernovae with Cepheids. " * 6
    page.insert_textbox(fitz.Rect(72, 420, 540, 700), body, fontsize=10.1)
    result = _parse_pdf(doc.tobytes())
    assert result.title == "JWST Validates HST Distance Measurements"


def test_parse_pdf_never_takes_a_lower_bold_heading_for_the_title() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 110), "A Title Set in Regular Type", fontsize=10)
    page.insert_text((72, 400), "1. Introduction and Motivation", fontsize=10, fontname="hebo")
    body = "The distance ladder calibrates type Ia supernovae with Cepheids. " * 6
    page.insert_textbox(fitz.Rect(72, 420, 540, 700), body, fontsize=10.1)
    assert _parse_pdf(doc.tobytes()).title != "1. Introduction and Motivation"


@pytest.mark.parametrize(
    "placeholder",
    [
        "Title",
        "Title (Use “Title” style here)",
        "Acknowledgement Title (Trebuchet MS Bold 15 pt)",
        "July 7, 2020",
        "Microsoft PowerPoint - EASD25 CVOT Symposium_Lilly Deck_FINAL_18SEP2025",
        "Microsoft Word - field_trial_v2.docx",
    ],
)
def test_parse_pdf_ignores_template_placeholder_titles(placeholder: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 110), "Heat Pump Field Trial Results", fontsize=18)
    page.insert_text((72, 150), "This is the body text of the report.")
    doc.set_metadata({"title": placeholder})
    result = _parse_pdf(doc.tobytes())
    assert result.title == "Heat Pump Field Trial Results"


def test_parse_docx_extracts_title_and_text() -> None:
    document = Document()
    document.add_heading("DOCX Title", 0)
    document.add_paragraph("This is a paragraph.")
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    result = _parse_docx(buffer.read())
    assert result.title == "DOCX Title"
    assert "paragraph" in result.text


def _docx_bytes(build) -> bytes:
    document = Document()
    build(document)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_parse_docx_keeps_paragraphs_and_tables_in_order() -> None:
    def build(document: Document) -> None:
        document.add_paragraph("Intro paragraph here.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Alice"
        table.cell(1, 1).text = "100"
        document.add_paragraph("Closing paragraph here.")

    result = _parse_docx(_docx_bytes(build))

    assert result.text.splitlines() == [
        "Intro paragraph here.",
        "Name | Value",
        "Alice | 100",
        "Closing paragraph here.",
    ]


def test_parse_xlsx_extracts_rows() -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Value"])
    ws.append(["Alice", 100])
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    result = _parse_xlsx(buffer.read())
    assert "Alice | 100" in result.text


def test_parse_csv_extracts_rows() -> None:
    result = _parse_csv(b"Name,Value\nAlice,100\nBob,200")
    assert "Alice | 100" in result.text
    assert "Bob | 200" in result.text


def test_parse_plaintext_extracts_text() -> None:
    result = _parse_plaintext(b"Hello world")
    assert result.text == "Hello world"



@pytest.mark.parametrize(
    "head,expected",
    [
        (
            '<meta name="prism.publicationDate" content="2024-03-01">'
            '<meta name="sailthru.date" content="2024-03-02">',
            "2024-03-01",
        ),
        (
            '<meta name="dc.date.issued" content="2024-04-01">',
            "2024-04-01",
        ),
        (
            '<meta name="parsely-pub-date" content="2024-05-01">',
            "2024-05-01",
        ),
        (
            '<script type="application/ld+json">{"uploadDate":"2024-06-01"}</script>',
            "2024-06-01",
        ),
    ],
)
def test_extract_published_at_finds_known_sources(head, expected) -> None:
    html = f"<html><head>{head}</head><body></body></html>"
    assert _extract_published_at(html) == expected



def test_detect_fetch_error_detects_paywall() -> None:
    html = '<html><body><div class="paywall">Subscribe to read more.</div></body></html>'
    assert _detect_fetch_error(html, "") == "paywall"


def test_detect_fetch_error_detects_captcha() -> None:
    html = '<html><body><div class="g-recaptcha"></div></body></html>'
    assert _detect_fetch_error(html, "") == "captcha"


def test_detect_fetch_error_detects_empty_js_shell() -> None:
    html = '<html><body>' + ' ' * 9000 + '<div id="root"></div></body></html>'
    assert _detect_fetch_error(html, "") == "empty_js_shell"


@pytest.mark.parametrize(
    "html,text",
    [
        (
            "<html><body><p>This is a normal page with plenty of content.</p></body></html>",
            "normal content here",
        ),
        (
            "<html><body><article><p>"
            + " ".join(["word"] * 120)
            + "</p></article><script>g-recaptcha</script></body></html>",
            " ".join(["word"] * 120),
        ),
    ],
)
def test_detect_fetch_error_returns_none_for_benign_pages(html, text) -> None:
    assert _detect_fetch_error(html, text) is None


@pytest.mark.parametrize(
    "title",
    [
        "EIA - Sorry! Unexpected Error",
        "Indisponibilité du site Web d’Hydro-Québec à partir de l’étranger | Hydro-Québec",
        "404 Not Found",
        "Page not found | Example Utility",
    ],
)
def test_detect_fetch_error_flags_short_error_pages(title) -> None:
    text = "Our website is temporarily unavailable. Please try again later. " * 8
    html = f"<html><head><title>{title}</title></head><body><p>{text}</p></body></html>"

    assert _detect_fetch_error(html, text) == "error_page"


def test_detect_fetch_error_keeps_long_pages_whose_title_mentions_errors() -> None:
    text = " ".join(["measured"] * 400)
    html = (
        "<html><head><title>Why sizing errors leave heat pumps unavailable at night</title>"
        f"</head><body><p>{text}</p></body></html>"
    )

    assert _detect_fetch_error(html, text) is None


def test_extract_html_dates_arxiv_paper_from_version_stamp() -> None:
    html = _long_html("Streaming Models", words=200).replace(
        "<body>", "<body><div>arXiv:2309.17453v4 [cs.CL] 07 Apr 2024</div>", 1
    )
    assert "arXiv:2309" in html
    assert _extract_html(html).published_at == "2024-04-07"


def test_extract_html_caps_long_successful_extraction() -> None:
    html = _long_html("Long Article", words=30000)
    result = _extract_html(html)
    assert len(result.text) <= 120_000



def test_scrape_retries_on_429_and_eventually_succeeds() -> None:
    service = ScrapeService()
    call_count = 0

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 429 if call_count <= 2 else 200
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                def raise_for_status(self):
                    pass
                async def text(self, errors=None):
                    return _long_html(words=120)
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake))
    assert result.fetch_error is None
    assert call_count == 3


def test_scrape_retries_on_transient_client_error() -> None:
    service = ScrapeService()
    service._backoff_delay = lambda attempt: 0
    call_count = 0

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientConnectionError("connection reset")
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 200
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                async def text(self, errors=None):
                    return _long_html(words=120)
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake))
    assert result.fetch_error is None
    assert call_count == 2


def test_scrape_does_not_retry_404() -> None:
    service = ScrapeService()
    call_count = 0

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 404
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                def raise_for_status(self):
                    pass
                async def text(self, errors=None):
                    return "not found"
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake))
    assert result.fetch_error == "scrape_failed"
    assert call_count == 1



def test_scrape_service_detects_pdf_from_content_type() -> None:
    service = ScrapeService()

    class _FakeContent:
        async def read(self) -> bytes:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "PDF Title")
            return doc.tobytes()

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            class FakeResponse:
                url = "https://example.com/doc.pdf"
                headers = {"content-type": "application/pdf"}
                status = 200
                content = _FakeContent()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                def raise_for_status(self):
                    pass
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com/doc.pdf", session=fake))
    assert result.fetch_error is None
    assert result.content_type == "application/pdf"
    assert "PDF Title" in result.text


def test_scrape_service_detects_docx_from_url_extension() -> None:
    service = ScrapeService()

    class _FakeContent:
        async def read(self) -> bytes:
            document = Document()
            document.add_heading("DOCX Title", 0)
            document.add_paragraph("Hello from docx.")
            buffer = io.BytesIO()
            document.save(buffer)
            return buffer.getvalue()

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            class FakeResponse:
                url = "https://example.com/doc.docx"
                headers = {"content-type": "application/octet-stream"}
                status = 200
                content = _FakeContent()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                def raise_for_status(self):
                    pass
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com/doc.docx", session=fake))
    assert result.fetch_error is None
    assert "DOCX Title" in result.text


def test_scrape_service_rejects_oversized_html_before_decoding() -> None:
    service = ScrapeService()

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html", "content-length": str(_MAX_DOWNLOAD_BYTES + 1)}
                status = 200
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                async def text(self, errors=None):
                    raise AssertionError("oversized HTML should not be decoded")
            return FakeResponse()

    fake = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake))
    assert result.fetch_error == "non_text_content"



def test_extract_produces_structured_blocks() -> None:
    long_para = " ".join(["word"] * 60)
    html = (
        f"<html><body><article>"
        f"<h2>Heading</h2>"
        f"<p>{long_para}</p>"
        f"<p>{long_para}</p>"
        f"<ul><li>List item one with enough words.</li><li>List item two with enough words.</li></ul>"
        f"<blockquote>A quote that is long enough to pass the filter.</blockquote>"
        f"<pre><code>print(hello)</code></pre>"
        f"</article></body></html>"
    )
    result = _extract_with_bs4(html)
    types = [b.type for b in result.blocks]
    assert "heading" in types
    assert "paragraph" in types
    assert "list_item" in types
    assert "blockquote" in types
    assert "code" in types


def test_scraped_page_defaults_blocks_to_empty_list() -> None:
    from shandu.services.scrape import ScrapedPage
    page = ScrapedPage(
        requested_url="https://example.com",
        url="https://example.com",
        title="Test",
        text="text",
        domain="example.com",
    )
    assert page.blocks == []


def test_domain_scheduler_staggers_concurrent_slots() -> None:
    scheduler = _DomainScheduler(max_concurrent_per_domain=2, base_delay=0.2)

    async def timed_acquire() -> float:
        await scheduler.acquire("d.example")
        return time.monotonic()

    async def run_all() -> list[float]:
        stamps = await asyncio.gather(timed_acquire(), timed_acquire())
        await scheduler.release("d.example")
        await scheduler.release("d.example")
        return sorted(stamps)

    first, second = asyncio.run(run_all())

    assert second - first >= 0.15


def test_fetch_acquires_scheduler_slot_for_hop_domain() -> None:
    service = ScrapeService()
    acquired: list[str] = []

    class RecordingScheduler:
        async def acquire(self, domain: str) -> None:
            acquired.append(domain)

        async def release(self, domain: str) -> None:
            del domain

        def bump_backoff(self, domain: str) -> None:
            del domain

        def reset_backoff(self, domain: str) -> None:
            del domain

    service._domain_scheduler = RecordingScheduler()

    class HopSession:
        def get(self, *args, **kwargs):
            del args, kwargs

            class HopResponse:
                status = 404
                headers: dict[str, str] = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    del args

            return HopResponse()

    asyncio.run(
        service._fetch_single_request(
            "https://orig.example/y", "https://hop.example/x", HopSession(), 0, 1
        )
    )

    assert acquired == ["hop.example"]


def test_extract_links_leads_with_other_hosts_and_skips_site_chrome() -> None:
    from shandu.services.scrape.extraction import _extract_links

    html = """<html><body>
    <nav><a href="https://other.org/menu">Menu item</a></nav>
    <main><p>See <a href="/docs/spec">the spec page</a> and
    <a href="https://arxiv.org/abs/1">the paper</a>, <a href="#top">back to top</a>,
    <a href="mailto:a@b.org">mail us</a>, <a href="https://arxiv.org/abs/1#s2">same paper</a>,
    <a href="https://x.org/y">ok</a>, <a href="https://example.com:443/about">about the site</a></p></main>
    <footer><a href="https://other.org/legal">Legal notice</a></footer>
    </body></html>"""

    links = _extract_links(html, "https://example.com/post")

    assert [(link.url, link.text) for link in links] == [
        ("https://arxiv.org/abs/1", "the paper"),
        ("https://example.com/docs/spec", "the spec page"),
        ("https://example.com:443/about", "about the site"),
    ]
