from __future__ import annotations

from shandu.services.scrape import ScrapeService


def test_scrape_service_canonicalizes_urls() -> None:
    service = ScrapeService()
    canonical = service._canonicalize_url("https://example.com/path?a=1#section")
    assert canonical == "https://example.com/path?a=1"


def test_scrape_service_extracts_main_content_and_drops_noise() -> None:
    service = ScrapeService()
    title, text = service._extract(
        """
        <html>
          <head>
            <title>Sample Article</title>
            <script>const token = "ignore me"</script>
          </head>
          <body>
            <header>header nav</header>
            <article>
              <p>This is a long paragraph that should be included in the extracted output because it is informative and content-rich.</p>
              <p>This is another long paragraph with additional context about the same topic and enough length to pass filtering.</p>
            </article>
          </body>
        </html>
        """
    )
    assert title == "Sample Article"
    assert "informative and content-rich" in text
    assert "ignore me" not in text
    assert "header nav" not in text
