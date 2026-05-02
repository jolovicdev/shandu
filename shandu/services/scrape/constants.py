from __future__ import annotations

import re

_USER_AGENTS = [
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
    ),
]

_HEADERS = {
    "User-Agent": _USER_AGENTS[0],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

_RETRYABLE_STATUSES: set[int] = {408, 425, 429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUSES: set[int] = {401, 403, 404, 410}
_MAX_ATTEMPTS = 3
_MIN_ARTICLE_WORDS = 80
_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_EXTRACTED_CHARS = 18000
_MAX_PDF_PAGES = 200
_MAX_XLSX_ROWS = 50000

_PAYWALL_PATTERNS = [
    re.compile(r'class=["\']?[^"\'>]*paywall', re.IGNORECASE),
    re.compile(r'id=["\']?[^"\'>]*paywall', re.IGNORECASE),
    re.compile(r'data-testid=["\']?paywall', re.IGNORECASE),
    re.compile(r'please\s+(subscribe|log\s+in|sign\s+in)\s+to\s+(read|access|view|continue)', re.IGNORECASE),
    re.compile(r'subscription\s+required\s+to\s+(read|access|view)', re.IGNORECASE),
    re.compile(r'premium\s+content', re.IGNORECASE),
    re.compile(r'class=["\']?[^"\'>]*membership', re.IGNORECASE),
]

_CAPTCHA_PATTERNS = [
    re.compile(r'g-recaptcha', re.IGNORECASE),
    re.compile(r'cf-turnstile', re.IGNORECASE),
    re.compile(r'hCaptcha', re.IGNORECASE),
    re.compile(r'class=["\'][^"\']*captcha', re.IGNORECASE),
    re.compile(r'id=["\'][^"\']*captcha', re.IGNORECASE),
]
