"""Fetch and extract readable text from an article URL."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Tags to try in order for main content extraction
_CONTENT_TAGS = ["article", "main", '[role="main"]', ".post-content",
                 ".article-body", ".entry-content", "#content", "body"]


def source_name_from_url(url: str) -> str:
    """Extract a readable source name from a URL (e.g. 'wired.com')."""
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return url


def fetch_article_text(url: str, timeout: int = 10) -> str | None:
    """Fetch a URL and return the main article text, or None on failure."""
    text, _title = _fetch_article(url, timeout=timeout)
    return text


def fetch_article_text_and_title(url: str, timeout: int = 10) -> tuple[str | None, str]:
    """Fetch a URL and return (main article text, page title). Title is '' if unavailable."""
    return _fetch_article(url, timeout=timeout)


def _fetch_article(url: str, timeout: int = 10) -> tuple[str | None, str]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"[article_fetcher] fetch failed {url}: {e}", file=sys.stderr)
        return None, ""

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type:
        return None, ""

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Remove noise
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Try to find the main content block
    text = ""
    for selector in _CONTENT_TAGS:
        el = soup.select_one(selector)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                break

    if not text:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = "\n".join(lines)

    # Treat short pages as failures — likely soft 404s, login walls, or empty pages
    if len(result) < 400:
        print(f"[article_fetcher] skipping {url}: content too short ({len(result)} chars)", file=sys.stderr)
        return None, ""

    # Check the first 20% of the text for soft-404 signals
    preview = result[:max(1, len(result) // 5)].lower()
    if "404" in preview or "page not found" in preview:
        print(f"[article_fetcher] skipping {url}: soft 404 detected in page header", file=sys.stderr)
        return None, ""

    return result, title


def enrich_with_full_text(
    articles: list[dict],
    url_workers: int,
    char_cap: int,
    log=print,
) -> None:
    """Fetch full article HTML for each entry in place, setting `body`.

    Falls back to the RSS feed's own `summary` blurb if the page can't be
    retrieved. Used by both the production pipeline and any model-comparison
    tooling so both read the same full-page content.
    """
    if not articles:
        return

    log(f"  Fetching full article text for {len(articles)} articles ({url_workers} workers)...")

    def _fetch(article: dict) -> None:
        url = article.get("link", "")
        if not url:
            return
        text = fetch_article_text(url)
        if text:
            article["body"] = text[:char_cap]
        else:
            article["body"] = article.get("summary", "")[:char_cap]

    with ThreadPoolExecutor(max_workers=url_workers) as executor:
        futures = [executor.submit(_fetch, a) for a in articles]
        for future in as_completed(futures):
            future.result()
