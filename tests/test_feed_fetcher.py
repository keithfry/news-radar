"""Tests for feed download and error reporting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import requests

from newsradar import feed_fetcher

CUTOFF = datetime(2026, 10, 1, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>Fresh</title><link>https://x/1</link><pubDate>Thu, 15 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>Old</title><link>https://x/2</link><pubDate>Thu, 01 Jan 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


class FakeResponse:
    def __init__(self, status: int, content: bytes, content_type: str = "application/rss+xml"):
        self.status_code = status
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture
def serve(monkeypatch):
    def _serve(response=None, exc=None):
        def fake_get(url, **kwargs):
            if exc:
                raise exc
            return response
        monkeypatch.setattr(feed_fetcher.requests, "get", fake_get)
    return _serve


def test_parses_items_inside_window(serve):
    serve(FakeResponse(200, RSS))
    result = feed_fetcher.fetch_feed("S", "https://x/feed", CUTOFF)
    assert "error" not in result
    assert [i["title"] for i in result["items"]] == ["Fresh"]


def test_http_error_is_reported(serve):
    serve(FakeResponse(403, b"<html>forbidden</html>", "text/html"))
    result = feed_fetcher.fetch_feed("S", "https://x/feed", CUTOFF)
    assert "403" in result["error"]


def test_connection_error_is_reported(serve):
    serve(exc=requests.exceptions.SSLError("certificate verify failed"))
    result = feed_fetcher.fetch_feed("S", "https://x/feed", CUTOFF)
    assert "certificate verify failed" in result["error"]


def test_html_page_is_reported(serve):
    serve(FakeResponse(200, b"<!DOCTYPE html><html><body><p>blog</body></html>", "text/html"))
    result = feed_fetcher.fetch_feed("S", "https://x/feed", CUTOFF)
    assert result["error"].startswith("not a feed")


def test_empty_valid_feed_is_not_an_error(serve):
    serve(FakeResponse(200, b'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title></channel></rss>'))
    result = feed_fetcher.fetch_feed("S", "https://x/feed", CUTOFF)
    assert result["items"] == []
