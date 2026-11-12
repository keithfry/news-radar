"""Tests for the opt-in shared stylesheet link and site bar on digest pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from newsradar.config import SiteConfig
from newsradar.html_generator import digest_page_chrome, generate_html, site_bar_html

GOLDEN = Path(__file__).parent / "test_data" / "digest_default.html"
_DATE = datetime(2026, 11, 12, 8, 0, tzinfo=ZoneInfo("America/New_York"))


def sample_digest_html(**kwargs) -> str:
    """A digest with every section filled, built from fixed inputs."""
    return generate_html(
        newsletters=[{
            "title": "Weekly agents roundup", "link": "https://example.com/n1",
            "source": "The Batch", "summary": "Agents shipped.", "tags": ["agents"],
        }],
        articles=[
            {"title": "New model release", "link": "https://example.com/a1",
             "source": "Example Blog", "summary": "A model came out.", "tags": ["model"]},
            {"title": "Humanoid demo", "link": "https://example.com/a2",
             "source": "Robot News", "summary": "A robot walked.", "tags": ["robotics"]},
        ],
        papers=[{
            "title": "Scaling laws, again", "link": "https://arxiv.org/abs/0000.00000",
            "source": "arXiv", "summary": "Bigger is better.", "tags": ["research"],
        }],
        errors=[{"source": "Dead Feed", "feed_url": "https://example.com/feed", "error": "HTTP 404"}],
        removed=[{"title": "Buy now", "link": "https://example.com/ad", "source": "Spam", "reason": "ad"}],
        date=_DATE,
        topic_label="AI",
        mp3_url="https://example.com/ai-radar-2026-11-12.mp3",
        podcast_rss_url="https://example.com/podcast.rss",
        og_image_url="https://example.com/ai-radar-2026-11-12.og.jpg",
        **kwargs,
    )


def test_default_output_is_unchanged():  # SC-005
    assert sample_digest_html() == GOLDEN.read_text(encoding="utf-8")


_EXPECTED_BAR = (
    '<nav class="site-bar" aria-label="Site">\n'
    '  <a class="site-bar-home" href="https://keithfry.net/">Keith Fry</a>\n'
    '  <ol class="site-bar-crumbs">\n'
    '    <li><a href="../../">Tech Radar</a></li>\n'
    '    <li><a href="../">AI</a></li>\n'
    '    <li aria-current="page">November 12, 2026</li>\n'
    '  </ol>\n'
    '</nav>'
)
_CRUMBS = [("Tech Radar", "../../"), ("AI", "../"), ("November 12, 2026", None)]
_SITE = SiteConfig(
    stylesheet="radar.css", home_url="https://keithfry.net/",
    home_label="Keith Fry", section_label="Tech Radar",
)


def test_stylesheet_href_replaces_inline_style():
    html = sample_digest_html(stylesheet_href="../../radar.css")
    assert '<link rel="stylesheet" href="../../radar.css">' in html
    assert "<style>" not in html


def test_site_bar_follows_body():
    bar = site_bar_html("https://keithfry.net/", "Keith Fry", _CRUMBS)
    html = sample_digest_html(site_bar=bar)
    assert html.index("<body>") < html.index('<nav class="site-bar"') < html.index('<div class="page-header">')


def test_site_bar_markup():
    assert site_bar_html("https://keithfry.net/", "Keith Fry", _CRUMBS) == _EXPECTED_BAR


def test_site_bar_empty_without_home_url():
    assert site_bar_html("", "Keith Fry", [("Tech Radar", "../")]) == ""


def test_site_bar_escapes_labels():
    bar = site_bar_html("https://keithfry.net/", "Keith Fry", [("R&D <Lab>", None)])
    assert "R&amp;D &lt;Lab&gt;" in bar


def test_digest_page_chrome_for_top_level_topic():
    href, bar = digest_page_chrome(_SITE, "AI", "AI", datetime(2026, 11, 12))
    assert href == "../../radar.css"
    assert bar == _EXPECTED_BAR


def test_digest_page_chrome_for_nested_topic():
    href, bar = digest_page_chrome(_SITE, "news/ai", "AI", datetime(2026, 11, 12))
    assert href == "../../../radar.css"
    assert '<li><a href="../../../">Tech Radar</a></li>' in bar


def test_digest_page_chrome_off_by_default():
    assert digest_page_chrome(SiteConfig(), "AI", "AI", datetime(2026, 11, 12)) == (None, "")


def test_section_label_falls_back_to_site_title():
    site = SiteConfig(title="My Radar", home_url="https://example.com/", home_label="Home")
    _, bar = digest_page_chrome(site, "AI", "AI", datetime(2026, 11, 12))
    assert '<li><a href="../../">My Radar</a></li>' in bar
