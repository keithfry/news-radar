"""Tests for keyboard navigation between cards on digest pages."""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from newsradar.html_generator import generate_html

_DATE = datetime(2026, 11, 23, 8, 0, tzinfo=ZoneInfo("America/New_York"))


def _digest(**kwargs) -> str:
    return generate_html(
        newsletters=[],
        articles=[{"title": "New model release", "link": "https://example.com/a1",
                   "source": "Example Blog", "summary": "A model came out.", "tags": ["model"]}],
        papers=[],
        errors=[],
        date=_DATE,
        **kwargs,
    )


def _header(html: str) -> str:
    return re.search(r'<div class="page-header">(.*?)\n</div>\n', html, re.DOTALL).group(1)


def test_help_line_is_last_in_header():
    html = _digest(mp3_url="https://example.com/a.mp3", podcast_rss_url="https://example.com/p.rss")
    lines = _header(html).splitlines()
    assert 'class="kb-help"' in lines[-1]
    assert 'class="podcast-player"' in "\n".join(lines[:-1])
    for key in ("↓", "n", "↑", "b", "Enter", "Esc"):
        assert f"<kbd>{key}</kbd>" in lines[-1]


def test_help_line_follows_stats_without_podcast():
    lines = _header(_digest()).splitlines()
    assert 'class="kb-help"' in lines[-1]
    assert lines[-2] == "  </div>"


def test_script_binds_keys_and_selects_cards():
    html = _digest()
    script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    for key in ("'ArrowDown'", "'ArrowUp'", "'n'", "'b'", "'Escape'"):
        assert key in script, key
    assert "kb-selected" in script
    assert "scrollIntoView" in script


def test_inline_css_styles_selection_and_hides_help_on_touch():
    html = _digest()
    assert ".card.kb-selected {" in html
    assert "@media (hover: none) { .kb-help { display: none; } }" in html


def test_touch_buttons_put_next_on_the_right():
    html = _digest()
    buttons = re.search(r'<div class="nav-buttons">(.*?)</div>', html, re.DOTALL).group(1)
    prev_at, next_at = buttons.index('data-step="-1"'), buttons.index('data-step="1"')
    assert prev_at < next_at
    assert ".nav-next { right: 16px; }" in html
    assert ".nav-prev { left: 16px; }" in html
    assert html.index('class="nav-buttons"') < html.rindex("<script>")


def test_touch_buttons_show_only_without_hover():
    html = _digest()
    assert ".nav-buttons { display: none; }" in html
    assert "@media (hover: none) { .nav-buttons { display: block; }" in html
    assert "querySelectorAll('.nav-btn')" in html
