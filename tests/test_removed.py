"""Tests for tracking items the pipeline drops, and showing them on the page."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from newsradar import cli, llm
from newsradar.config import load_config
from newsradar.html_generator import generate_html

from conftest import EXAMPLE_CONFIG

_config = load_config(EXAMPLE_CONFIG)
_TOPIC = _config.topic("ai")


@pytest.fixture(autouse=True)
def _fresh_ad_cache():
    cli._ad_verdicts.clear()
    yield
    cli._ad_verdicts.clear()


def _no_models(monkeypatch):
    monkeypatch.setattr(cli, "classify_ad", lambda *a: (False, ""))
    monkeypatch.setattr(cli, "classify_topic", lambda *a: False)


def test_heuristic_ad_is_recorded(monkeypatch):
    _no_models(monkeypatch)
    removed: list[dict] = []
    item = {"title": "AI gadgets: shop now", "link": "https://x/1", "source": "X"}
    assert cli._process_one(1, 1, item, "rss", _TOPIC, _config, removed) is None
    assert removed == [{"title": "AI gadgets: shop now", "link": "https://x/1",
                        "source": "X", "reason": "advertisement"}]


def test_ad_gate_is_recorded(monkeypatch):
    _no_models(monkeypatch)
    monkeypatch.setattr(cli, "classify_ad", lambda *a: (True, "sponsored"))
    removed: list[dict] = []
    cli._process_one(1, 1, {"title": "AI news"}, "rss", _TOPIC, _config, removed)
    assert [r["reason"] for r in removed] == ["ad detector"]


def test_off_topic_is_recorded(monkeypatch):
    _no_models(monkeypatch)
    removed: list[dict] = []
    cli._process_one(1, 1, {"title": "Armrests on flights"}, "rss", _TOPIC, _config, removed)
    assert [r["reason"] for r in removed] == ["off-topic"]


def test_summarize_removed_groups_by_reason():
    removed = [{"reason": "duplicate"}, {"reason": "off-topic"}, {"reason": "duplicate"}]
    assert llm.removed_counts(removed) == [("duplicate", 2), ("off-topic", 1)]


def _html(removed):
    return generate_html(
        newsletters=[], articles=[], papers=[], errors=[],
        date=datetime(2026, 10, 16, 8, tzinfo=ZoneInfo("America/New_York")),
        removed=removed,
    )


def test_page_shows_removed_chip_and_section():
    html = _html([
        {"title": "Dup <story>", "link": "https://x/1", "source": "X", "reason": "duplicate"},
        {"title": "Promo", "link": "", "source": "Y", "reason": "advertisement"},
    ])
    assert "2 Removed" in html
    assert 'id="removed-items"' in html
    assert "Dup &lt;story&gt;" in html
    assert "Duplicate (1)" in html


def test_page_without_removed_shows_zero_chip_and_no_section():
    html = _html([])
    assert "0 Removed" in html
    assert 'id="removed-items"' not in html
