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


def _rest_of_pipeline_passes(monkeypatch):
    monkeypatch.setattr(cli, "classify_topic", lambda *a: True)
    monkeypatch.setattr(cli, "summarize", lambda *a: "A live webinar on agent evaluation, register at the site.")
    monkeypatch.setattr(cli, "tag", lambda *a: [])
    monkeypatch.setattr(cli, "_apply_exclusions", lambda title, summary, tags, topic, config: (tags, None))


@pytest.mark.parametrize("title", [
    "Freshworks alums raise $53M to take on Salesforce",  # "on sale" used to match inside "Salesforce"
    "AI gadgets: shop now",
])
def test_every_title_reaches_the_ad_model(monkeypatch, title):
    _rest_of_pipeline_passes(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(cli, "classify_ad", lambda t, text, model: seen.append(t) or (False, ""))
    removed: list[dict] = []
    out = cli._process_one(1, 1, {"title": title, "source": "X"}, "rss", _TOPIC, _config, removed)
    assert seen == [title] and out is not None and removed == []


def test_webinar_summary_is_not_removed(monkeypatch):
    _no_models(monkeypatch)
    _rest_of_pipeline_passes(monkeypatch)
    removed: list[dict] = []
    out = cli._process_one(1, 1, {"title": "Agent evals webinar recap"}, "rss", _TOPIC, _config, removed)
    assert out is not None and removed == []


def _reasons(item: dict) -> list[str]:
    removed: list[dict] = []
    cli._process_one(1, 1, item, "rss", _TOPIC, _config, removed)
    return [r["reason"] for r in removed]


def test_ad_phrase_in_page_chrome_is_not_an_ad(monkeypatch):
    # TechCrunch's sidebar promo ends up in every fetched article body.
    _no_models(monkeypatch)
    item = {"title": "Meta launches an open-weight model that runs on Macs",
            "body": "Meta released the model today. Last day to save Oct. 16 Register now Most Popular"}
    assert _reasons(item) == ["off-topic"]


def test_dollar_amounts_in_body_are_not_an_ad(monkeypatch):
    _no_models(monkeypatch)
    item = {"title": "Acme tells investors quarterly revenue is up nearly 50%",
            "body": "Revenue reached $16.8bn, up from $11bn, with $4bn in new bookings."}
    assert _reasons(item) == ["off-topic"]


def test_arxiv_items_skip_the_ad_gate(monkeypatch):
    _no_models(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(cli, "classify_ad", lambda t, text, model: seen.append(t) or (True, "x"))
    item = {"title": "Planning under a limited time budget", "source": "arXiv - cs.LG",
            "summary": "We bound regret by $2$, $5$ and $20$ rounds."}
    assert _reasons(item) == ["off-topic"] and seen == []


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
        {"title": "Promo", "link": "", "source": "Y", "reason": "ad detector"},
    ])
    assert "2 Removed" in html
    assert 'id="removed-items"' in html
    assert "Dup &lt;story&gt;" in html
    assert "Duplicate (1)" in html


def test_page_without_removed_shows_zero_chip_and_no_section():
    html = _html([])
    assert "0 Removed" in html
    assert 'id="removed-items"' not in html
