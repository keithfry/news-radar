"""Smoke tests for podcast LLM functions — requires a local Ollama daemon.

Skipped automatically (module-wide) when the configured model isn't pulled
locally, rather than being unconditionally marked skip — these are real,
meaningful tests when Ollama is available.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from newsradar.config import load_config
from newsradar.llm import generate_audio_script, generate_intro_script, rank_items

from conftest import EXAMPLE_CONFIG, ollama_has_model

ET = ZoneInfo("America/New_York")

MODEL = "llama3.2:3b"

pytestmark = pytest.mark.skipif(
    not ollama_has_model(MODEL), reason=f"Ollama model {MODEL!r} not available locally"
)

_TOPIC = load_config(EXAMPLE_CONFIG).topic("ai")

SAMPLE_ITEMS = [
    {"title": "Google releases Gemini 2.5 Flash", "source": "Google DeepMind", "summary": "Google announced Gemini 2.5 Flash with improved speed.", "tags": ["model"], "_source_type": "rss"},
    {"title": "OpenAI raises $40B at $340B valuation", "source": "TechCrunch", "summary": "OpenAI closed a record funding round.", "tags": ["policy"], "_source_type": "rss"},
    {"title": "Anthropic releases Claude 4 Opus", "source": "Anthropic", "summary": "Claude 4 Opus is Anthropic's most capable model.", "tags": ["model"], "_source_type": "rss"},
]


def test_rank_items_returns_all_with_rank():
    ranked = rank_items(SAMPLE_ITEMS, MODEL)
    assert len(ranked) == len(SAMPLE_ITEMS)
    assert "rank" in ranked[0]
    ranks = [item["rank"] for item in ranked]
    assert sorted(ranks) == list(range(1, len(SAMPLE_ITEMS) + 1))


def test_generate_audio_script_returns_string():
    script = generate_audio_script(SAMPLE_ITEMS[0], MODEL)
    assert isinstance(script, str)
    assert len(script) > 50
    word_count = len(script.split())
    assert word_count <= 200, f"Script too long: {word_count} words"


def test_generate_intro_script_mentions_date():
    date = datetime(2026, 5, 21, 8, 0, tzinfo=ET)
    intro = generate_intro_script(SAMPLE_ITEMS, date, MODEL, _TOPIC)
    assert isinstance(intro, str)
    assert len(intro) > 20
    assert "May" in intro or "2026" in intro, f"Intro doesn't mention date: {intro!r}"
