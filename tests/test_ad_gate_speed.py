"""Tests for the ad-gate prompt choice and for skipping redundant ad-gate calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from newsradar import cli, llm
from newsradar.config import load_config

from conftest import EXAMPLE_CONFIG

_config = load_config(EXAMPLE_CONFIG)
_TOPIC = _config.topic("ai")


@pytest.fixture(autouse=True)
def _fresh_caches():
    llm._has_system_prompt.cache_clear()
    cli._ad_verdicts.clear()
    yield
    llm._has_system_prompt.cache_clear()
    cli._ad_verdicts.clear()


def _capture_prompt(monkeypatch) -> list[str]:
    prompts: list[str] = []

    def fake_chat(prompt, model, **kwargs):
        prompts.append(prompt)
        return '{"is_ad": false, "reason": "news"}'

    monkeypatch.setattr(llm, "_chat", fake_chat)
    return prompts


def test_model_with_system_prompt_gets_only_the_item(monkeypatch):
    monkeypatch.setattr(llm, "_has_system_prompt", lambda model: True)
    prompts = _capture_prompt(monkeypatch)
    llm.classify_ad("A title", "Some text", "ollama/ad-detector")
    assert prompts == ["Title: A title\nSummary: Some text"]


def test_base_model_gets_full_instructions(monkeypatch):
    monkeypatch.setattr(llm, "_has_system_prompt", lambda model: False)
    prompts = _capture_prompt(monkeypatch)
    llm.classify_ad("A title", "Some text", "ollama/qwen3.5:9b")
    assert "REFERRAL PROGRAM" in prompts[0]
    assert "Title: A title" in prompts[0]


def test_has_system_prompt_reads_modelfile(monkeypatch):
    modelfiles = {
        "ad-detector": 'FROM qwen3.5:9b\nPARAMETER temperature 0.0\nSYSTEM """\nYou are AdDetector.\n"""\n',
        "qwen3.5:9b": "FROM /models/blobs/sha256-abc\nTEMPLATE {{ .Prompt }}\n",
    }
    monkeypatch.setattr(llm.ollama, "show", lambda name: SimpleNamespace(modelfile=modelfiles[name]))
    assert llm._has_system_prompt("ollama/ad-detector") is True
    assert llm._has_system_prompt("ollama/qwen3.5:9b") is False


def test_has_system_prompt_is_false_when_show_fails(monkeypatch):
    def boom(name):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(llm.ollama, "show", boom)
    assert llm._has_system_prompt("ollama/ad-detector") is False


def test_has_system_prompt_is_false_for_non_ollama_models():
    assert llm._has_system_prompt("anthropic/claude-haiku-4-5") is False


def _count_ad_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_classify_ad(title, text, model):
        calls.append(title)
        return False, ""

    monkeypatch.setattr(cli, "classify_ad", fake_classify_ad)
    monkeypatch.setattr(cli, "classify_topic", lambda *a: False)
    return calls


def test_arxiv_items_skip_the_ad_gate(monkeypatch):
    calls = _count_ad_calls(monkeypatch)
    item = {"title": "A new RL paper", "summary": "Abstract", "source": "arXiv - cs.AI"}
    cli._process_one(1, 1, item, "rss", _TOPIC, _config, [])
    assert calls == []


def test_ad_verdict_is_reused_across_topics(monkeypatch):
    calls = _count_ad_calls(monkeypatch)
    item = {"title": "Newsletter issue", "body": "Body text", "source": "The Hustle"}
    cli._process_one(1, 1, item, "email", _TOPIC, _config, [])
    cli._process_one(1, 1, item, "email", _TOPIC, _config, [])
    assert calls == ["Newsletter issue"]
