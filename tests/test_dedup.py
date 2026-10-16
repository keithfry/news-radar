"""Tests for dedup batching and the dedup_batch_size config field."""

from __future__ import annotations

import pytest

from newsradar import llm
from newsradar.config import load_config


def _items(n: int) -> list[dict]:
    return [{"title": f"Story {i}"} for i in range(n)]


@pytest.fixture
def batch_sizes(monkeypatch):
    """Replace the LLM call with a recorder; drops nothing."""
    sizes: list[int] = []

    def fake_batch(batch, model):
        sizes.append(len(batch))
        return set()

    monkeypatch.setattr(llm, "_dedup_batch", fake_batch)
    return sizes


@pytest.mark.parametrize("batch_size", [0, None])
def test_zero_or_missing_batch_size_sends_all_at_once(batch_sizes, batch_size):
    llm.deduplicate(_items(95), "m", batch_size=batch_size)
    assert batch_sizes == [95]


def test_default_batch_size_sends_all_at_once(batch_sizes):
    llm.deduplicate(_items(95), "m")
    assert batch_sizes == [95]


def test_positive_batch_size_splits_items(batch_sizes):
    llm.deduplicate(_items(95), "m", batch_size=40)
    assert batch_sizes == [40, 40, 15]


def test_drops_returned_indices(monkeypatch):
    monkeypatch.setattr(llm, "_dedup_batch", lambda batch, model: {1})
    out = llm.deduplicate(_items(3), "m", batch_size=0)
    assert [i["title"] for i in out] == ["Story 0", "Story 2"]


def _write_config(tmp_path, models_toml: str, topic_models_toml: str = "") -> str:
    (tmp_path / "t.toml").write_text(
        '[topic]\nname = "t"\nclassifier_prompt = "p"\n' + topic_models_toml
    )
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[models]\n{models_toml}\n[topics]\nfiles = ["t.toml"]\n')
    return str(cfg)


def test_config_batch_size_defaults_to_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("DEDUP_BATCH_SIZE", raising=False)
    config = load_config(_write_config(tmp_path, ""), env_file=tmp_path / "none.env")
    assert config.models.dedup_batch_size == 0


def test_config_batch_size_from_toml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEDUP_BATCH_SIZE", raising=False)
    config = load_config(
        _write_config(tmp_path, "dedup_batch_size = 40"), env_file=tmp_path / "none.env"
    )
    assert config.models.dedup_batch_size == 40


def test_config_batch_size_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DEDUP_BATCH_SIZE", "25")
    config = load_config(
        _write_config(tmp_path, "dedup_batch_size = 40"), env_file=tmp_path / "none.env"
    )
    assert config.models.dedup_batch_size == 25


def test_topic_can_override_batch_size(tmp_path, monkeypatch):
    monkeypatch.delenv("DEDUP_BATCH_SIZE", raising=False)
    config = load_config(
        _write_config(tmp_path, "dedup_batch_size = 40", "[models]\ndedup_batch_size = 10\n"),
        env_file=tmp_path / "none.env",
    )
    assert config.topic("t").models.dedup_batch_size == 10
