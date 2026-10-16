"""Tests for dedup batching and the dedup_batch_size config field."""

from __future__ import annotations

import json

import httpx
import ollama
import pytest

from newsradar import llm
from newsradar.config import load_config


def _items(n: int) -> list[dict]:
    return [{"title": f"Story {i}"} for i in range(n)]


@pytest.fixture
def batch_sizes(monkeypatch):
    """Replace the LLM call with a recorder; drops nothing. Validation
    retries are off so recorded sizes show batching only."""
    sizes: list[int] = []
    monkeypatch.setattr(llm, "_DEDUP_RETRY_MIN_BATCH", 10**9)

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


# --- Single-batch parsing (groups format) ---------------------------------


def _batch(n: int) -> list[tuple[int, dict]]:
    return list(enumerate(_items(n)))


@pytest.fixture
def chat(monkeypatch):
    """Script _chat_meta responses; records (batch ids, num_ctx) per call."""
    calls: list[dict] = []
    responses: list = []

    def fake(prompt, model, json_mode=False, think=False, num_ctx=None, _retries=3):
        ids = [p["id"] for p in json.loads(prompt.split("Articles:\n", 1)[1].split("\n\nResponse", 1)[0])]
        calls.append({"ids": ids, "num_ctx": num_ctx})
        r = responses.pop(0) if responses else llm.ChatResult('{"groups": []}')
        if isinstance(r, Exception):
            raise r
        return r(ids) if callable(r) else r

    monkeypatch.setattr(llm, "_chat_meta", fake)
    return calls, responses


def _groups(*groups, confidence=0.9) -> llm.ChatResult:
    """Each group is (keep, *duplicates)."""
    return llm.ChatResult(json.dumps({"groups": [
        {"ids": [str(i) for i in g], "keep": str(g[0]), "confidence": confidence} for g in groups
    ]}))


def test_batch_drops_all_but_keep(chat):
    _, responses = chat
    responses.append(_groups((0, 1, 4), (2, 3)))
    assert llm._dedup_batch(_batch(5), "m") == {1, 3, 4}


def test_batch_omitted_ids_are_kept(chat):
    _, responses = chat
    responses.append(llm.ChatResult('{"groups": []}'))
    assert llm._dedup_batch(_batch(5), "m") == set()


def test_batch_ignores_low_confidence(chat):
    _, responses = chat
    responses.append(_groups((0, 1), confidence=0.2))
    assert llm._dedup_batch(_batch(5), "m") == set()


def test_batch_ignores_unknown_ids(chat):
    _, responses = chat
    responses.append(_groups((0, 99, 1), (98, 2), (3,)))
    assert llm._dedup_batch(_batch(5), "m") == {1}


def test_batch_keep_outside_group_is_ignored(chat):
    _, responses = chat
    responses.append(llm.ChatResult('{"groups": [{"ids": ["1", "2"], "keep": "0"}]}'))
    assert llm._dedup_batch(_batch(5), "m") == set()


def test_batch_overlapping_groups_never_drop_a_keep(chat):
    _, responses = chat
    responses.append(_groups((0, 1), (1, 0)))
    assert llm._dedup_batch(_batch(5), "m") == set()


@pytest.mark.parametrize("result", [
    llm.ChatResult("not json"),
    llm.ChatResult('["drop"]'),
    llm.ChatResult('{"drop": []}'),
    llm.ChatResult('{"groups": []}', done_reason="length"),
])
def test_batch_bad_response_raises(chat, result):
    _, responses = chat
    responses.append(result)
    with pytest.raises(llm._DedupFailed):
        llm._dedup_batch(_batch(5), "m")


def test_batch_prompt_near_num_ctx_raises(chat):
    calls, responses = chat
    responses.append(lambda ids: llm.ChatResult('{"groups": []}', prompt_eval_count=calls[-1]["num_ctx"]))
    with pytest.raises(llm._DedupFailed, match="truncated"):
        llm._dedup_batch(_batch(5), "m")


def test_batch_server_error_raises(chat):
    _, responses = chat
    responses.append(ollama.ResponseError("context too long", 400))
    with pytest.raises(llm._DedupFailed):
        llm._dedup_batch(_batch(5), "m")


def test_batch_connection_error_propagates(chat):
    _, responses = chat
    responses.append(httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        llm._dedup_batch(_batch(5), "m")


def test_batch_too_many_drops_raises(chat):
    _, responses = chat
    responses.append(_groups(tuple(range(8))))
    with pytest.raises(llm._DedupFailed, match="dropped"):
        llm._dedup_batch(_batch(12), "m")


def test_batch_small_batch_may_drop_most(chat):
    _, responses = chat
    responses.append(_groups((0, 1, 2)))
    assert llm._dedup_batch(_batch(3), "m") == {1, 2}


def test_batch_num_ctx_scales_and_clamps(chat):
    calls, _ = chat
    llm._dedup_batch(_batch(5), "m")
    llm._dedup_batch(_batch(400), "m")
    llm._dedup_batch(_batch(5000), "m")
    small, mid, huge = (c["num_ctx"] for c in calls)
    assert small == llm._DEDUP_MIN_NUM_CTX
    assert small < mid < huge == llm._DEDUP_MAX_NUM_CTX


# --- Split-in-half fallback ---------------------------------------------------


def test_failure_splits_in_half(chat):
    calls, responses = chat
    responses.append(ollama.ResponseError("too big", 400))
    responses.append(_groups((0, 1)))
    responses.append(_groups((25, 30)))
    out = llm.deduplicate(_items(40), "m")
    assert [len(c["ids"]) for c in calls] == [40, 20, 20]
    kept = {i["title"] for i in out}
    assert {f"Story {i}" for i in range(40)} - kept == {"Story 1", "Story 30"}


def test_split_recurses(chat):
    calls, responses = chat
    responses.append(llm.ChatResult("", done_reason="length"))
    responses.append(llm.ChatResult("", done_reason="length"))
    out = llm.deduplicate(_items(80), "m")
    # The last 40 comes back empty, so it gets a validation retry.
    assert [len(c["ids"]) for c in calls] == [80, 40, 20, 20, 40, 40]
    assert len(out) == 80


def test_leaf_failure_keeps_all(chat):
    calls, responses = chat
    responses.extend([llm.ChatResult("not json")] * 3)
    out = llm.deduplicate(_items(25), "m")
    assert [len(c["ids"]) for c in calls] == [25, 12, 13]
    assert len(out) == 25


# --- Exact-title pre-pass -------------------------------------------------------


def test_exact_titles_removed_before_llm(batch_sizes):
    items = [
        {"title": "Best OCR for Agents", "summary": "short"},
        {"title": "Other story"},
        {"title": "  best ocr   FOR agents ", "summary": "much longer summary"},
    ]
    out = llm.deduplicate(items, "m")
    assert [i["title"] for i in out] == ["Other story", "  best ocr   FOR agents "]
    assert batch_sizes == [2]


def test_exact_title_tie_keeps_first(batch_sizes):
    items = [{"title": "Same", "summary": "a"}, {"title": "SAME", "summary": "b"}]
    out = llm.deduplicate(items, "m")
    assert out == [items[0]]
    assert batch_sizes == []  # one survivor left — no LLM call


def test_distinct_titles_untouched(batch_sizes):
    out = llm.deduplicate(_items(5), "m")
    assert len(out) == 5
    assert batch_sizes == [5]


# --- Validation retry on empty result -------------------------------------------


def test_empty_result_on_large_batch_retries_once(chat, capsys):
    calls, responses = chat
    responses.append(llm.ChatResult('{"groups": []}'))
    responses.append(_groups((0, 1)))
    out = llm.deduplicate(_items(40), "m")
    assert [len(c["ids"]) for c in calls] == [40, 40]
    assert len(out) == 39
    err = capsys.readouterr().err
    assert "validation retry" in err
    assert "retry found 1 duplicate" in err


def test_empty_result_confirmed_by_retry(chat, capsys):
    calls, _ = chat
    out = llm.deduplicate(_items(40), "m")
    assert [len(c["ids"]) for c in calls] == [40, 40]
    assert len(out) == 40
    assert "retry confirmed no duplicates" in capsys.readouterr().err


def test_empty_result_on_small_batch_not_retried(chat):
    calls, _ = chat
    llm.deduplicate(_items(39), "m")
    assert len(calls) == 1


def test_retry_failure_falls_back_to_split(chat):
    calls, responses = chat
    responses.append(llm.ChatResult('{"groups": []}'))
    responses.append(llm.ChatResult("", done_reason="length"))
    llm.deduplicate(_items(40), "m")
    assert [len(c["ids"]) for c in calls] == [40, 40, 20, 20]
