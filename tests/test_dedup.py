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
    """Replace the LLM call with a recorder; drops nothing. Voting is off
    so recorded sizes show batching only."""
    sizes: list[int] = []
    monkeypatch.setattr(llm, "_DEDUP_VOTES", 1)

    def fake_batch(batch, model):
        sizes.append(len(batch))
        return {}

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
    monkeypatch.setattr(llm, "_dedup_batch", lambda batch, model: {1: 0})
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
    """Script _chat_meta responses; records (batch ids, num_ctx) per call.
    Voting is off unless a test turns it back on."""
    monkeypatch.setattr(llm, "_DEDUP_VOTES", 1)
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
    """Each group is (keep, *duplicates). Members echo the _items() titles."""
    return llm.ChatResult(json.dumps({"groups": [
        {"members": [{"id": str(i), "title": f"Story {i}"} for i in g], "keep": str(g[0]), "confidence": confidence}
        for g in groups
    ]}))


def test_batch_drops_all_but_keep(chat):
    _, responses = chat
    responses.append(_groups((0, 1, 4), (2, 3)))
    assert set(llm._dedup_batch(_batch(5), "m")) == {1, 3, 4}


def test_batch_omitted_ids_are_kept(chat):
    _, responses = chat
    responses.append(llm.ChatResult('{"groups": []}'))
    assert set(llm._dedup_batch(_batch(5), "m")) == set()


def test_batch_ignores_low_confidence(chat):
    _, responses = chat
    responses.append(_groups((0, 1), confidence=0.2))
    assert set(llm._dedup_batch(_batch(5), "m")) == set()


def test_batch_ignores_unknown_ids(chat):
    _, responses = chat
    responses.append(_groups((0, 99, 1), (98, 2), (3,)))
    assert set(llm._dedup_batch(_batch(5), "m")) == {1}


def test_batch_keep_outside_group_is_ignored(chat):
    _, responses = chat
    responses.append(llm.ChatResult(
        '{"groups": [{"members": [{"id": "1", "title": "Story 1"}, {"id": "2", "title": "Story 2"}], "keep": "0"}]}'
    ))
    assert set(llm._dedup_batch(_batch(5), "m")) == set()


def test_batch_overlapping_groups_never_drop_a_keep(chat):
    _, responses = chat
    responses.append(_groups((0, 1), (1, 0)))
    assert set(llm._dedup_batch(_batch(5), "m")) == set()


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
    assert set(llm._dedup_batch(_batch(3), "m")) == {1, 2}


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
    assert [len(c["ids"]) for c in calls] == [80, 40, 20, 20, 40]
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


# --- Title check on group members ---------------------------------------------


def _members(*pairs, keep, confidence=0.9) -> dict:
    return {"members": [{"id": str(i), "title": t} for i, t in pairs], "keep": str(keep), "confidence": confidence}


def test_member_with_wrong_title_is_ignored(chat, capsys):
    _, responses = chat
    # The model means Story 1 and Story 2 but cites id 3 for Story 2.
    responses.append(llm.ChatResult(json.dumps({"groups": [_members((1, "Story 1"), (3, "Story 2"), keep=1)]})))
    assert llm._dedup_batch(_batch(5), "m") == {}
    assert "didn't match its id" in capsys.readouterr().err


def test_wrong_member_does_not_spoil_the_rest_of_its_group(chat):
    _, responses = chat
    responses.append(llm.ChatResult(json.dumps({"groups": [
        _members((0, "Story 0"), (1, "Story 1"), (3, "Story 4"), keep=0),
    ]})))
    assert llm._dedup_batch(_batch(5), "m") == {1: 0}


def test_member_without_title_is_ignored(chat):
    _, responses = chat
    responses.append(llm.ChatResult('{"groups": [{"members": [{"id": "0"}, {"id": "1"}], "keep": "0"}]}'))
    assert llm._dedup_batch(_batch(5), "m") == {}


@pytest.mark.parametrize("echoed, actual", [
    ("Meta releases Muse Spark Open, a new 'open-weight' AI model",
     "Meta releases Muse Spark Open, a new \u2018open-weight\u2019 AI model"),
    ("Bluesky's AI tool now lets you make games", "Bluesky&#8217;s AI tool now lets you make games and apps to share"),
    ("NVIDIA RELEASES NEMOTRON 3.5 SUPER VL", "NVIDIA Releases Nemotron 3.5 Super VL"),
])
def test_titles_match_tolerates_quotes_entities_case_and_truncation(echoed, actual):
    assert llm._titles_match(echoed, actual)


@pytest.mark.parametrize("echoed, actual", [
    ("NVIDIA Nemotron 3.5 Super VL Powers Specialized Agents",
     "Dial Up the Cloud: NVIDIA DLSS 4.5 Controls Arrive on GeForce NOW"),
    ("", "Anything"),
])
def test_titles_match_rejects_a_different_article(echoed, actual):
    assert not llm._titles_match(echoed, actual)


# --- Voting across runs ---------------------------------------------------------


@pytest.fixture
def votes(monkeypatch):
    monkeypatch.setattr(llm, "_DEDUP_VOTES", 3)


def test_only_majority_drops_are_kept(chat, votes):
    calls, responses = chat
    responses.append(_groups((0, 1), (2, 3)))
    responses.append(_groups((0, 1)))
    responses.append(_groups((0, 1), (4, 5)))
    assert llm._dedup_voted(_batch(12), "m") == {1}
    assert len(calls) == 3


def test_one_failed_run_still_votes_with_the_other_two(chat, votes):
    _, responses = chat
    responses.append(llm.ChatResult("not json"))
    responses.append(_groups((0, 1)))
    responses.append(_groups((0, 1)))
    assert llm._dedup_voted(_batch(12), "m") == {1}


def test_two_good_runs_must_agree(chat, votes):
    _, responses = chat
    responses.append(llm.ChatResult("not json"))
    responses.append(_groups((0, 1)))
    responses.append(_groups((2, 3)))
    assert llm._dedup_voted(_batch(12), "m") == set()


def test_most_runs_failing_raises(chat, votes):
    _, responses = chat
    responses.extend([llm.ChatResult("not json"), llm.ChatResult("not json"), _groups((0, 1))])
    with pytest.raises(llm._DedupFailed):
        llm._dedup_voted(_batch(12), "m")


def test_votes_never_remove_a_whole_story(chat, votes):
    _, responses = chat
    # Each run keeps a different member, so every member gets two drop votes.
    responses.append(_groups((0, 1, 2)))
    responses.append(_groups((1, 0, 2)))
    responses.append(_groups((2, 0, 1)))
    dropped = llm._dedup_voted(_batch(12), "m")
    assert len(dropped) == 2 and dropped < {0, 1, 2}


def test_runs_keeping_different_members_still_drop_one(chat, votes):
    _, responses = chat
    responses.append(_groups((5, 3)))
    responses.append(_groups((3, 5)))
    responses.append(_groups())
    assert llm._dedup_voted(_batch(12), "m") == {5}


def test_group_keeps_the_most_chosen_member(chat, votes):
    _, responses = chat
    responses.append(_groups((2, 0, 1)))
    responses.append(_groups((2, 0, 1)))
    responses.append(_groups((0, 1, 2)))
    assert llm._dedup_voted(_batch(12), "m") == {0, 1}


def test_deduplicate_votes_by_default(chat, votes):
    calls, responses = chat
    responses.extend([_groups((0, 1)), _groups((0, 1)), _groups()])
    out = llm.deduplicate(_items(12), "m")
    assert len(calls) == 3
    assert "Story 1" not in {i["title"] for i in out}


# --- arXiv papers -----------------------------------------------------------------


def test_arxiv_papers_skip_the_llm_pass(batch_sizes):
    items = _items(4) + [{"title": f"Paper {i}", "_is_arxiv": True} for i in range(3)]
    out = llm.deduplicate(items, "m")
    assert batch_sizes == [4]
    assert len(out) == 7


def test_arxiv_cross_lists_still_dropped_by_exact_title(batch_sizes):
    items = [
        {"title": "Story 0"},
        {"title": "Same paper", "_is_arxiv": True},
        {"title": "Story 1"},
        {"title": "same  PAPER", "_is_arxiv": True},
    ]
    out = llm.deduplicate(items, "m")
    assert [i["title"] for i in out] == ["Story 0", "Same paper", "Story 1"]
    assert batch_sizes == [2]


def test_only_papers_left_returns_them_all(batch_sizes):
    items = [{"title": "Story 0"}] + [{"title": f"Paper {i}", "_is_arxiv": True} for i in range(3)]
    out = llm.deduplicate(items, "m")
    assert len(out) == 4
    assert batch_sizes == []


# --- Removed-item tracking ------------------------------------------------------


def test_records_removed_items(monkeypatch):
    monkeypatch.setattr(llm, "_dedup_batch", lambda batch, model: {1: 0})
    items = [{"title": "A"}, {"title": "B"}, {"title": "a"}, {"title": "C"}]
    removed: list[dict] = []
    llm.deduplicate(items, "m", removed=removed)
    assert sorted((r["title"], r["reason"]) for r in removed) == [
        ("B", "duplicate"),
        ("a", "duplicate (same title)"),
    ]
