"""The dedup prompt includes learned examples when a block is set."""

from __future__ import annotations

from newsradar import llm

_BATCH = [(0, {"title": "A"}), (1, {"title": "B"})]


def _capture(monkeypatch) -> list[str]:
    prompts: list[str] = []
    monkeypatch.setattr(
        llm, "_chat_meta", lambda p, m, **kw: prompts.append(p) or llm.ChatResult('{"groups": []}')
    )
    return prompts


def test_learned_block_appears_in_dedup_prompt(monkeypatch):
    prompts = _capture(monkeypatch)
    llm.set_learned_dedup_examples("LEARNED EXAMPLES:\nX and Y are different stories.")
    try:
        assert llm.learned_dedup_examples().startswith("LEARNED EXAMPLES")
        llm._dedup_batch(_BATCH, "m")
    finally:
        llm.set_learned_dedup_examples("")
    assert "X and Y are different stories." in prompts[0]
    assert prompts[0].index("LEARNED EXAMPLES") < prompts[0].index("Articles:")
    assert "stories.\n\nArticles:" in prompts[0]


def test_no_block_leaves_prompt_unchanged(monkeypatch):
    prompts = _capture(monkeypatch)
    llm._dedup_batch(_BATCH, "m")
    llm.set_learned_dedup_examples("LEARNED EXAMPLES:\nfoo")
    llm.set_learned_dedup_examples("")
    llm._dedup_batch(_BATCH, "m")
    assert prompts[0] == prompts[1]
    assert "LEARNED EXAMPLES" not in prompts[1]
    assert prompts[1].split("Articles:")[0].endswith("return {\"groups\": []}.\n\n")
