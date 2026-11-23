"""Tests for the Opus judge. run_claude is replaced with a fake; no real claude calls."""

from __future__ import annotations

import json
import re

import pytest

from newsradar.claude_cli import ClaudeCallError, ClaudeResult
from newsradar.config import TuningConfig
from newsradar.tuning import judge
from newsradar.tuning.evidence import AdRecord, DedupRecord
from newsradar.tuning.judge import (
    DedupCase,
    JudgeError,
    ProposedRule,
    judge_ads,
    judge_dedup,
    second_opinion_ad,
    second_opinion_dedup,
)

TUNING = TuningConfig()
EX = {"title": "T", "summary": "S", "source": "The AI Report", "link": "https://a.com/", "notes": "n"}


class FakeClaude:
    def __init__(self):
        self.calls: list[dict] = []
        self.reply = lambda prompt: {"verdicts": [verdict(k) for k in keys_in(prompt)], "proposed_rules": []}
        self.error: Exception | None = None
        self.cost = 0.0

    def __call__(self, prompt, model, **kwargs):
        self.calls.append({"prompt": prompt, "model": model, **kwargs})
        if self.error:
            raise self.error
        return ClaudeResult(text="", structured=self.reply(prompt), usage={}, cost_usd=self.cost)


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeClaude()
    monkeypatch.setattr(judge, "run_claude", fake)
    return fake


def rec(key, verdict="not_ad", reason="r", source="The AI Report", title="Title", text_sent="SENT", text_full="FULL"):
    return AdRecord(key=key, date="2026-11-23", topic="AI", title=title, link="https://x.com/a?utm=1",
                    source=source, source_type="email", text_sent=text_sent, text_full=text_full,
                    verdict=verdict, reason=reason, model="ad-detector")


def verdict(key, correct=True, example=None, gate_said_ad=False):
    is_ad = gate_said_ad if correct else not gate_said_ad
    return {"key": key, "correct": correct, "is_ad": is_ad,
            "ad_type": "sponsor_content" if is_ad else "not_ad", "reason": "why",
            "visible_in_excerpt": True, "example": example}


def keys_in(prompt):
    return re.findall(r'"key": "([^"]+)"', prompt)


def drec(titles, dropped, key="d1"):
    return DedupRecord(key=key, date="2026-11-23", topic="AI", titles=titles, dropped=dropped)


TITLES = ["OpenAI ships GPT-6", "GPT-6 is out from OpenAI", "Nvidia earnings beat", "Rust 2.0 released",
          "Apple buys a startup"]


# ---------- judge_ads ----------

def test_judge_ads_calls_isolated_opus_medium_with_schema(fake_claude):
    judge_ads([rec("k1")], TUNING)
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["effort"] == "medium"
    assert call["json_schema"]["type"] == "object" and call["system_prompt"]


def test_judge_ads_batches_and_maps_verdicts_by_key(fake_claude):
    fake_claude.reply = lambda prompt: {"verdicts": [verdict(k) for k in keys_in(prompt)], "proposed_rules": []}
    out = judge_ads([rec(f"k{i}") for i in range(45)], TUNING, batch_size=20)
    assert len(fake_claude.calls) == 3 and [v.key for v in out.verdicts] == [f"k{i}" for i in range(45)]


def test_verdicts_follow_record_order_not_reply_order(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(k) for k in reversed(keys_in(p))], "proposed_rules": []}
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["a", "b", "c"]


def test_missing_key_raises_judge_error(fake_claude):
    fake_claude.reply = lambda prompt: {"verdicts": [], "proposed_rules": []}
    with pytest.raises(JudgeError, match="k1"):
        judge_ads([rec("k1")], TUNING)


@pytest.mark.parametrize("extra", ["k1", "stranger"])
def test_duplicate_or_unknown_key_raises(fake_claude, extra):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1"), verdict(extra)], "proposed_rules": []}
    with pytest.raises(JudgeError, match="k1"):
        judge_ads([rec("k1")], TUNING)


def test_correct_flag_must_agree_with_label(fake_claude):
    # The gate kept the item (not_ad); the judge says it is an ad yet calls the verdict correct.
    bad = {**verdict("k1"), "is_ad": True, "ad_type": "sponsor_content"}
    fake_claude.reply = lambda p: {"verdicts": [bad], "proposed_rules": []}
    with pytest.raises(JudgeError, match="k1"):
        judge_ads([rec("k1")], TUNING)


@pytest.mark.parametrize("is_ad,ad_type", [(True, "not_ad"), (False, "sponsor_content"), (False, "made_up")])
def test_ad_type_must_agree_with_label(fake_claude, is_ad, ad_type):
    bad = {**verdict("k1", correct=not is_ad), "is_ad": is_ad, "ad_type": ad_type}
    fake_claude.reply = lambda p: {"verdicts": [bad], "proposed_rules": []}
    with pytest.raises(JudgeError):
        judge_ads([rec("k1")], TUNING)


def test_mistake_without_example_raises(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1", correct=False, example=None)], "proposed_rules": []}
    with pytest.raises(JudgeError, match="example"):
        judge_ads([rec("k1")], TUNING)


def test_correct_verdict_drops_any_example(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1", example=EX)], "proposed_rules": []}
    assert judge_ads([rec("k1")], TUNING).verdicts[0].example is None


def test_example_summary_capped_and_link_cut_to_domain(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1", correct=False, example={**EX, "summary": "x" * 300,
                                   "link": "https://a.com/p?ref=1"})], "proposed_rules": []}
    ex = judge_ads([rec("k1")], TUNING).verdicts[0].example
    assert len(ex["summary"]) == 200 and ex["link"] == "https://a.com/"


def test_example_gets_stable_id_and_label_from_verdict(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("abcdef123456", correct=False, example=EX)],
                                   "proposed_rules": []}
    ex = judge_ads([rec("abcdef123456", source="The AI Report")], TUNING).verdicts[0].example
    assert ex["id"] == "the-ai-report-2026-11-23-abcdef12"
    assert ex["is_ad"] is True and ex["ad_type"] == "sponsor_content"
    assert set(ex) == {"id", "title", "summary", "source", "link", "is_ad", "ad_type", "notes"}


def test_prompt_carries_both_texts_and_gate_verdict(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1", gate_said_ad=True)], "proposed_rules": []}
    judge_ads([rec("k1", verdict="ad", reason="GATE-REASON", text_sent="EXCERPT-1000", text_full="LONGER-4000")],
              TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    payload = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])
    assert payload[0]["key"] == "k1" and payload[0]["verdict"] == "ad" and payload[0]["reason"] == "GATE-REASON"
    assert payload[0]["text_sent"] == "EXCERPT-1000" and payload[0]["text_full"] == "LONGER-4000"


def test_proposed_rules_parsed_and_leaning_checked(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(k) for k in keys_in(p)],
                                   "proposed_rules": [{"text": "Keep paywalled news.", "leaning": "keep"}]}
    out = judge_ads([rec("k1")], TUNING)
    assert out.proposed_rules == [ProposedRule(text="Keep paywalled news.", leaning="keep")]

    fake_claude.reply = lambda p: {"verdicts": [verdict(k) for k in keys_in(p)],
                                   "proposed_rules": [{"text": "x", "leaning": "sideways"}]}
    with pytest.raises(JudgeError):
        judge_ads([rec("k1")], TUNING)


def test_cost_sums_across_batches(fake_claude):
    fake_claude.cost = 0.25
    out = judge_ads([rec(f"k{i}") for i in range(3)], TUNING, batch_size=1)
    assert out.cost_usd == pytest.approx(0.75)


def test_claude_failure_becomes_judge_error(fake_claude):
    fake_claude.error = ClaudeCallError("rate limited")
    with pytest.raises(JudgeError):
        judge_ads([rec("k1")], TUNING)


def test_missing_structured_output_becomes_judge_error(fake_claude):
    fake_claude.reply = lambda p: None
    with pytest.raises(JudgeError, match="k1"):
        judge_ads([rec("k1")], TUNING)


def test_no_records_makes_no_call(fake_claude):
    out = judge_ads([], TUNING)
    assert out.verdicts == [] and out.cost_usd == 0.0 and fake_claude.calls == []


# ---------- second_opinion_ad ----------

def test_second_opinion_prompt_omits_first_verdict(fake_claude):
    fake_claude.reply = lambda p: {"is_ad": True, "ad_type": "sponsor_content", "reason": "x"}
    second_opinion_ad(rec("k1", verdict="ad", reason="FIRST-REASON"), TUNING)
    assert "FIRST-REASON" not in fake_claude.calls[0]["prompt"]
    assert '"verdict"' not in fake_claude.calls[0]["prompt"]


@pytest.mark.parametrize("answer", [True, False])
def test_second_opinion_ad_returns_is_ad(fake_claude, answer):
    fake_claude.reply = lambda p: {"is_ad": answer, "ad_type": "sponsor_content" if answer else "not_ad",
                                   "reason": "x"}
    assert second_opinion_ad(rec("k1"), TUNING) is answer
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["effort"] == "medium" and call["json_schema"]
    assert call["system_prompt"] != judge._prompt("judge_ads.md")


def test_second_opinion_ad_failure_becomes_judge_error(fake_claude):
    fake_claude.error = ClaudeCallError("boom")
    with pytest.raises(JudgeError, match="k1"):
        second_opinion_ad(rec("k1"), TUNING)


# ---------- judge_dedup ----------

def case_reply(*cases):
    return lambda p: {"cases": list(cases)}


def test_judge_dedup_returns_cases_with_valid_indices(fake_claude):
    fake_claude.cost = 0.5
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "titles": [TITLES[0], TITLES[1], TITLES[3]],
                                    "expected_groups": [[0, 1]], "note": "same launch"})
    cases, cost = judge_dedup([drec(TITLES, [])], TUNING)
    assert cases == [DedupCase(titles=[TITLES[0], TITLES[1], TITLES[3]], expected_groups=[[0, 1]],
                               kind="missed_duplicate", note="same launch")]
    assert cost == 0.5
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["effort"] == "medium" and call["json_schema"]


@pytest.mark.parametrize("groups", [[[0, 3]], [[0]], [[-1, 0]], [[0, 1], [1, 2]]])
def test_judge_dedup_bad_groups_raise(fake_claude, groups):
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "titles": [TITLES[0], TITLES[1], TITLES[3]],
                                    "expected_groups": groups, "note": ""})
    with pytest.raises(JudgeError, match="d1"):
        judge_dedup([drec(TITLES, [])], TUNING)


def test_judge_dedup_title_not_in_batch_raises(fake_claude):
    fake_claude.reply = case_reply({"kind": "wrong_merge", "titles": ["Invented title", TITLES[2]],
                                    "expected_groups": [], "note": ""})
    with pytest.raises(JudgeError, match="not in the batch"):
        judge_dedup([drec(TITLES, [1])], TUNING)


def test_missed_duplicate_needs_a_group(fake_claude):
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "titles": [TITLES[0], TITLES[2]],
                                    "expected_groups": [], "note": ""})
    with pytest.raises(JudgeError):
        judge_dedup([drec(TITLES, [])], TUNING)


def test_wrong_merge_with_no_groups_is_valid(fake_claude):
    fake_claude.reply = case_reply({"kind": "wrong_merge", "titles": [TITLES[2], TITLES[3]],
                                    "expected_groups": [], "note": "different stories"})
    cases, _ = judge_dedup([drec(TITLES, [3])], TUNING)
    assert cases[0].kind == "wrong_merge" and cases[0].expected_groups == []


def test_judge_dedup_one_call_per_record_and_marks_drops(fake_claude):
    fake_claude.cost = 0.1
    fake_claude.reply = case_reply()
    titles = ["Same story", "same  STORY", "Other"]
    cases, cost = judge_dedup([drec(titles, [1], key="a"), drec(TITLES, [], key="b")], TUNING)
    assert cases == [] and len(fake_claude.calls) == 2 and cost == pytest.approx(0.2)
    prompt = fake_claude.calls[0]["prompt"]
    items = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])
    assert items[1] == {"id": 1, "title": "same  STORY", "dropped": True, "exact_title_drop": True}
    assert items[0]["dropped"] is False and items[2]["exact_title_drop"] is False


def test_judge_dedup_failure_names_record(fake_claude):
    fake_claude.error = ClaudeCallError("rate limited")
    with pytest.raises(JudgeError, match="d1"):
        judge_dedup([drec(TITLES, [])], TUNING)


# ---------- second_opinion_dedup ----------

CASE = DedupCase(titles=["A launch", "A launches", "B", "C"], expected_groups=[[0, 1]],
                 kind="missed_duplicate", note="FIRST-NOTE")


@pytest.mark.parametrize("groups,agrees", [([[1, 0]], True), ([], False), ([[0, 1, 2]], False),
                                            ([[0, 1], [2, 3]], False)])
def test_second_opinion_dedup_compares_groups(fake_claude, groups, agrees):
    fake_claude.reply = lambda p: {"groups": groups, "reason": "x"}
    assert second_opinion_dedup(CASE, TUNING) is agrees


def test_second_opinion_dedup_prompt_hides_expected_groups_and_note(fake_claude):
    fake_claude.reply = lambda p: {"groups": [[0, 1]], "reason": "x"}
    second_opinion_dedup(CASE, TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    assert "FIRST-NOTE" not in prompt and "missed_duplicate" not in prompt and "[[0, 1]]" not in prompt
    assert "A launches" in prompt


def test_second_opinion_dedup_bad_index_raises(fake_claude):
    fake_claude.reply = lambda p: {"groups": [[0, 9]], "reason": "x"}
    with pytest.raises(JudgeError):
        second_opinion_dedup(CASE, TUNING)


# ---------- prompts ----------

@pytest.mark.parametrize("name", ["judge_ads.md", "judge_dedup.md", "second_opinion.md"])
def test_prompts_load_from_package(name):
    assert len(judge._prompt(name)) > 500
