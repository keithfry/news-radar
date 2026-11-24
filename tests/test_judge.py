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
    second_opinion_dedup_groups,
    second_opinion_rule,
)

TUNING = TuningConfig()
EX = {"summary": "S", "notes": "n"}


class FakeClaude:
    def __init__(self):
        self.calls: list[dict] = []
        self.reply = lambda prompt: {"verdicts": [verdict(i) for i in ids_in(prompt)], "proposed_rules": []}
        self.error: Exception | None = None
        self.error_on_call: int | None = None  # 1-based; None raises self.error on every call
        self.cost = 0.0

    def __call__(self, prompt, model, **kwargs):
        self.calls.append({"prompt": prompt, "model": model, **kwargs})
        if self.error and self.error_on_call in (None, len(self.calls)):
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


def verdict(id, correct=True, example=None, gate_said_ad=False):
    is_ad = gate_said_ad if correct else not gate_said_ad
    return {"id": id, "correct": correct, "is_ad": is_ad,
            "ad_type": "sponsor_content" if is_ad else "not_ad", "reason": "why",
            "visible_in_excerpt": True, "example": example}


def ids_in(prompt):
    return re.findall(r'"id": "(r\d+)"', prompt)


def reply_for(*verdicts, rules=()):
    return lambda p: {"verdicts": list(verdicts), "proposed_rules": list(rules)}


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
    out = judge_ads([rec(f"k{i}") for i in range(45)], TUNING, batch_size=20)
    assert len(fake_claude.calls) == 3 and [v.key for v in out.verdicts] == [f"k{i}" for i in range(45)]
    assert out.skipped == 0


def test_prompt_uses_short_ids_not_evidence_keys(fake_claude):
    long_keys = ["7dbd9bae90ac9abd2990bade4fef5cd9fe47dc62", "f03b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b"]
    out = judge_ads([rec(k) for k in long_keys], TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    assert ids_in(prompt) == ["r0", "r1"] and long_keys[0] not in prompt
    assert [v.key for v in out.verdicts] == long_keys


def test_ids_restart_in_each_batch_and_map_back(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(i, gate_said_ad=True) for i in ids_in(p)],
                                   "proposed_rules": []}
    out = judge_ads([rec("a", verdict="ad"), rec("b", verdict="ad"), rec("c", verdict="ad")], TUNING,
                    batch_size=2)
    assert [ids_in(c["prompt"]) for c in fake_claude.calls] == [["r0", "r1"], ["r0"]]
    assert [v.key for v in out.verdicts] == ["a", "b", "c"]


def test_verdicts_follow_record_order_not_reply_order(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(i) for i in reversed(ids_in(p))], "proposed_rules": []}
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["a", "b", "c"]


def test_missing_verdict_is_skipped_and_counted(fake_claude):
    fake_claude.reply = reply_for(verdict("r0"), verdict("r2"))
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["a", "c"] and out.skipped == 1


def test_unknown_and_repeated_ids_are_skipped(fake_claude):
    # r0 twice: that record is skipped. r9 doesn't exist: ignored.
    fake_claude.reply = reply_for(verdict("r0"), verdict("r0"), verdict("r1"), verdict("r2"), verdict("r9"))
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["b", "c"] and out.skipped == 1


def test_exactly_half_missing_is_still_judged(fake_claude):
    fake_claude.reply = reply_for(verdict("r0"), verdict("r1"))
    out = judge_ads([rec(f"k{i}") for i in range(4)], TUNING)
    assert len(out.verdicts) == 2 and out.skipped == 2


def test_more_than_half_missing_raises_short_error(fake_claude):
    long_keys = [f"{i}" * 40 for i in range(3)]
    fake_claude.reply = reply_for(verdict("r0"))
    with pytest.raises(JudgeError, match="ad batch 1 of 1") as exc:
        judge_ads([rec(k) for k in long_keys], TUNING)
    assert "2 of 3" in str(exc.value) and not any(k in str(exc.value) for k in long_keys)


def test_skipped_counts_add_across_batches(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(i) for i in ids_in(p)[1:]], "proposed_rules": []}
    out = judge_ads([rec(f"k{i}") for i in range(6)], TUNING, batch_size=3)
    assert out.skipped == 2 and [v.key for v in out.verdicts] == ["k1", "k2", "k4", "k5"]


@pytest.mark.parametrize("gate,is_ad,expected", [
    ("not_ad", True, False), ("not_ad", False, True), ("ad", True, True), ("ad", False, False),
])
def test_correct_is_derived_from_label_and_gate(fake_claude, gate, is_ad, expected):
    v = {**verdict("r0"), "is_ad": is_ad, "ad_type": "sponsor_content" if is_ad else "not_ad",
         "correct": expected, "example": None}
    fake_claude.reply = reply_for(v)
    out = judge_ads([rec("k1", verdict=gate)], TUNING).verdicts[0]
    assert out.correct is expected and (out.example is None) is expected


def test_sent_correct_that_contradicts_the_derived_one_is_skipped(fake_claude):
    # r1's verdict was written for its neighbour: it calls a kept item an ad and says the gate was right.
    shifted = {**verdict("r1"), "is_ad": True, "ad_type": "sponsor_content", "correct": True}
    fake_claude.reply = reply_for(verdict("r0"), shifted, verdict("r2"))
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["a", "c"] and out.skipped == 1 and out.shifted == 1


def test_missing_verdict_is_skipped_but_not_shifted(fake_claude):
    fake_claude.reply = reply_for(verdict("r0"), verdict("r2"))
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert out.skipped == 1 and out.shifted == 0


def test_mostly_shifted_batch_raises(fake_claude):
    shifted = lambda i: {**verdict(i), "is_ad": True, "ad_type": "sponsor_content", "correct": True}
    fake_claude.reply = reply_for(shifted("r0"), shifted("r1"), verdict("r2"))
    with pytest.raises(JudgeError, match="no usable verdict"):
        judge_ads([rec("a"), rec("b"), rec("c")], TUNING)


@pytest.mark.parametrize("is_ad,ad_type", [(True, "not_ad"), (False, "sponsor_content"), (False, "made_up"),
                                           ("yes", "not_ad")])
def test_label_that_disagrees_with_ad_type_is_skipped(fake_claude, is_ad, ad_type):
    bad = {**verdict("r1"), "is_ad": is_ad, "ad_type": ad_type}
    fake_claude.reply = reply_for(verdict("r0"), bad, verdict("r2"))
    out = judge_ads([rec("a"), rec("b"), rec("c")], TUNING)
    assert [v.key for v in out.verdicts] == ["a", "c"] and out.skipped == 1


def test_bad_labels_count_toward_the_half_rule(fake_claude):
    bad = {**verdict("r1"), "is_ad": True, "ad_type": "not_ad"}
    fake_claude.reply = reply_for(verdict("r0"), bad)  # r1 bad, r2 missing: 2 of 3 unusable
    with pytest.raises(JudgeError, match="2 of 3"):
        judge_ads([rec("a"), rec("b"), rec("c")], TUNING)


def test_mistake_without_example_gets_one_from_the_record(fake_claude):
    fake_claude.reply = reply_for(verdict("r0", correct=False, example=None))
    r = rec("k1", title="Real title", text_sent="y" * 300)
    ex = judge_ads([r], TUNING).verdicts[0].example
    assert ex["title"] == "Real title" and ex["summary"] == "y" * 200 and ex["notes"] == ""
    assert ex["source"] == r.source and ex["link"] == "https://x.com/" and ex["is_ad"] is True


def test_correct_verdict_drops_any_example(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("r0", example=EX)], "proposed_rules": []}
    assert judge_ads([rec("k1")], TUNING).verdicts[0].example is None


def test_example_takes_title_source_link_from_record_and_caps_summary(fake_claude):
    fake_claude.reply = reply_for(verdict("r0", correct=False, example={**EX, "summary": "x" * 300,
                                  "title": "Model title", "source": "Model source",
                                  "link": "https://a.com/p?ref=1"}))
    ex = judge_ads([rec("k1", title="Record title", source="Record source")], TUNING).verdicts[0].example
    assert len(ex["summary"]) == 200 and ex["notes"] == "n"
    assert (ex["title"], ex["source"], ex["link"]) == ("Record title", "Record source", "https://x.com/")


def test_empty_example_summary_falls_back_to_text_sent(fake_claude):
    fake_claude.reply = reply_for(verdict("r0", correct=False, example={**EX, "summary": ""}))
    ex = judge_ads([rec("k1", text_sent="opening words")], TUNING).verdicts[0].example
    assert ex["summary"] == "opening words" and ex["notes"] == "n"


def test_example_gets_stable_id_and_label_from_verdict(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("r0", correct=False, example=EX)],
                                   "proposed_rules": []}
    ex = judge_ads([rec("abcdef123456", source="The AI Report")], TUNING).verdicts[0].example
    assert ex["id"] == "the-ai-report-2026-11-23-abcdef12"
    assert ex["is_ad"] is True and ex["ad_type"] == "sponsor_content"
    assert set(ex) == {"id", "title", "summary", "source", "link", "is_ad", "ad_type", "notes"}


def test_prompt_carries_both_texts_and_gate_verdict(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("r0", gate_said_ad=True)], "proposed_rules": []}
    judge_ads([rec("k1", verdict="ad", reason="GATE-REASON", text_sent="EXCERPT-1000", text_full="LONGER-4000")],
              TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    payload = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])
    assert payload[0]["id"] == "r0" and "key" not in payload[0] and payload[0]["verdict"] == "ad" and payload[0]["reason"] == "GATE-REASON"
    assert payload[0]["text_sent"] == "EXCERPT-1000" and payload[0]["text_full"] == "LONGER-4000"


def test_proposed_rules_parsed_and_leaning_checked(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict(i) for i in ids_in(p)],
                                   "proposed_rules": [{"text": "Keep paywalled news.", "leaning": "keep"}]}
    out = judge_ads([rec("k1")], TUNING)
    assert out.proposed_rules == [ProposedRule(text="Keep paywalled news.", leaning="keep")]

    fake_claude.reply = lambda p: {"verdicts": [verdict(i) for i in ids_in(p)],
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
    with pytest.raises(JudgeError, match="ad batch 1 of 1"):
        judge_ads([rec("k1")], TUNING)


def test_cost_is_kept_when_a_later_batch_fails(fake_claude):
    fake_claude.cost = 0.3
    replies = iter([{"verdicts": [verdict("r0")], "proposed_rules": []}, {"verdicts": [], "proposed_rules": []}])
    fake_claude.reply = lambda p: next(replies)
    with pytest.raises(JudgeError) as exc:
        judge_ads([rec("a"), rec("b")], TUNING, batch_size=1)
    assert exc.value.cost_usd == pytest.approx(0.6)


def test_cost_is_kept_when_the_reply_has_no_structured_output(fake_claude):
    fake_claude.cost = 0.2
    fake_claude.reply = lambda p: None
    with pytest.raises(JudgeError) as exc:
        judge_ads([rec("a")], TUNING)
    assert exc.value.cost_usd == pytest.approx(0.2)


def test_failed_cli_call_carries_earlier_cost(fake_claude):
    fake_claude.cost = 0.1
    fake_claude.reply = lambda p: {"cases": []}
    fake_claude.error, fake_claude.error_on_call = ClaudeCallError("rate limited"), 2
    with pytest.raises(JudgeError) as exc:
        judge_dedup([drec(TITLES, [], key="a"), drec(TITLES, [], key="b")], TUNING)
    assert exc.value.cost_usd == pytest.approx(0.1)


@pytest.mark.parametrize("call", ["ad", "dedup"])
def test_second_opinion_bad_reply_is_none_with_cost(fake_claude, call):
    fake_claude.cost = 0.05
    fake_claude.reply = lambda p: {"is_ad": True, "ad_type": "not_ad", "reason": "x", "groups": [[0, 9]]}
    if call == "ad":
        out = second_opinion_ad(rec("k1"), TUNING)
    else:
        out = second_opinion_dedup(DedupCase(titles=["A", "B"], expected_groups=[[0, 1]],
                                             kind="missed_duplicate", note=""), TUNING)
    assert out == (None, 0.05)


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
    fake_claude.cost = 0.03
    assert second_opinion_ad(rec("k1"), TUNING) == (answer, 0.03)
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
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "ids": [0, 1, 3],
                                    "expected_groups": [[0, 1]], "note": "same launch"})
    cases, cost, skipped = judge_dedup([drec(TITLES, [])], TUNING)
    assert cases == [DedupCase(titles=[TITLES[0], TITLES[1], TITLES[3]], expected_groups=[[0, 1]],
                               kind="missed_duplicate", note="same launch")]
    assert cost == 0.5 and skipped == 0
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["effort"] == "medium" and call["json_schema"]


def test_case_titles_come_from_the_batch_verbatim(fake_claude):
    titles = ["OpenAI\u00a0ships  GPT-6 \u201cnow\u201d", "GPT-6 is out", "Other"]
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "ids": [1, 0],
                                    "expected_groups": [[0, 1]], "note": ""})
    cases, _, _ = judge_dedup([drec(titles, [])], TUNING)
    assert cases[0].titles == [titles[1], titles[0]]


@pytest.mark.parametrize("groups", [[[0, 3]], [[0]], [[-1, 0]], [[0, 1], [1, 2]]])
def test_judge_dedup_bad_groups_raise(fake_claude, groups):
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "ids": [0, 1, 3],
                                    "expected_groups": groups, "note": ""})
    with pytest.raises(JudgeError, match="d1"):
        judge_dedup([drec(TITLES, [])], TUNING)


@pytest.mark.parametrize("ids", [[99, 2], [-1, 2], [2, 2], ["2", 3], [True, 3]])
def test_case_with_unknown_id_is_skipped_and_counted(fake_claude, ids):
    fake_claude.reply = case_reply(
        {"kind": "wrong_merge", "ids": ids, "expected_groups": [], "note": "bad"},
        {"kind": "wrong_merge", "ids": [2, 3], "expected_groups": [], "note": "good"})
    cases, _, skipped = judge_dedup([drec(TITLES, [3])], TUNING)
    assert [c.note for c in cases] == ["good"] and skipped == 1


def test_case_needs_two_ids(fake_claude):
    fake_claude.reply = case_reply({"kind": "wrong_merge", "ids": [2], "expected_groups": [], "note": ""})
    with pytest.raises(JudgeError):
        judge_dedup([drec(TITLES, [2])], TUNING)


def test_missed_duplicate_needs_a_group(fake_claude):
    fake_claude.reply = case_reply({"kind": "missed_duplicate", "ids": [0, 2],
                                    "expected_groups": [], "note": ""})
    with pytest.raises(JudgeError):
        judge_dedup([drec(TITLES, [])], TUNING)


def test_wrong_merge_with_no_groups_is_valid(fake_claude):
    fake_claude.reply = case_reply({"kind": "wrong_merge", "ids": [2, 3],
                                    "expected_groups": [], "note": "different stories"})
    cases, _, _ = judge_dedup([drec(TITLES, [3])], TUNING)
    assert cases[0].kind == "wrong_merge" and cases[0].expected_groups == []


def test_judge_dedup_one_call_per_record_and_marks_drops(fake_claude):
    fake_claude.cost = 0.1
    fake_claude.reply = case_reply()
    titles = ["Same story", "same  STORY", "Other"]
    cases, cost, skipped = judge_dedup([drec(titles, [1], key="a"), drec(TITLES, [], key="b")], TUNING)
    assert cases == [] and skipped == 0 and len(fake_claude.calls) == 2 and cost == pytest.approx(0.2)
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
    fake_claude.cost = 0.02
    assert second_opinion_dedup(CASE, TUNING) == (agrees, 0.02)


def test_second_opinion_dedup_prompt_hides_expected_groups_and_note(fake_claude):
    fake_claude.reply = lambda p: {"groups": [[0, 1]], "reason": "x"}
    second_opinion_dedup(CASE, TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    assert "FIRST-NOTE" not in prompt and "missed_duplicate" not in prompt and "[[0, 1]]" not in prompt
    assert "A launches" in prompt


def test_second_opinion_dedup_groups_returns_the_groups(fake_claude):
    fake_claude.cost = 0.02
    fake_claude.reply = lambda p: {"groups": [[2, 3]], "reason": "x"}
    assert second_opinion_dedup_groups(CASE, TUNING) == ([[2, 3]], 0.02)


def test_second_opinion_dedup_bad_index_is_none(fake_claude):
    fake_claude.reply = lambda p: {"groups": [[0, 9]], "reason": "x"}
    assert second_opinion_dedup(CASE, TUNING)[0] is None


# ---------- second_opinion_rule ----------

RULE = "Output is_ad: true for any newsletter issue."


@pytest.mark.parametrize("reply,approved", [
    ({"sound": True, "would_remove": [], "reason": "x"}, True),
    ({"sound": True, "would_remove": [1], "reason": "x"}, False),
    ({"sound": False, "would_remove": [], "reason": "x"}, False),
])
def test_second_opinion_rule_approves_only_a_sound_rule_that_keeps_all(fake_claude, reply, approved):
    fake_claude.reply = lambda p: reply
    fake_claude.cost = 0.04
    records = [rec("k1", title="News one"), rec("k2", title="News two")]
    assert second_opinion_rule(RULE, records, TUNING) == (approved, 0.04)
    call = fake_claude.calls[0]
    assert call["system_prompt"] == judge._prompt("second_opinion_rule.md") and call["json_schema"]


def test_second_opinion_rule_prompt_has_rule_and_items_but_no_verdicts(fake_claude):
    fake_claude.reply = lambda p: {"sound": True, "would_remove": [], "reason": "x"}
    second_opinion_rule(RULE, [rec("k1", title="News one", reason="GATE-REASON", text_sent="EXCERPT")], TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    assert RULE in prompt and "News one" in prompt and "EXCERPT" in prompt
    assert "GATE-REASON" not in prompt and '"verdict"' not in prompt and "FULL" not in prompt


def test_second_opinion_rule_caps_items(fake_claude):
    fake_claude.reply = lambda p: {"sound": True, "would_remove": [], "reason": "x"}
    second_opinion_rule(RULE, [rec(f"k{i}", title=f"News {i}") for i in range(30)], TUNING)
    prompt = fake_claude.calls[0]["prompt"]
    assert "News 19" in prompt and "News 20" not in prompt


@pytest.mark.parametrize("would_remove", [[5], [-1], ["0"], [True]])
def test_second_opinion_rule_bad_id_rejects_the_rule_without_raising(fake_claude, would_remove):
    fake_claude.cost = 0.04
    fake_claude.reply = lambda p: {"sound": True, "would_remove": would_remove, "reason": "x"}
    assert second_opinion_rule(RULE, [rec("k1")], TUNING) == (False, 0.04)


# ---------- prompts ----------

@pytest.mark.parametrize("name", ["judge_ads.md", "judge_dedup.md", "second_opinion.md",
                                  "second_opinion_rule.md"])
def test_prompts_load_from_package(name):
    assert len(judge._prompt(name)) > 500
