import pytest

from newsradar import llm
from newsradar.tuning import gate
from newsradar.tuning.gate import Case, GateError, GateResult, decide, dedup_case_ok, predict_ads, predict_dedup, stabilize
from newsradar.tuning.judge import DedupCase


def case(id, today_mistake=False, countable=True, expected=False):
    return Case(id=id, title="t", text="x", expected=expected,
                today_mistake=today_mistake, countable=countable)


CASES = [case("today-1", today_mistake=True), case("c2"), case("c3")]
CASES4 = [case("c1", today_mistake=True), case("c2", today_mistake=True), case("c3"), case("c4")]


def test_accepts_when_a_today_mistake_is_fixed_and_nothing_breaks():
    r = decide(CASES, current_ok=[False, True, True], candidate_ok=[True, True, True])
    assert r.accepted and r.fixed == ["today-1"] and r.broken == []
    assert r.current_errors == 1 and r.candidate_errors == 0


def test_rejects_any_regression_even_with_net_gain():
    r = decide(CASES4, current_ok=[False, False, True, True], candidate_ok=[True, True, False, True])
    assert not r.accepted and r.broken == ["c3"]


def test_rejects_when_only_uncountable_mistakes_are_fixed():
    r = decide([case("t", today_mistake=True, countable=False)], [False], [True])
    assert not r.accepted and "no fix" in r.reason


def test_rejects_when_fix_is_not_a_today_mistake():
    r = decide([case("a"), case("b", today_mistake=True)], [False, False], [True, False])
    assert not r.accepted and "no fix" in r.reason


def test_rejects_when_nothing_changes():
    assert not decide(CASES, [True, True, True], [True, True, True]).accepted


def test_stabilize_keeps_only_differences_confirmed_on_rerun():
    current, candidate = [True, True], [False, False]
    reruns = {0: (True, False), 1: (True, True)}
    assert stabilize(current, candidate, lambda i: reruns[i]) == [False, True]


def test_stabilize_rerun_failure_raises_gate_error():
    def rerun(i):
        raise RuntimeError("boom")

    with pytest.raises(GateError):
        stabilize([True, True], [True, False], rerun)


def test_dedup_case_ok():
    c = DedupCase(titles=["A", "A2", "B"], expected_groups=[[0, 1]], kind="missed_duplicate", note="")
    assert dedup_case_ok(c, {1}) and not dedup_case_ok(c, set()) and not dedup_case_ok(c, {1, 2})


def test_dedup_case_ok_wrong_merge_expects_nothing_dropped():
    c = DedupCase(titles=["A", "B"], expected_groups=[[0], [1]], kind="wrong_merge", note="")
    assert dedup_case_ok(c, set()) and not dedup_case_ok(c, {1})


def test_predict_ads_calls_classify_ad_with_truncated_text(monkeypatch):
    seen = []

    def fake(title, text, model):
        seen.append((title, len(text), model))
        return True, "r"

    monkeypatch.setattr(gate, "classify_ad", fake)
    c = Case(id="a", title="T", text="x" * 3000, expected=True, today_mistake=False, countable=True)
    assert predict_ads("m", [c]) == [True]
    assert seen == [("T", 1000, "m")]


def test_predict_ads_raises_gate_error_when_the_model_raises(monkeypatch):
    def boom(*a):
        raise RuntimeError("down")

    monkeypatch.setattr(gate, "classify_ad", boom)
    with pytest.raises(GateError):
        predict_ads("current", [case("a")])


def test_predict_ads_raises_gate_error_when_only_the_candidate_raises(monkeypatch):
    def fake(title, text, model):
        if model == "cand":
            raise RuntimeError("down")
        return False, "ok"

    monkeypatch.setattr(gate, "classify_ad", fake)
    assert predict_ads("cur", [case("a")]) == [False]
    with pytest.raises(GateError):
        predict_ads("cand", [case("a")])


def test_predict_ads_parse_error_raises_gate_error(monkeypatch):
    monkeypatch.setattr(gate, "classify_ad", lambda *a: (False, "parse error"))
    with pytest.raises(GateError):
        predict_ads("m", [case("a")])


def test_predict_dedup_sets_block_and_restores_previous(monkeypatch):
    llm.set_learned_dedup_examples("OLD")
    seen = []

    def fake(batch, model):
        seen.append((llm.learned_dedup_examples(), [i for i, _ in batch], [d["title"] for _, d in batch], model))
        return {1}

    monkeypatch.setattr(gate.llm, "_dedup_voted", fake)
    c = DedupCase(titles=["A", "A2"], expected_groups=[[0, 1]], kind="missed_duplicate", note="")
    try:
        assert predict_dedup([c], "m", "NEW") == [{1}]
        assert seen == [("NEW", [0, 1], ["A", "A2"], "m")]
        assert llm.learned_dedup_examples() == "OLD"
    finally:
        llm.set_learned_dedup_examples("")


def test_predict_dedup_error_raises_gate_error_and_restores_block(monkeypatch):
    llm.set_learned_dedup_examples("OLD")

    def boom(batch, model):
        raise llm._DedupFailed("bad")

    monkeypatch.setattr(gate.llm, "_dedup_voted", boom)
    c = DedupCase(titles=["A", "B"], expected_groups=[[0], [1]], kind="wrong_merge", note="")
    try:
        with pytest.raises(GateError):
            predict_dedup([c], "m", "NEW")
        assert llm.learned_dedup_examples() == "OLD"
    finally:
        llm.set_learned_dedup_examples("")
