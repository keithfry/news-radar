"""Tests for `newsradar tune`. Every judge, gate, Ollama build and `ollama rm`
call is replaced with a fake in the tuner, gating or models module, where the
call is made; nothing real runs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from newsradar import cli
from newsradar.config import load_config
from newsradar.tuning import gating, judge, models, tuner
from newsradar.tuning.evidence import AdRecord, DedupRecord
from newsradar.tuning.gate import GateError
from newsradar.tuning.judge import AdJudgement, AdVerdict, DedupCase, JudgeError, ProposedRule
from newsradar.tuning.store import TuningStore
from newsradar.tuning.tuner import rebuild_live_model, rollback, run_tune

DATE = "2026-11-23"
LIVE = "ad-detector"  # from the test config's "ollama/ad-detector"
CANDIDATE = "ad-detector-candidate"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("AD_DETECTOR_MODEL", raising=False)
    monkeypatch.delenv("DEDUP_MODEL", raising=False)
    (tmp_path / "t.toml").write_text('[topic]\nname = "t"\nclassifier_prompt = "p"\n')
    path = tmp_path / "config.toml"
    path.write_text(
        '[topics]\nfiles = ["t.toml"]\n\n'
        '[models]\nad_detector_model = "ollama/ad-detector"\ndedup_model = "llama3.2"\n\n'
        '[tuning]\nenabled = true\ndata_dir = "data"\n'
    )
    return load_config(path, env_file=tmp_path / "none.env")


def ad_rec(key, verdict, title, source="The AI Report"):
    return AdRecord(key=key, date=DATE, topic="AI", title=title, link="https://x.com/a?ref=1",
                    source=source, source_type="email", text_sent=f"text of {title}",
                    text_full=f"full text of {title}", verdict=verdict, reason="gate reason",
                    model="ollama/ad-detector")


RECORDS = [
    ad_rec("k_ok_aaaaaaaa", "not_ad", "OpenAI ships a model", source="Decoder"),
    ad_rec("k_fp_aaaaaaaa", "ad", "AI Report issue 42"),        # a false removal
    ad_rec("k_ad_aaaaaaaa", "ad", "Refer a friend and win", source="Hustle"),
    ad_rec("k_h_aaaaaaaaa", "heuristic", "Sponsored: buy now"),  # old evidence; load_day skips it
]
DEDUP = DedupRecord(key="d1", date=DATE, topic="AI",
                    titles=["A launches X", "B raises money", "C ships Y"], dropped=[1])


@pytest.fixture
def evidence(cfg):
    day = cfg.tuning.data_dir / "evidence" / DATE
    day.mkdir(parents=True)
    lines = [json.dumps({"kind": "ad", **asdict(r)}) for r in RECORDS]
    lines.append(json.dumps({"kind": "dedup", **asdict(DEDUP)}))
    (day / "AI.jsonl").write_text("\n".join(lines) + "\n")
    old = cfg.tuning.data_dir / "evidence" / "2026-10-01"
    old.mkdir(parents=True)
    (old / "AI.jsonl").write_text("")
    return day


class Fakes:
    def __init__(self):
        self.judge_error: Exception | None = None
        self.judged: list[AdRecord] = []
        self.unjudged_keys: set[str] = set()  # records the judge returns no verdict for
        self.mistakes = {"k_fp_aaaaaaaa": (False, True)}  # key -> (correct is_ad, visible)
        self.rules: list[ProposedRule] = []
        self.dedup_cases: list[DedupCase] = []
        self.second_opinion_ad = True
        self.second_opinion_dedup = True
        self.second_opinion_rule = True
        self.second_cost = 0.01
        self.rule_checks: list[tuple[str, list[AdRecord]]] = []
        self.second_ad_keys: list[str] = []
        self.second_ad_answers: dict[str, bool | None] = {}  # per record key; else second_opinion_ad
        self.second_dedup_answers: list[bool | None] = []  # answered in order; then second_opinion_dedup
        self.second_dedup_groups: list[list[list[int]] | None] = []  # D1 blocker answers, in order
        self.second_dedup_group_calls = 0
        self.second_pass_break_ids: set[str] = set()  # case ids the candidate gets wrong once rebuilt
        self.candidate_break_ids: set[str] = set()  # case ids the candidate gets wrong
        self.live_wrong_ids: set[str] = set()  # case ids the live model gets wrong and the candidate fixes
        self.ad_case_ids: list[str] = []
        self.candidate_fix = False
        self.candidate_break_base = False
        self.dedup_gate_error = False
        self.installed: list[str] = []
        self.modelfiles: dict[str, str] = {}
        self.fail_install_of: str | None = None
        self.fail_install_times: int | None = None  # None fails every time
        self.removed: list[str] = []
        self.live_text = 'FROM /blobs/live-weights\nSYSTEM """the live model before this run"""\n'
        self.show_error = False
        self.show_fail_from: int | None = None  # fail from this call on (0 is the preflight read)
        self.shown: list[str] = []
        self.ollama_missing = False

    # scenario helpers
    def candidate_fixes_today_mistake(self):
        self.candidate_fix = True

    def candidate_breaks_a_base_example(self):
        self.candidate_fix = True
        self.candidate_break_base = True

    def judge_proposes_ad_example(self):
        self.mistakes["k_ok_aaaaaaaa"] = (True, True)  # the gate missed an ad

    # fakes
    def judge_ads(self, records, tuning):
        if self.judge_error:
            raise self.judge_error
        self.judged = list(records)
        verdicts = []
        for r in records:
            if r.key in self.unjudged_keys:
                continue
            gate_ad = r.verdict == "ad"
            if r.key in self.mistakes:
                is_ad, visible = self.mistakes[r.key]
                v = AdVerdict(key=r.key, correct=False, is_ad=is_ad,
                              ad_type="sponsor_content" if is_ad else "not_ad",
                              reason="judge reason", visible_in_excerpt=visible, example=None)
                v.example = judge._make_example(r, v, {"summary": "a summary", "notes": "n"})
            else:
                v = AdVerdict(key=r.key, correct=True, is_ad=gate_ad,
                              ad_type="sponsor_content" if gate_ad else "not_ad",
                              reason="judge agrees", visible_in_excerpt=True, example=None)
            verdicts.append(v)
        return AdJudgement(verdicts=verdicts, proposed_rules=list(self.rules), cost_usd=0.25,
                           skipped=len(records) - len(verdicts))

    def judge_dedup(self, records, tuning):
        return list(self.dedup_cases), 0.5, 2

    def second_ad(self, record, tuning):
        self.second_ad_keys.append(record.key)
        answer = self.second_ad_answers.get(record.key, self.second_opinion_ad)
        if isinstance(answer, list):
            answer = answer.pop(0)
        return answer, self.second_cost

    def second_dedup(self, case, tuning):
        if self.second_dedup_answers:
            return self.second_dedup_answers.pop(0), self.second_cost
        return self.second_opinion_dedup, self.second_cost

    def second_dedup_groups_fake(self, case, tuning):
        self.second_dedup_group_calls += 1
        return self.second_dedup_groups.pop(0), self.second_cost

    def second_rule(self, rule_text, records, tuning):
        self.rule_checks.append((rule_text, list(records)))
        return self.second_opinion_rule, self.second_cost

    def predict_ads(self, model, cases):
        out = []
        for c in cases:
            self.ad_case_ids.append(c.id)
            p = (not c.expected) if c.today_mistake else c.expected  # the live gate's behavior
            if model != CANDIDATE and c.id in self.live_wrong_ids:
                p = not c.expected
            if model == CANDIDATE:
                if self.candidate_fix and c.today_mistake:
                    p = c.expected
                if self.candidate_break_base and c.id.startswith("base:"):
                    p = not c.expected
                if c.id in self.candidate_break_ids:
                    p = not c.expected
                if c.id in self.second_pass_break_ids and self.installed.count(CANDIDATE) >= 2:
                    p = not c.expected
            out.append(p)
        return out

    def predict_dedup(self, cases, model, block):
        if self.dedup_gate_error:
            raise GateError("ollama is down")
        # The current prompt wrongly drops id 1; a block that holds the case fixes it.
        return [set() if c.titles[0] in block else {1} for c in cases]

    def install(self, modelfile_path, model_name):
        self.installed.append(model_name)
        self.modelfiles[model_name] = Path(modelfile_path).read_text()
        if model_name == self.fail_install_of and self.fail_install_times != 0:
            if self.fail_install_times is not None:
                self.fail_install_times -= 1
            raise subprocess.CalledProcessError(1, ["ollama", "create", model_name])

    def remove(self, model_name):
        self.removed.append(model_name)

    def show(self, model_name):
        """`ollama show --modelfile`: what the fake last installed under that name."""
        self.shown.append(model_name)
        failing = self.show_fail_from is not None and len(self.shown) > self.show_fail_from
        if self.show_error or failing:
            raise tuner.SnapshotError("ollama show failed")
        return self.modelfiles.get(model_name, self.live_text)

    def ollama_bin(self):
        if self.ollama_missing:
            raise tuner.OllamaNotFoundError("can't find the ollama command")
        return "/fake/bin/ollama"


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    monkeypatch.setattr(tuner, "judge_ads", f.judge_ads)
    monkeypatch.setattr(tuner, "judge_dedup", f.judge_dedup)
    monkeypatch.setattr(tuner, "second_opinion_ad", f.second_ad)
    monkeypatch.setattr(tuner, "second_opinion_dedup", f.second_dedup)
    monkeypatch.setattr(tuner, "second_opinion_rule", f.second_rule)
    monkeypatch.setattr(tuner, "second_opinion_dedup_groups", f.second_dedup_groups_fake)

    def no_real_claude(*a, **kw):
        raise AssertionError("a test reached the real claude CLI")

    monkeypatch.setattr(judge, "run_claude", no_real_claude)

    def no_real_subprocess(*a, **kw):
        raise AssertionError("a test reached the real subprocess.run")

    monkeypatch.setattr(subprocess, "run", no_real_subprocess)
    monkeypatch.setattr(tuner, "ollama_bin", f.ollama_bin)
    monkeypatch.setattr(gating, "predict_ads", f.predict_ads)
    monkeypatch.setattr(gating, "predict_dedup", f.predict_dedup)
    monkeypatch.setattr(models, "install_ad_detector_model", f.install)
    monkeypatch.setattr(tuner, "remove_ollama_model", f.remove)
    monkeypatch.setattr(tuner, "installed_modelfile", f.show)
    return f


def store(cfg) -> TuningStore:
    return TuningStore(cfg.tuning.data_dir)


WRONG_MERGE = DedupCase(titles=["A launches X", "B raises money"], expected_groups=[],
                        kind="wrong_merge", note="different stories")


# ---------- the brief's tests ----------

def test_no_evidence_writes_short_report_and_applies_nothing(cfg):
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "no evidence" in out.report_path.read_text()


def test_judge_failure_applies_nothing_and_reports(cfg, evidence, fakes):
    fakes.judge_error = JudgeError("rate limited")
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "rate limited" in out.report_path.read_text()
    assert store(cfg).ad_examples() == []


def test_judge_failure_report_keeps_the_cost_of_calls_made(cfg, evidence, fakes):
    fakes.judge_error = JudgeError("ad batch 1 of 1: no usable verdict for 3 of 3 records", cost_usd=0.42)
    text = run_tune(cfg, DATE, log=lambda s: None).report_path.read_text()
    assert "no usable verdict" in text and "Judge cost: $0.42" in text


def test_unjudged_record_is_counted_and_gets_no_case(cfg, evidence, fakes, monkeypatch):
    fakes.unjudged_keys = {"k_ad_aaaaaaaa"}
    fakes.candidate_fixes_today_mistake()
    seen: list[str] = []
    real_predict = gating.predict_ads
    monkeypatch.setattr(gating, "predict_ads", lambda model, cases: seen.extend(c.id for c in cases)
                        or real_predict(model, cases))
    text = run_tune(cfg, DATE, log=lambda s: None).report_path.read_text()
    assert "Records skipped: 1" in text and "Ad-gate decisions judged: 2." in text
    assert "today:k_ad_aaa" not in seen and "today:k_fp_aaa" in seen


def test_accepted_change_snapshots_then_saves_and_rebuilds(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    assert out.ads_applied and s.latest_snapshot(DATE) is not None
    assert len(s.ad_examples()) >= 1 and fakes.installed[-1] == LIVE
    assert all(e["added"] == DATE for e in s.ad_examples())
    snap = s.latest_snapshot(DATE)
    assert not (snap / "learned_ad_examples.json").exists()  # taken before the save
    assert "AI Report issue 42" in fakes.modelfiles[LIVE]


def test_rejected_change_leaves_learned_data_alone(cfg, evidence, fakes):
    fakes.candidate_breaks_a_base_example()
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == []
    assert store(cfg).latest_snapshot(DATE) is None and LIVE not in fakes.installed
    assert "regression" in out.report_path.read_text()


def test_remove_more_example_needs_second_opinion(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.judge_proposes_ad_example()
    fakes.second_opinion_ad = False
    run_tune(cfg, DATE, log=lambda s: None)
    examples = store(cfg).ad_examples()
    assert examples and all(not e["is_ad"] for e in examples)
    assert "k_ok_aaaaaaaa" in fakes.second_ad_keys


def test_rebuild_failure_restores_previous_learned_data(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.fail_install_of = LIVE
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "restored" in out.report_path.read_text()
    assert store(cfg).ad_examples() == []


def test_failed_live_create_keeps_the_old_model_and_restores_files_only(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.fail_install_of = LIVE
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    snap = s.latest_snapshot(DATE)
    text = out.report_path.read_text()
    assert not out.ads_applied and fakes.installed.count(LIVE) == 1  # no rebuild from the snapshot
    assert f"{LIVE} was not replaced" in text and "Rebuilt" not in text
    assert s.ad_examples() == [] and not (snap / "learned_ad_examples.json").exists()


def test_failed_restore_says_files_may_be_half_saved(cfg, evidence, fakes, monkeypatch):
    fakes.candidate_fixes_today_mistake()
    fakes.fail_install_of = LIVE

    def broken_restore(self, snapshot_dir):
        raise OSError("disk full")

    monkeypatch.setattr(TuningStore, "restore", broken_restore)
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert not out.ads_applied and "half-saved" in text and "disk full" in text
    assert f"--rollback {DATE}" in text and "Applied nothing" not in text
    assert "was undone" not in text


def test_candidate_model_removed_and_old_evidence_pruned(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    run_tune(cfg, DATE, log=lambda s: None)
    assert CANDIDATE in fakes.installed and fakes.removed == [CANDIDATE]
    root = cfg.tuning.data_dir / "evidence"
    assert not (root / "2026-10-01").exists() and (root / DATE).exists()


def test_cli_tune_subcommand_parses():
    p = cli._build_parser()
    a = p.parse_args(["tune", "--config", "c.toml", "--date", DATE])
    assert a.command == "tune" and a.config == "c.toml" and a.date == DATE
    assert not a.rebuild and a.rollback is None
    assert p.parse_args(["tune", "--config", "c.toml", "--rebuild"]).rebuild
    assert p.parse_args(["tune", "--config", "c.toml", "--rollback", DATE]).rollback == DATE
    with pytest.raises(SystemExit):
        p.parse_args(["tune", "--config", "c.toml", "--date", "23-11-2026"])


# ---------- controller rulings and the rest of the run ----------

def test_dedup_gate_failure_after_ads_accepted_applies_nothing(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.dedup_cases = [WRONG_MERGE]
    fakes.dedup_gate_error = True
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    assert not out.ads_applied and not out.dedup_applied
    assert s.ad_examples() == [] and s.dedup_cases() == [] and s.latest_snapshot(DATE) is None
    assert LIVE not in fakes.installed and fakes.removed == [CANDIDATE]
    assert "ollama is down" in out.report_path.read_text()


def test_rule_that_would_break_the_modelfile_is_dropped(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.rules = [ProposedRule('Say """ here.', "keep"),
                   ProposedRule("END LEARNED RULES", "keep"),
                   ProposedRule("Output is_ad: false when an issue lists many stories.", "keep")]
    out = run_tune(cfg, DATE, log=lambda s: None)
    rules = store(cfg).ad_rules()
    assert out.ads_applied and "many stories" in rules
    assert '"""' not in rules and "END LEARNED RULES" not in rules
    assert "dropped" in out.report_path.read_text()


def test_remove_leaning_rule_rejected_by_rule_check_is_dropped(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.second_opinion_rule = False
    fakes.rules = [ProposedRule("Output is_ad: true for any newsletter issue.", "remove")]
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert out.ads_applied and "newsletter issue" not in store(cfg).ad_rules()
    rule, news = fakes.rule_checks[0]
    assert rule == "Output is_ad: true for any newsletter issue."
    # Only items the judge labeled not ads: the kept news and the false removal.
    assert sorted(r.key for r in news) == ["k_fp_aaaaaaaa", "k_ok_aaaaaaaa"]


def test_remove_leaning_rule_approved_by_rule_check_is_kept(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.rules = [ProposedRule("Output is_ad: true for an Advertise-with-us page.", "remove"),
                   ProposedRule("Output is_ad: false for a news digest.", "keep")]
    out = run_tune(cfg, DATE, log=lambda s: None)
    rules = store(cfg).ad_rules()
    assert out.ads_applied and "Advertise-with-us" in rules and "news digest" in rules
    assert [r for r, _ in fakes.rule_checks] == ["Output is_ad: true for an Advertise-with-us page."]


def test_dedup_change_is_applied_with_added_date(cfg, evidence, fakes):
    fakes.dedup_cases = [WRONG_MERGE]
    out = run_tune(cfg, DATE, log=lambda s: None)
    cases = store(cfg).dedup_cases()
    assert out.dedup_applied and not out.ads_applied
    assert cases == [{"titles": WRONG_MERGE.titles, "expected_groups": [], "kind": "wrong_merge",
                      "note": "different stories", "added": DATE}]
    assert store(cfg).latest_snapshot(DATE) is not None


def test_missed_duplicate_needs_second_opinion(cfg, evidence, fakes):
    fakes.dedup_cases = [DedupCase(titles=["A launches X", "A ships X"], expected_groups=[[0, 1]],
                                   kind="missed_duplicate", note="same story")]
    fakes.second_opinion_dedup = False
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.dedup_applied and store(cfg).dedup_cases() == []


def test_mistake_not_visible_in_excerpt_does_not_count(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.mistakes["k_fp_aaaaaaaa"] = (False, False)
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert not out.ads_applied and "needs a longer input" in text.lower()
    assert store(cfg).ad_examples() == [] and CANDIDATE not in fakes.installed
    assert "not learned" in text


def test_old_heuristic_records_skipped_and_removals_judged_first(cfg, evidence, fakes):
    cfg.tuning.max_items_per_run = 2
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert [r.verdict for r in fakes.judged] == ["ad", "ad"]
    assert "Decisions over the cap of items per run: 1." in out.report_path.read_text()


def test_report_lists_cost_skipped_and_commands(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    lines: list[str] = []
    out = run_tune(cfg, DATE, log=lines.append)
    text = out.report_path.read_text()
    assert "AI Report issue 42" in text and "judge reason" in text
    assert "$0.76" in text and "skipped: 2" in text  # 0.25 + 0.5 + one second opinion at 0.01
    assert f"newsradar tune --config {cfg.config_path} --rebuild" in text
    assert f"newsradar tune --config {cfg.config_path} --rollback {DATE}" in text
    assert len(lines) >= 1 and DATE in lines[-1]


def test_run_logs_each_step_in_order(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    lines: list[str] = []
    run_tune(cfg, DATE, log=lines.append)
    steps = ["loaded", "judging", "judging", "second opinions", "building and installing", "ads gate, ",
             "ads gate accepted", "applying", "cleaning up"]
    pos = -1
    for step in steps:
        pos = next((i for i in range(pos + 1, len(lines)) if step in lines[i]), -1)
        assert pos >= 0, f"missing step {step!r} in order: {lines}"
    assert DATE in lines[-1]


def test_run_logs_the_dedup_gate(cfg, evidence, fakes):
    fakes.dedup_cases = [WRONG_MERGE]
    lines: list[str] = []
    run_tune(cfg, DATE, log=lines.append)
    gate = next(i for i, s in enumerate(lines) if "running the dedup gate" in s)
    assert "dedup gate accepted" in lines[gate + 1]
    assert "applying" in lines[gate + 2]


def test_unexpected_exception_is_reported_not_raised(cfg, evidence, fakes):
    fakes.judge_error = KeyError("surprise")
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "surprise" in out.report_path.read_text()


def test_rebuild_live_model_uses_base_plus_learned(cfg, fakes):
    s = store(cfg)
    s.save_ad_examples([{"id": "x", "title": "Learned title", "summary": "s", "source": "S", "link": "",
                         "is_ad": False, "ad_type": "not_ad", "notes": "n", "added": DATE}])
    text = rebuild_live_model(cfg)
    assert fakes.installed == [LIVE] and "'Learned title'" in text
    assert fakes.modelfiles[LIVE] == text


def test_rollback_restores_snapshot_and_reinstalls_the_model_that_was_live(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    run_tune(cfg, DATE, log=lambda s: None)
    assert store(cfg).ad_examples()
    assert rollback(cfg, DATE) is True
    assert store(cfg).ad_examples() == [] and fakes.installed[-1] == LIVE
    assert fakes.modelfiles[LIVE] == fakes.live_text


def test_rollback_without_snapshot_raises(cfg, fakes):
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        rollback(cfg, DATE)


def test_rollback_of_an_ads_snapshot_without_modelfile_changes_nothing(cfg, fakes):
    s = store(cfg)
    s.save_ad_examples([{"id": "before"}])
    s.save_ad_rules("rule before\n")
    s.save_dedup_cases([{"titles": ["a", "b"], "expected_groups": []}])
    snap = s.snapshot(DATE, None, ["ads", "dedup"])  # damaged: an ads snapshot needs the Modelfile
    s.save_ad_examples([{"id": "after"}])
    s.save_ad_rules("rule after\n")
    s.save_dedup_cases([{"titles": ["c", "d"], "expected_groups": []}])
    with pytest.raises(FileNotFoundError, match="holds no Modelfile"):
        rollback(cfg, DATE)
    assert s.ad_examples() == [{"id": "after"}] and s.ad_rules() == "rule after\n"
    assert s.dedup_cases() == [{"titles": ["c", "d"], "expected_groups": []}]
    assert fakes.installed == [] and (snap / "learned_ad_examples.json").exists()


def test_rollback_of_an_ads_snapshot_with_a_blank_modelfile_changes_nothing(cfg, fakes):
    s = store(cfg)
    s.save_ad_examples([{"id": "before"}])
    s.snapshot(DATE, "   \n", ["ads"])
    s.save_ad_examples([{"id": "after"}])
    with pytest.raises(FileNotFoundError, match="holds no Modelfile"):
        rollback(cfg, DATE)
    assert s.ad_examples() == [{"id": "after"}] and fakes.installed == []


def test_rollback_without_ollama_changes_nothing(cfg, fakes):
    s = store(cfg)
    s.save_ad_examples([{"id": "before"}])
    s.snapshot(DATE, "FROM /blobs/old\n", ["ads"])
    s.save_ad_examples([{"id": "after"}])
    fakes.ollama_missing = True
    with pytest.raises(tuner.OllamaNotFoundError):
        rollback(cfg, DATE)
    assert s.ad_examples() == [{"id": "after"}] and fakes.installed == []


def test_dedup_only_rollback_needs_no_ollama(cfg, fakes):
    s = store(cfg)
    s.snapshot(DATE, None, ["dedup"])
    s.save_dedup_cases([{"titles": ["a", "b"], "expected_groups": []}])
    fakes.ollama_missing = True
    assert rollback(cfg, DATE) is False and s.dedup_cases() == []


def _cli_rollback(cfg, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["newsradar", "tune", "--config", str(cfg.config_path), "--rollback", DATE])
    with pytest.raises(SystemExit) as info:
        cli.main()
    return info.value.code


@pytest.mark.parametrize("text", ["{not json", '{"other": 1}'])
def test_cli_rollback_of_bad_applied_json_prints_one_line_and_exits_nonzero(cfg, fakes, monkeypatch, capsys, text):
    s = store(cfg)
    s.save_ad_examples([{"id": "now"}])
    snap = s.snapshot(DATE, "FROM /blobs/old\n", ["ads"])
    (snap / "applied.json").write_text(text)
    assert _cli_rollback(cfg, monkeypatch) not in (0, None)
    err = capsys.readouterr().err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1 and "applied.json" in err
    assert s.ad_examples() == [{"id": "now"}] and fakes.installed == []


def test_cli_rollback_of_an_ads_snapshot_without_modelfile_exits_nonzero(cfg, fakes, monkeypatch, capsys):
    store(cfg).snapshot(DATE, None, ["ads"])
    assert _cli_rollback(cfg, monkeypatch) not in (0, None)
    err = capsys.readouterr().err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1 and "holds no Modelfile" in err


def test_cli_tune_runs_with_today_by_default(cfg, monkeypatch):
    seen = {}
    monkeypatch.setattr(tuner, "run_tune", lambda config, date, log=print, **kw: seen.update(date=date, **kw))
    monkeypatch.setattr(sys, "argv", ["newsradar", "tune", "--config", str(cfg.config_path)])
    cli.main()
    assert len(seen["date"]) == 10 and seen["date"][4] == "-"



def test_build_files_removed_after_every_run(cfg, evidence, fakes):
    build = store(cfg).root / "build"
    fakes.candidate_fixes_today_mistake()
    fakes.fail_install_of = CANDIDATE
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and CANDIDATE in fakes.installed and not build.exists()
    fakes.fail_install_of = None
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert out.ads_applied and fakes.installed[-1] == LIVE and not build.exists()


def test_run_prunes_old_snapshots_and_reports(cfg, fakes):
    s = store(cfg)
    for name in ("2026-09-01", "2026-10-01"):
        (s.root / "snapshots" / name).mkdir(parents=True)
    old_report = s.report_path("2026-10-01")
    old_report.parent.mkdir(parents=True)
    old_report.write_text("r")
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not (s.root / "snapshots" / "2026-09-01").exists()
    assert (s.root / "snapshots" / "2026-10-01").exists()
    assert not old_report.exists() and out.report_path.exists()


# ---------- Task 12: second opinion on blockers ----------

NEWS_CASE = "today:k_ok_aaa"  # "OpenAI ships a model", judged not an ad


def test_today_blocker_disputed_by_second_opinion_is_dropped_and_gate_accepts(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {NEWS_CASE}
    fakes.second_ad_answers = {"k_ok_aaaaaaaa": True}  # the second opinion says it is an ad
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert out.ads_applied and "k_ok_aaaaaaaa" in fakes.second_ad_keys
    assert f"{NEWS_CASE} (OpenAI ships a model): disputed, not counted" in text


def test_today_blocker_confirmed_by_second_opinion_still_rejects(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {NEWS_CASE}
    fakes.second_ad_answers = {"k_ok_aaaaaaaa": False}  # agrees: not an ad
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == []
    assert "supports the first judge's label, so it still blocks" in out.report_path.read_text()


def test_today_blocker_with_unusable_second_opinion_still_rejects(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {NEWS_CASE}
    fakes.second_ad_answers = {"k_ok_aaaaaaaa": None}
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "no usable second opinion" in out.report_path.read_text()


def test_base_blocker_rejects_without_a_second_opinion(cfg, evidence, fakes):
    fakes.candidate_breaks_a_base_example()
    fakes.candidate_break_ids = {NEWS_CASE}
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied
    assert fakes.second_ad_keys == ["k_ad_aaaaaaaa"]  # only the balancing example's check


def test_dedup_blocker_disputed_is_dropped_from_gate_and_saved_cases(cfg, evidence, fakes):
    miss = DedupCase(titles=["Q one", "Q two"], expected_groups=[[0, 1]], kind="missed_duplicate", note="")
    fakes.dedup_cases = [WRONG_MERGE, miss]
    fakes.second_dedup_answers = [True]  # agrees when planned
    fakes.second_dedup_groups = [[]]  # as a blocker: no duplicates, so the candidate's empty drop is right
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert out.dedup_applied
    assert [c["titles"] for c in store(cfg).dedup_cases()] == [WRONG_MERGE.titles]
    assert "dedup:1 (Q one): disputed, not counted" in text and "gated the change again" in text


def test_dedup_blocker_with_an_unrelated_group_mismatch_still_blocks(cfg, evidence, fakes):
    miss = DedupCase(titles=["Q one", "Q two", "R one", "R two"], expected_groups=[[0, 1]],
                     kind="missed_duplicate", note="")
    fakes.dedup_cases = [WRONG_MERGE, miss]
    fakes.second_dedup_answers = [True]
    # The second opinion also pairs R one and R two, which differs from the case, but it
    # still groups Q one with Q two, so the candidate's empty drop is wrong either way.
    fakes.second_dedup_groups = [[[0, 1], [2, 3]]]
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.dedup_applied and store(cfg).dedup_cases() == []
    assert "supports the first judge's label, so it still blocks" in out.report_path.read_text()


def test_second_opinion_call_failure_on_a_blocker_applies_nothing(cfg, evidence, fakes, monkeypatch):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {NEWS_CASE}

    def fail(record, tuning):
        if record.key == "k_ok_aaaaaaaa":
            raise JudgeError("rate limited")
        return True, 0.0

    monkeypatch.setattr(tuner, "second_opinion_ad", fail)
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and "rate limited" in out.report_path.read_text()


# ---------- Task 12: the seed goes through the gate ----------

def write_seed(tmp_path, examples=None, rules=None):
    seed = tmp_path / "seed"
    seed.mkdir()
    if examples is not None:
        (seed / "learned_ad_examples.json").write_text(json.dumps(examples))
    if rules is not None:
        (seed / "learned_ad_rules.txt").write_text(rules)
    return seed


SEED_EXAMPLES = [
    {"id": "seed-1", "title": "AI Report issue as web copy", "summary": "s" * 300,
     "source": "theaireport.ai", "link": "https://www.theaireport.ai/p/x?ref=abc", "is_ad": False,
     "ad_type": "not_ad", "notes": "news issue", "added": "2026-11-20"},
    {"id": "seed-2", "title": "Tech Buzz digest", "summary": "digest", "source": "The Tech Buzz",
     "link": "", "is_ad": False, "ad_type": "not_ad", "notes": "digest"},
]


def test_seed_is_included_and_saved_on_accept(cfg, evidence, fakes, tmp_path):
    cfg.tuning.max_new_examples_per_day = 1
    fakes.candidate_fixes_today_mistake()
    seed = write_seed(tmp_path, SEED_EXAMPLES, "Judge the whole item.\n")
    out = run_tune(cfg, DATE, log=lambda s: None, seed_dir=seed)
    by_id = {e["id"]: e for e in store(cfg).ad_examples()}
    assert out.ads_applied and {"seed-1", "seed-2"} <= set(by_id)
    assert any(i.startswith("the-ai-report") for i in by_id)  # the judge's example, despite the cap of 1
    assert by_id["seed-1"]["added"] == "2026-11-20" and by_id["seed-2"]["added"] == DATE
    assert len(by_id["seed-1"]["summary"]) == 200 and by_id["seed-1"]["link"] == "https://www.theaireport.ai/"
    assert "Judge the whole item." in store(cfg).ad_rules()
    assert "seed:seed-1" in fakes.ad_case_ids
    text = out.report_path.read_text()
    assert "## Seed" in text and "Saved with the rest" in text


def test_seed_is_not_saved_on_reject(cfg, evidence, fakes, tmp_path):
    fakes.candidate_breaks_a_base_example()
    seed = write_seed(tmp_path, SEED_EXAMPLES, "Judge the whole item.\n")
    out = run_tune(cfg, DATE, log=lambda s: None, seed_dir=seed)
    assert not out.ads_applied and store(cfg).ad_examples() == [] and store(cfg).ad_rules() == ""
    assert "Not saved: the ads change was rejected" in out.report_path.read_text()


def test_seed_rule_with_triple_quotes_is_dropped(cfg, evidence, fakes, tmp_path):
    fakes.candidate_fixes_today_mistake()
    seed = write_seed(tmp_path, rules='Bad """ rule.\nJudge the whole item.\n')
    out = run_tune(cfg, DATE, log=lambda s: None, seed_dir=seed)
    rules = store(cfg).ad_rules()
    assert out.ads_applied and "Judge the whole item." in rules and '"""' not in rules
    seed_section = out.report_path.read_text().split("## Seed")[1].split("## ")[0]
    assert "Bad" in seed_section and "dropped" in seed_section


def test_seed_rule_saying_is_ad_true_goes_through_the_rule_check(cfg, evidence, fakes, tmp_path):
    fakes.candidate_fixes_today_mistake()
    fakes.second_opinion_rule = False
    seed = write_seed(tmp_path, rules="Output is_ad: true only when the whole item is a pitch.\n")
    run_tune(cfg, DATE, log=lambda s: None, seed_dir=seed)
    assert [r for r, _ in fakes.rule_checks] == ["Output is_ad: true only when the whole item is a pitch."]
    assert store(cfg).ad_rules() == ""


def test_missing_seed_folder_applies_nothing(cfg, evidence, fakes, tmp_path):
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None, seed_dir=tmp_path / "nope")
    assert not out.ads_applied and "does not exist" in out.report_path.read_text()


def test_cli_parses_include_seed(cfg, monkeypatch):
    p = cli._build_parser()
    a = p.parse_args(["tune", "--config", "c.toml", "--include-seed", "data/tuning/seed"])
    assert a.include_seed == Path("data/tuning/seed")
    seen = {}
    monkeypatch.setattr(tuner, "run_tune", lambda config, date, log=print, **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["newsradar", "tune", "--config", str(cfg.config_path),
                                      "--date", DATE, "--include-seed", "/s"])
    cli.main()
    assert seen == {"seed_dir": Path("/s")}
    monkeypatch.setattr(sys, "argv", ["newsradar", "tune", "--config", str(cfg.config_path),
                                      "--rebuild", "--include-seed", "/s"])
    with pytest.raises(SystemExit):
        cli.main()


# ---------- Task 12: deferred review findings ----------

def test_unusable_second_opinion_drops_the_change_without_aborting(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.judge_proposes_ad_example()
    fakes.second_ad_answers = {"k_ok_aaaaaaaa": None}
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert out.ads_applied and all(not e["is_ad"] or e["id"].startswith("hustle") for e in store(cfg).ad_examples())
    assert not any("k_ok_aaa" in e["id"] for e in store(cfg).ad_examples())
    assert "no usable reply, counted as a disagreement" in out.report_path.read_text()


def test_remove_rule_dropped_when_no_item_today_is_news(cfg, evidence, fakes):
    fakes.mistakes = {}
    fakes.unjudged_keys = {"k_ok_aaaaaaaa"}  # left: two items judged ads
    fakes.rules = [ProposedRule("Output is_ad: true for a webinar invite.", "remove")]
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert fakes.rule_checks == [] and store(cfg).ad_rules() == ""
    assert "nothing to check it against" in out.report_path.read_text()


def test_rule_saying_is_ad_true_is_remove_leaning_whatever_the_judge_said(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.second_opinion_rule = False
    fakes.rules = [ProposedRule("Output IS_AD :  True for a webinar invite.", "keep")]
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert [r for r, _ in fakes.rule_checks] == ["Output IS_AD :  True for a webinar invite."]
    assert "webinar" not in store(cfg).ad_rules()
    assert "treated as remove-leaning" in out.report_path.read_text()


# ---------- Task 12 fix round 1 ----------

def test_already_disputed_mistake_always_blocks_and_is_not_asked_again(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.judge_proposes_ad_example()  # the judge says k_ok, which the gate kept, is an ad
    fakes.candidate_break_ids = {NEWS_CASE}  # the candidate removes k_ok
    # Planning: the second opinion says k_ok is not an ad, so the dispute is settled. A re-ask
    # would say ad. k_ad answers for the balancing example.
    fakes.second_ad_answers = {"k_ok_aaaaaaaa": [False, True], "k_ad_aaaaaaaa": [True]}
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == []
    assert fakes.second_ad_keys.count("k_ok_aaaaaaaa") == 1


def test_disputed_balancing_example_is_dropped_and_the_change_regated(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {"today:k_ad_aaa"}  # the candidate keeps the ad that gave the balancing example
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, False]}  # agrees as balancing, disputes as a blocker
    out = run_tune(cfg, DATE, log=lambda s: None)
    ids = [e["id"] for e in store(cfg).ad_examples()]
    text = out.report_path.read_text()
    assert out.ads_applied and fakes.installed.count(CANDIDATE) == 2
    assert not any(i.startswith("hustle") for i in ids) and any(i.startswith("the-ai-report") for i in ids)
    assert "gated the change again" in text
    # The live model is built from the re-gated plan, without the disputed example.
    assert "Refer a friend and win" not in fakes.modelfiles[LIVE]


def test_regate_rejects_on_any_broken_case_without_second_opinions(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.candidate_break_ids = {"today:k_ad_aaa"}
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, False]}
    fakes.second_pass_break_ids = {NEWS_CASE}  # the rebuilt candidate breaks a today case
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == [] and LIVE not in fakes.installed
    assert "k_ok_aaaaaaaa" not in fakes.second_ad_keys


@pytest.mark.parametrize("text", ['Return {"is_ad": true} for a webinar invite.', "Set is_ad=true for promos.",
                                  "Output is_ad : TRUE when it pitches."])
def test_rule_writing_is_ad_true_any_way_is_remove_leaning(cfg, evidence, fakes, text):
    fakes.candidate_fixes_today_mistake()
    fakes.second_opinion_rule = False
    fakes.rules = [ProposedRule(text, "keep")]
    run_tune(cfg, DATE, log=lambda s: None)
    assert [r for r, _ in fakes.rule_checks] == [text] and store(cfg).ad_rules() == ""


def test_report_counts_shifted_verdicts_separately(cfg, evidence, fakes, monkeypatch):
    real = fakes.judge_ads

    def with_shift(records, tuning):
        out = real(records, tuning)
        out.verdicts = out.verdicts[:-1]
        out.skipped, out.shifted = 2, 1
        return out

    monkeypatch.setattr(tuner, "judge_ads", with_shift)
    text = run_tune(cfg, DATE, log=lambda s: None).report_path.read_text()
    assert "Records skipped: 2 (no usable verdict: 1; label didn't match the gate's record: 1)" in text


def test_every_seed_rule_gets_the_rule_second_opinion(cfg, evidence, fakes, tmp_path):
    fakes.candidate_fixes_today_mistake()
    seed = write_seed(tmp_path, rules="Judge the whole item.\nKeep newsletter issues.\n")
    run_tune(cfg, DATE, log=lambda s: None, seed_dir=seed)
    assert [r for r, _ in fakes.rule_checks] == ["Judge the whole item.", "Keep newsletter issues."]


# ---------- final review fixes ----------

def test_snapshot_stores_the_live_modelfile_from_ollama_show(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    run_tune(cfg, DATE, log=lambda s: None)
    snap = store(cfg).latest_snapshot(DATE)
    assert fakes.shown == [LIVE, LIVE] and (snap / "Modelfile").read_text() == fakes.live_text  # preflight, snapshot
    assert json.loads((snap / "applied.json").read_text()) == {"parts": ["ads"]}


def test_failed_ollama_show_applies_nothing(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.dedup_cases = [WRONG_MERGE]
    fakes.show_fail_from = 1  # the preflight read works; the snapshot's read fails
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    assert not out.ads_applied and not out.dedup_applied
    assert s.ad_examples() == [] and s.dedup_cases() == [] and s.latest_snapshot(DATE) is None
    assert LIVE not in fakes.installed
    assert "applied nothing" in out.report_path.read_text() and "ollama show failed" in out.report_path.read_text()


def test_installed_modelfile_runs_ollama_show(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "FROM /blobs/x\nSYSTEM hi\n", "")

    monkeypatch.setattr(models.subprocess, "run", fake_run)
    monkeypatch.setattr(models, "ollama_bin", lambda: "/usr/local/bin/ollama")
    assert tuner.installed_modelfile(LIVE) == "FROM /blobs/x\nSYSTEM hi\n"
    assert calls == [["/usr/local/bin/ollama", "show", "--modelfile", LIVE]]


def test_installed_modelfile_without_ollama_raises_snapshot_error(monkeypatch):
    def missing():
        raise tuner.OllamaNotFoundError("can't find the ollama command")

    monkeypatch.setattr(models, "ollama_bin", missing)
    with pytest.raises(tuner.SnapshotError, match="can't find the ollama command"):
        tuner.installed_modelfile(LIVE)


def test_remove_ollama_model_runs_the_resolved_binary(monkeypatch):
    calls = []
    monkeypatch.setattr(models.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(models, "ollama_bin", lambda: "/opt/homebrew/bin/ollama")
    tuner.remove_ollama_model(CANDIDATE)
    assert calls == [["/opt/homebrew/bin/ollama", "rm", CANDIDATE]]


def test_remove_ollama_model_ignores_a_missing_ollama(monkeypatch):
    def missing():
        raise tuner.OllamaNotFoundError("can't find the ollama command")

    def no_run(*a, **kw):
        raise AssertionError("subprocess.run called")

    monkeypatch.setattr(models, "ollama_bin", missing)
    monkeypatch.setattr(models.subprocess, "run", no_run)
    tuner.remove_ollama_model(CANDIDATE)  # no raise


# ---------- preflight: ollama and the live Modelfile, before any paid call ----------

class _NoJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        raise AssertionError("the preflight should have stopped the run before any judge call")


@pytest.mark.parametrize("failure", ["no_ollama", "no_modelfile"])
def test_preflight_failure_makes_no_judge_calls_and_applies_nothing(cfg, evidence, fakes, monkeypatch, failure):
    no_ads, no_dedup = _NoJudge(), _NoJudge()
    monkeypatch.setattr(tuner, "judge_ads", no_ads)
    monkeypatch.setattr(tuner, "judge_dedup", no_dedup)
    if failure == "no_ollama":
        fakes.ollama_missing = True
    else:
        fakes.show_error = True
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert no_ads.calls == 0 and no_dedup.calls == 0 and fakes.second_ad_keys == []
    assert not out.ads_applied and not out.dedup_applied and fakes.installed == []
    assert store(cfg).latest_snapshot(DATE) is None and store(cfg).ad_examples() == []
    assert "Preflight failed" in text and "no judge calls" in text
    assert ("can't find the ollama command" if failure == "no_ollama" else "ollama show failed") in text
    assert "$0.00" in text


def test_preflight_reads_the_live_modelfile_before_judging(cfg, evidence, fakes, monkeypatch):
    order = []
    real_show, real_judge = fakes.show, fakes.judge_ads
    monkeypatch.setattr(tuner, "installed_modelfile", lambda name: (order.append("show"), real_show(name))[1])
    monkeypatch.setattr(tuner, "judge_ads",
                        lambda records, tuning: (order.append("judge"), real_judge(records, tuning))[1])
    run_tune(cfg, DATE, log=lambda s: None)
    assert order[:2] == ["show", "judge"]


@pytest.mark.parametrize("outcome,message", [
    ("exit", "exited 1: model not found"), ("empty", "printed nothing"), ("missing", "failed: ollama gone"),
])
def test_installed_modelfile_failure_raises_snapshot_error(monkeypatch, outcome, message):
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd[0])
        if outcome == "missing":
            raise FileNotFoundError("ollama gone")
        if outcome == "exit":
            raise subprocess.CalledProcessError(1, cmd, "", "model not found")
        return subprocess.CompletedProcess(cmd, 0, "  \n", "")

    monkeypatch.setattr(models, "ollama_bin", lambda: "/fake/ollama")
    monkeypatch.setattr(models.subprocess, "run", fake_run)
    with pytest.raises(tuner.SnapshotError, match=message):
        tuner.installed_modelfile(LIVE)
    assert ran == ["/fake/ollama"]  # the branch under test ran, not a lookup failure


def test_undo_with_rebuild_reinstalls_the_stored_live_text(cfg, fakes):
    s = store(cfg)
    snap = s.snapshot(DATE, "FROM /blobs/old\n", ["ads"])
    s.save_ad_examples([{"id": "x"}])
    rep = tuner._Report(date=DATE, config_path=cfg.config_path)
    t = tuner._Tuner(cfg, DATE, rep, tuner.TuneOutcome(False, False, "", s.report_path(DATE)))
    t._undo(snap, rebuild=True)
    assert s.ad_examples() == [] and fakes.installed == [LIVE] and fakes.modelfiles[LIVE] == "FROM /blobs/old\n"


def test_dedup_only_day_snapshots_without_ollama_and_rollback_leaves_the_ad_model(cfg, evidence, fakes):
    fakes.dedup_cases = [WRONG_MERGE]
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    snap = s.latest_snapshot(DATE)
    assert out.dedup_applied and fakes.shown == [LIVE] and not (snap / "Modelfile").exists()  # preflight only
    assert json.loads((snap / "applied.json").read_text()) == {"parts": ["dedup"]}
    later = [{"id": "later", "title": "t", "summary": "s", "is_ad": False, "ad_type": "not_ad"}]
    s.save_ad_examples(later)  # an ads change made after that day
    assert rollback(cfg, DATE) is False
    assert s.dedup_cases() == [] and s.ad_examples() == later and LIVE not in fakes.installed


def test_ads_and_dedup_day_rollback_restores_both_and_reinstalls(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.dedup_cases = [WRONG_MERGE]
    out = run_tune(cfg, DATE, log=lambda s: None)
    snap = store(cfg).latest_snapshot(DATE)
    assert out.ads_applied and out.dedup_applied
    assert json.loads((snap / "applied.json").read_text()) == {"parts": ["ads", "dedup"]}
    assert rollback(cfg, DATE) is True
    assert store(cfg).ad_examples() == [] and store(cfg).dedup_cases() == []
    assert fakes.modelfiles[LIVE] == fakes.live_text


def test_rollback_of_a_snapshot_without_applied_file_restores_all_and_reinstalls(cfg, fakes):
    s = store(cfg)
    s.save_dedup_cases([{"titles": ["a", "b"], "expected_groups": []}])
    snap = s.root / "snapshots" / DATE
    snap.mkdir(parents=True)
    (snap / "Modelfile").write_text("FROM /blobs/legacy\n")
    assert rollback(cfg, DATE) is True
    assert s.dedup_cases() == [] and fakes.modelfiles[LIVE] == "FROM /blobs/legacy\n"


def test_rule_with_backticked_is_ad_true_is_remove_leaning(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.second_opinion_rule = False
    text = "Output `is_ad`: true when the whole item is a pitch."
    fakes.rules = [ProposedRule(text, "keep")]
    run_tune(cfg, DATE, log=lambda s: None)
    assert [r for r, _ in fakes.rule_checks] == [text] and store(cfg).ad_rules() == ""


def test_wrong_merge_that_teaches_a_merge_needs_second_opinion(cfg, evidence, fakes):
    case = DedupCase(titles=["A launches X", "A ships X", "B raises money"], expected_groups=[[0, 1]],
                     kind="wrong_merge", note="merged the wrong pair")
    fakes.dedup_cases = [case]
    fakes.second_opinion_dedup = False
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.dedup_applied and store(cfg).dedup_cases() == []
    assert "Second opinion: disagreed" in out.report_path.read_text()


def _missed(n):
    return [DedupCase(titles=[f"Q{i} one", f"Q{i} two"], expected_groups=[[0, 1]], kind="missed_duplicate",
                      note="") for i in range(n)]


def test_more_than_two_blockers_reject_without_second_opinions(cfg, evidence, fakes):
    fakes.dedup_cases = _missed(3)  # the candidate breaks all three
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.dedup_applied and fakes.second_dedup_group_calls == 0
    assert "3 cases broke, more than 2" in out.report_path.read_text()


def test_two_blockers_still_get_second_opinions(cfg, evidence, fakes):
    fakes.dedup_cases = _missed(2)
    fakes.second_dedup_groups = [[], []]
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert fakes.second_dedup_group_calls == 2 and not out.dedup_applied  # nothing new is left


def test_report_warns_when_the_live_model_was_not_built_by_the_tuner(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    first = run_tune(cfg, DATE, log=lambda s: None).report_path.read_text()
    assert "not built by the tuner" in first and "num_ctx" in first
    second = run_tune(cfg, DATE, log=lambda s: None).report_path.read_text()
    assert "not built by the tuner" not in second


@pytest.mark.parametrize("configured,live,candidate", [
    ("ollama/ad-detector", "ad-detector", "ad-detector-candidate"),
    ("ollama/ad-detector:latest", "ad-detector:latest", "ad-detector-candidate"),
    ("ad-detector:v2", "ad-detector:v2", "ad-detector-candidate"),
    ("ollama/me/ad-detector:latest", "me/ad-detector:latest", "me/ad-detector-candidate"),
])
def test_candidate_name_drops_the_tag(cfg, configured, live, candidate):
    cfg.models.ad_detector_model = configured
    assert tuner.live_model_name(cfg) == live and tuner.candidate_model_name(cfg) == candidate


# ---------- D5: kept ads on today's cases ----------

KEPT_AD = "today:k_ad_aaa"  # "Refer a friend and win", judged an ad
BASE_IDS = [f"base:{e['id']}" for e in models.load_examples(models.BASE_EXAMPLES)]


def test_kept_today_ad_passes_when_fixes_outnumber_it_five_to_one(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:4])  # 4 base fixes plus today's mistake
    fakes.candidate_break_ids = {KEPT_AD}
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert out.ads_applied and fakes.second_ad_keys == ["k_ad_aaaaaaaa"]  # only the balancing check
    assert f"with 1 kept ad: {KEPT_AD} (Refer a friend and win)" in text
    assert "Kept ads (allowed: fixes outside the seed outnumber them 5 to 1 or more)" in text
    assert f"- Kept ad: {KEPT_AD} (Refer a friend and win)" in text
    assert f"- Broken: {KEPT_AD}" not in text


def test_kept_today_ad_rejects_with_fewer_than_five_fixes_per_kept_ad(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:3])
    fakes.candidate_break_ids = {KEPT_AD}
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, True]}  # balancing, then the blocker: an ad
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == []
    assert "Kept ads" not in out.report_path.read_text()


def test_kept_today_ad_with_a_confirmed_removal_rejects(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:4])
    fakes.candidate_break_ids = {KEPT_AD, NEWS_CASE}
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, True], "k_ok_aaaaaaaa": False}  # both labels confirmed
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and store(cfg).ad_examples() == []
    assert "k_ok_aaaaaaaa" in fakes.second_ad_keys


def test_kept_today_ad_passes_once_the_removal_is_disputed(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:4])
    fakes.candidate_break_ids = {KEPT_AD, NEWS_CASE}
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, True], "k_ok_aaaaaaaa": True}  # k_ok is an ad
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert out.ads_applied and f"- Kept ad: {KEPT_AD}" in out.report_path.read_text()


def test_kept_ad_on_a_base_case_rejects_whatever_the_fixes(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:6])
    fakes.candidate_break_ids = {BASE_IDS[7]}  # a base ad the candidate now keeps
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and fakes.second_ad_keys == ["k_ad_aaaaaaaa"]


def test_second_gate_pass_allows_a_kept_today_ad(cfg, evidence, fakes):
    extra = ad_rec("k_ad2aaaaaaa", "ad", "Webinar: grow your pipeline", source="Vendor")
    with (evidence / "AI.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "ad", **asdict(extra)}) + "\n")
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:4])
    fakes.candidate_break_ids = {KEPT_AD, NEWS_CASE}
    # Balancing agrees k_ad is an ad; as blockers, both are disputed, so the balancing example drops.
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, False], "k_ok_aaaaaaaa": True}
    fakes.second_pass_break_ids = {"today:k_ad2aaa"}  # the rebuilt candidate keeps another ad
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert out.ads_applied and fakes.installed.count(CANDIDATE) == 2 and "gated the change again" in text
    assert len(fakes.second_ad_keys) == 3  # no second opinions on the second pass
    assert "- Kept ad: today:k_ad2aaa (Webinar: grow your pipeline)" in text


def test_second_gate_pass_rejects_a_kept_ad_under_five_to_one(cfg, evidence, fakes):
    extra = ad_rec("k_ad2aaaaaaa", "ad", "Webinar: grow your pipeline", source="Vendor")
    with (evidence / "AI.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "ad", **asdict(extra)}) + "\n")
    fakes.candidate_fixes_today_mistake()
    fakes.live_wrong_ids = set(BASE_IDS[:3])
    fakes.candidate_break_ids = {KEPT_AD}
    fakes.second_ad_answers = {"k_ad_aaaaaaaa": [True, False]}
    fakes.second_pass_break_ids = {"today:k_ad2aaa"}
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert not out.ads_applied and fakes.installed.count(CANDIDATE) == 2 and LIVE not in fakes.installed


# ---------- the gate's case set stays bounded ----------

def _learned(i, is_ad, added):
    return {"id": f"l{i}", "title": f"learned {i}", "summary": f"summary {i}", "source": "s",
            "link": "https://x.com/", "is_ad": is_ad, "ad_type": "sponsor_content" if is_ad else "not_ad",
            "notes": "", "added": added}


def test_gate_scores_every_learned_example_even_ones_trimmed_from_the_modelfile(cfg, evidence, fakes):
    cfg.tuning.learned_examples_per_class = 2
    store(cfg).save_ad_examples([_learned(i, i % 2 == 0, f"2026-11-{10 + i:02d}") for i in range(8)])
    fakes.candidate_fixes_today_mistake()
    run_tune(cfg, DATE, log=lambda s: None)
    gated = {c for c in fakes.ad_case_ids if c.startswith("learned:")}
    assert gated == {f"learned:l{i}" for i in range(8)}


def _passed(n):
    return [ad_rec(f"k_n{i:02d}aaaaaa", "not_ad", f"News story {i}", source="Wire") for i in range(n)]


def _add_records(evidence, records):
    with (evidence / "AI.jsonl").open("a") as f:
        for r in records:
            f.write(json.dumps({"kind": "ad", **asdict(r)}) + "\n")


ALWAYS = {"today:k_ok_aaa", "today:k_fp_aaa", "today:k_ad_aaa"}  # a missed ad, a false removal, a removal


def test_gate_samples_passed_records_up_to_the_cap_and_keeps_mistakes_and_removals(cfg, evidence, fakes):
    _add_records(evidence, _passed(30))
    cfg.tuning.max_gate_today_cases = 25  # 3 always scored, so 22 of the 30 passed records
    fakes.judge_proposes_ad_example()  # k_ok, which the gate kept, is now a missed ad
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    gated = {c for c in fakes.ad_case_ids if c.startswith("today:")}
    sampled = gated - ALWAYS
    assert ALWAYS <= gated and len(sampled) == 22 and all(c.startswith("today:k_n") for c in sampled)
    assert "8 of today's 30 passed records were sampled out" in out.report_path.read_text()

    fakes.ad_case_ids.clear()
    shutil.rmtree(store(cfg).root)  # forget the first run's examples, so the rerun gates again
    run_tune(cfg, DATE, log=lambda s: None)
    assert {c for c in fakes.ad_case_ids if c.startswith("today:k_n")} == sampled  # same sample every run


def test_gate_sample_is_seeded_by_date():
    keys = [f"k_n{i:02d}aaaaaa" for i in range(40)]
    a = gating._sample_passed(keys, 5, "2026-11-23")
    assert a == gating._sample_passed(list(reversed(keys)), 5, "2026-11-23") and len(a) == 5
    assert a != gating._sample_passed(keys, 5, "2026-11-24")


def test_mistakes_and_removals_are_scored_even_past_the_cap_with_all_of_a_few_passed(cfg, evidence, fakes):
    _add_records(evidence, _passed(5))
    cfg.tuning.max_gate_today_cases = 1  # fewer than the 3 that are always scored
    fakes.judge_proposes_ad_example()
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    gated = {c for c in fakes.ad_case_ids if c.startswith("today:")}
    assert ALWAYS <= gated and len(gated - ALWAYS) == 5  # min(20, 5) passed records still scored
    assert "sampled out" not in out.report_path.read_text()


def test_a_full_cap_still_scores_twenty_passed_records(cfg, evidence, fakes):
    _add_records(evidence, _passed(26))
    cfg.tuning.max_gate_today_cases = 2  # the 3 always-scored records already fill it
    fakes.judge_proposes_ad_example()
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    gated = {c for c in fakes.ad_case_ids if c.startswith("today:")}
    assert ALWAYS <= gated and len(gated - ALWAYS) == 20
    assert "6 of today's 26 passed records were sampled out" in out.report_path.read_text()


def test_gate_scores_the_record_behind_a_balancing_example_past_the_cap(cfg, evidence, fakes, monkeypatch):
    _add_records(evidence, _passed(5))
    cfg.tuning.max_gate_today_cases = 0
    monkeypatch.setattr(gating, "MIN_PASSED_SCORED", 0)  # so only the always-scored rule keeps it
    fakes.judge_proposes_ad_example()  # one ad example, so the day balances with a passed news item
    fakes.mistakes.pop("k_fp_aaaaaaaa")
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    balancing = [line for line in out.report_path.read_text().splitlines() if "Balancing example" in line]
    assert len(balancing) == 1 and '"News story 0" (not ad' in balancing[0]
    assert "today:k_n00aaa" in fakes.ad_case_ids


def test_gate_under_the_cap_scores_every_today_record(cfg, evidence, fakes):
    _add_records(evidence, _passed(4))
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, DATE, log=lambda s: None)
    assert len({c for c in fakes.ad_case_ids if c.startswith("today:")}) == 7
    assert "sampled out" not in out.report_path.read_text()


def test_candidate_over_the_context_budget_applies_nothing(cfg, evidence, fakes, monkeypatch):
    monkeypatch.setattr(models, "CONTEXT_SHARE", 0.01)
    fakes.candidate_fixes_today_mistake()
    fakes.dedup_cases = [WRONG_MERGE]
    out = run_tune(cfg, DATE, log=lambda s: None)
    text = out.report_path.read_text()
    assert not out.ads_applied and not out.dedup_applied and fakes.installed == []
    assert ("The candidate Modelfile is too large for num_ctx, so the run applied nothing: "
            "ModelfileTooLargeError:") in text and "A call failed" not in text
    assert "too large for num_ctx" in out.summary


def test_rebuild_over_the_context_budget_installs_nothing(cfg, fakes, monkeypatch):
    monkeypatch.setattr(models, "CONTEXT_SHARE", 0.01)
    with pytest.raises(tuner.ModelfileTooLargeError):
        rebuild_live_model(cfg)
    assert fakes.installed == []


def test_cli_rollback_whose_reinstall_fails_says_the_files_were_restored(cfg, fakes, monkeypatch, capsys):
    s = store(cfg)
    s.save_ad_examples([{"id": "before"}])
    s.snapshot(DATE, "FROM /blobs/old\n", ["ads"])
    s.save_ad_examples([{"id": "after"}])
    fakes.fail_install_of = LIVE
    assert _cli_rollback(cfg, monkeypatch) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert "restored the learned files" in err and f"{LIVE} was not reinstalled" in err
    assert "--rebuild" in err and "try the rollback again" in err
    assert s.ad_examples() == [{"id": "before"}]


def test_cli_rollback_without_ollama_exits_one_with_one_line(cfg, fakes, monkeypatch, capsys):
    store(cfg).snapshot(DATE, "FROM /blobs/old\n", ["ads"])
    fakes.ollama_missing = True
    assert _cli_rollback(cfg, monkeypatch) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1 and "ollama" in err
