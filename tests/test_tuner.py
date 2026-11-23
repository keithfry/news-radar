"""Tests for `newsradar tune`. Every judge, gate, Ollama build and `ollama rm`
call is replaced with a fake in newsradar.tuning.tuner; nothing real runs."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from newsradar import cli
from newsradar.config import load_config
from newsradar.tuning import judge, tuner
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
    ad_rec("k_h_aaaaaaaaa", "heuristic", "Sponsored: buy now"),
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
        self.candidate_fix = False
        self.candidate_break_base = False
        self.dedup_gate_error = False
        self.installed: list[str] = []
        self.modelfiles: dict[str, str] = {}
        self.fail_install_of: str | None = None
        self.fail_install_times: int | None = None  # None fails every time
        self.removed: list[str] = []

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
        return self.second_opinion_ad, self.second_cost

    def second_dedup(self, case, tuning):
        return self.second_opinion_dedup, self.second_cost

    def second_rule(self, rule_text, records, tuning):
        self.rule_checks.append((rule_text, list(records)))
        return self.second_opinion_rule, self.second_cost

    def predict_ads(self, model, cases):
        out = []
        for c in cases:
            p = (not c.expected) if c.today_mistake else c.expected  # the live gate's behavior
            if model == CANDIDATE:
                if self.candidate_fix and c.today_mistake:
                    p = c.expected
                if self.candidate_break_base and c.id.startswith("base:"):
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


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    monkeypatch.setattr(tuner, "judge_ads", f.judge_ads)
    monkeypatch.setattr(tuner, "judge_dedup", f.judge_dedup)
    monkeypatch.setattr(tuner, "second_opinion_ad", f.second_ad)
    monkeypatch.setattr(tuner, "second_opinion_dedup", f.second_dedup)
    monkeypatch.setattr(tuner, "second_opinion_rule", f.second_rule)
    monkeypatch.setattr(tuner, "predict_ads", f.predict_ads)
    monkeypatch.setattr(tuner, "predict_dedup", f.predict_dedup)
    monkeypatch.setattr(tuner, "install_ad_detector_model", f.install)
    monkeypatch.setattr(tuner, "remove_ollama_model", f.remove)
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
    real_predict = tuner.predict_ads
    monkeypatch.setattr(tuner, "predict_ads", lambda model, cases: seen.extend(c.id for c in cases)
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


def test_first_live_rebuild_fails_and_snapshot_rebuild_succeeds(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    fakes.fail_install_of = LIVE
    fakes.fail_install_times = 1
    out = run_tune(cfg, DATE, log=lambda s: None)
    s = store(cfg)
    snap = s.latest_snapshot(DATE)
    text = out.report_path.read_text()
    assert not out.ads_applied and f"Rebuilt {LIVE} from the snapshot" in text
    assert fakes.installed[-2:] == [LIVE, LIVE]
    assert fakes.modelfiles[LIVE] == (snap / "Modelfile").read_text()
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


def test_heuristic_records_skipped_and_removals_judged_first(cfg, evidence, fakes):
    cfg.tuning.max_items_per_run = 2
    run_tune(cfg, DATE, log=lambda s: None)
    assert [r.verdict for r in fakes.judged] == ["ad", "ad"]


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


def test_rollback_restores_snapshot_and_rebuilds(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    run_tune(cfg, DATE, log=lambda s: None)
    assert store(cfg).ad_examples()
    rollback(cfg, DATE)
    assert store(cfg).ad_examples() == [] and fakes.installed[-1] == LIVE
    assert "AI Report issue 42" not in fakes.modelfiles[LIVE]


def test_rollback_without_snapshot_raises(cfg, fakes):
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        rollback(cfg, DATE)


def test_cli_tune_runs_with_today_by_default(cfg, monkeypatch):
    seen = {}
    monkeypatch.setattr(tuner, "run_tune", lambda config, date, log=print: seen.update(date=date))
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
