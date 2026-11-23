"""`newsradar tune`: the daily run that judges the day's ad-gate and dedup
decisions, gates a candidate against the live versions, applies what passes,
and writes a report.

Both gates run before either part is applied. Any failed judge, second
opinion, candidate build or gate call means nothing is applied for either
part. The tuner writes only under `[tuning] data_dir` and to the local Ollama
models, and never runs git.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from newsradar.ad_detector.install import install_ad_detector_model
from newsradar.ad_detector.update_modelfile import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODELFILE_PATH,
    LEARNED_RULES_END,
    LEARNED_RULES_START,
    TRAINING_SECTION_START,
    build_live_modelfile,
    load_examples,
)
from newsradar.claude_cli import ClaudeCallError
from newsradar.config import Config
from newsradar.tuning.evidence import AdRecord, load_day, prune_evidence
from newsradar.tuning.gate import (
    Case,
    GateError,
    GateResult,
    decide,
    dedup_case_ok,
    predict_ads,
    predict_dedup,
    stabilize,
)
from newsradar.tuning.judge import (
    RULE_CHECK_ITEMS,
    AdJudgement,
    AdVerdict,
    DedupCase,
    JudgeError,
    ProposedRule,
    _make_example,
    judge_ads,
    judge_dedup,
    second_opinion_ad,
    second_opinion_dedup,
    second_opinion_rule,
)
from newsradar.tuning.store import TuningStore

BASE_MODELFILE = DEFAULT_MODELFILE_PATH
BASE_EXAMPLES = DEFAULT_DATA_PATH

# Text that would end the SYSTEM block or confuse the marker search in
# build_live_modelfile, so it must never reach the learned files.
_FORBIDDEN = ('"""', LEARNED_RULES_START.strip(), LEARNED_RULES_END.strip(), TRAINING_SECTION_START.strip())

# A failure of any of these means the run can't trust its results and applies nothing.
_RUN_ERRORS = (JudgeError, ClaudeCallError, GateError, subprocess.CalledProcessError, OSError)


@dataclass
class TuneOutcome:
    ads_applied: bool
    dedup_applied: bool
    summary: str
    report_path: Path


def live_model_name(config: Config) -> str:
    return config.models.ad_detector_model.removeprefix("ollama/")


def candidate_model_name(config: Config) -> str:
    return f"{live_model_name(config)}-candidate"


def remove_ollama_model(name: str) -> None:
    """Run `ollama rm`, ignoring every failure (the model may not exist)."""
    try:
        subprocess.run(["ollama", "rm", name], capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def _build_text(config: Config, examples: list[dict], rules: str) -> str:
    return build_live_modelfile(
        BASE_MODELFILE.read_text(encoding="utf-8"),
        load_examples(BASE_EXAMPLES),
        examples,
        rules,
        config.tuning.learned_examples_per_class,
    )


def _build_dir(config: Config) -> Path:
    return TuningStore(config.tuning.data_dir).root / "build"


def _install(config: Config, text: str, model_name: str) -> None:
    """Write text to a temporary Modelfile under data_dir, run `ollama create`, delete the file."""
    path = _build_dir(config) / f"{model_name}.Modelfile"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        install_ad_detector_model(path, model_name)
    finally:
        path.unlink(missing_ok=True)


def rebuild_live_model(config: Config, model_name: str | None = None) -> str:
    """Build the live ad-detector from the committed base plus the learned data.
    Returns the Modelfile text. Raises CalledProcessError when `ollama create` fails."""
    store = TuningStore(config.tuning.data_dir)
    text = _build_text(config, store.ad_examples(), store.ad_rules())
    _install(config, text, model_name or live_model_name(config))
    return text


def rollback(config: Config, date: str) -> None:
    """Restore the latest snapshot taken on date and rebuild the live model from it."""
    store = TuningStore(config.tuning.data_dir)
    snap = store.latest_snapshot(date)
    if snap is None:
        raise FileNotFoundError(f"no snapshot for {date} in {store.root / 'snapshots'}")
    text = store.restore(snap)
    _install(config, text, live_model_name(config))


def _text(exc: BaseException) -> str:
    return str(exc).rstrip(".")


def _modelfile_problem(*texts: str) -> str | None:
    for text in texts:
        for bad in _FORBIDDEN:
            if bad in text:
                return f"it contains {bad!r}, which would break the Modelfile"
    return None


# ---------- run state and report ----------

@dataclass
class _Mistake:
    record: AdRecord
    verdict: AdVerdict
    second: bool | None  # None when no second opinion was needed


@dataclass
class _AdPlan:
    examples: list[dict]  # the full candidate learned list
    rules: str
    new_examples: list[dict]


@dataclass
class _DedupPlan:
    cases: list[dict]  # the full candidate learned list
    new_count: int


@dataclass
class _Report:
    date: str
    config_path: Path
    no_evidence: bool = False
    bad_lines: int = 0
    judged_ads: int = 0
    heuristic_skipped: int = 0
    over_cap: int = 0
    dedup_batches: int = 0
    dedup_skipped: int = 0
    cost: float = 0.0
    second_calls: int = 0
    error: str = ""
    mistakes: list[_Mistake] = field(default_factory=list)
    example_lines: list[str] = field(default_factory=list)
    rule_lines: list[str] = field(default_factory=list)
    dedup_lines: list[str] = field(default_factory=list)
    ad_gate: GateResult | None = None
    dedup_gate: GateResult | None = None
    case_titles: dict[str, str] = field(default_factory=dict)
    apply_lines: list[str] = field(default_factory=list)
    cleanup_lines: list[str] = field(default_factory=list)
    ads_status: str = "no change proposed"
    undo_status: str = ""
    dedup_status: str = "no change proposed"

    def summary(self, path: Path) -> str:
        if self.no_evidence:
            return f"tune {self.date}: no evidence, nothing to tune; report {path}"
        if self.error:
            return f"tune {self.date}: applied nothing ({self.error}); report {path}"
        return (f"tune {self.date}: ads {self.ads_status}, dedup {self.dedup_status}, "
                f"judge cost ${self.cost:.2f}; report {path}")

    def _gate_lines(self, name: str, result: GateResult | None, status: str) -> list[str]:
        if result is None:
            return [f"{name}: not run ({status})."]
        verdict = "accepted" if result.accepted else "rejected"
        lines = [f"{name}: current errors {result.current_errors}, candidate errors "
                 f"{result.candidate_errors}. {verdict.capitalize()}: {result.reason}."]
        for label, ids in (("Fixed", result.fixed), ("Broken", result.broken)):
            for case_id in ids:
                lines.append(f"- {label}: {case_id} ({self.case_titles.get(case_id, '')})")
        return lines

    def render(self) -> str:
        if self.no_evidence:
            extra = f" ({self.bad_lines} bad evidence lines skipped)" if self.bad_lines else ""
            return f"Tuning report {self.date}: no evidence for this day, nothing to tune{extra}.\n"

        out = [f"# Tuning report {self.date}", ""]
        if self.error:
            out += [f"Applied nothing. {self.error}", ""]
        else:
            out += [f"Ads: {self.ads_status}. Dedup: {self.dedup_status}.", ""]

        out += ["## Evidence", "",
                f"- Ad-gate decisions judged: {self.judged_ads}.",
                f"- Title-phrase removals skipped: {self.heuristic_skipped}.",
                f"- Decisions over the cap of items per run: {self.over_cap}.",
                f"- Dedup batches judged: {self.dedup_batches}.",
                f"- Bad evidence lines skipped: {self.bad_lines}.", ""]

        out += ["## Ad gate mistakes", ""]
        if not self.mistakes:
            out.append("None found.")
        for m in self.mistakes:
            label = "an ad" if m.verdict.is_ad else "not an ad"
            if m.second is None:
                second = "not needed, the change keeps more"
            else:
                second = "agreed" if m.second else "disagreed, so the change was dropped"
            out.append(f'- "{m.record.title}" ({m.record.source}). Gate verdict: {m.record.verdict}. '
                       f"Judge: {label} ({m.verdict.ad_type}), {m.verdict.reason}. Second opinion: {second}.")
        out.append("")

        hidden = [m for m in self.mistakes if not m.verdict.visible_in_excerpt]
        if hidden:
            out += ["## Needs a longer input", "",
                    "The judge says the 1,000-character excerpt the gate sees does not show the answer "
                    "for these. They are not learned as examples, and fixing them does not count "
                    "toward accepting a change.", ""]
            out += [f'- "{m.record.title}" ({m.record.source})' for m in hidden]
            out.append("")

        out += ["## Proposed ad changes", ""]
        out += (self.example_lines + self.rule_lines) or ["None."]
        out.append("")

        out += ["## Dedup cases", ""]
        out += self.dedup_lines or ["None."]
        out += [f"- Cases skipped: {self.dedup_skipped} (the judge used an unknown or repeated id).", ""]

        out += ["## Gate results", ""]
        out += self._gate_lines("Ads", self.ad_gate, self.ads_status)
        out += self._gate_lines("Dedup", self.dedup_gate, self.dedup_status)
        out.append("")

        if self.apply_lines:
            out += ["## Applied", ""] + self.apply_lines + [""]

        out += ["## Usage", "",
                f"Judge cost: ${self.cost:.2f}, the summed cost_usd of every judge and "
                f"second-opinion call ({self.second_calls} second-opinion calls).", ""]

        if self.cleanup_lines:
            out += ["## Clean up", ""] + self.cleanup_lines + [""]

        out += ["## Commands", "",
                "Rebuild the live ad-detector from base plus learned data:",
                "", f"    newsradar tune --config {self.config_path} --rebuild", "",
                f"Undo the latest change applied on {self.date}:",
                "", f"    newsradar tune --config {self.config_path} --rollback {self.date}", ""]
        return "\n".join(out)


# ---------- the run ----------

class _Tuner:
    def __init__(self, config: Config, date: str, report: _Report, outcome: TuneOutcome):
        self.config = config
        self.tuning = config.tuning
        self.date = date
        self.store = TuningStore(config.tuning.data_dir)
        self.rep = report
        self.outcome = outcome
        self.live = live_model_name(config)
        self.candidate = candidate_model_name(config)
        self.candidate_attempted = False

    def run(self) -> None:
        ads, dedups, bad = load_day(self.tuning.data_dir, self.date)
        self.rep.bad_lines = bad
        if not ads and not dedups:
            self.rep.no_evidence = True
            return

        records = self._select_ads(ads)
        judgement = judge_ads(records, self.tuning) if records else AdJudgement([], [], 0.0)
        self.rep.cost += judgement.cost_usd
        cases: list[DedupCase] = []
        if dedups:
            cases, cost, skipped = judge_dedup(dedups, self.tuning)
            self.rep.cost += cost
            self.rep.dedup_skipped = skipped
        self.rep.dedup_batches = len(dedups)

        mistakes = self._second_opinions(records, judgement)
        ad_plan = self._plan_ads(records, judgement, mistakes)
        dedup_plan = self._plan_dedup(cases)

        # Both gates run before either part is applied.
        if ad_plan:
            self.rep.ad_gate = self._gate_ads(records, judgement, mistakes, ad_plan)
            self.rep.ads_status = "accepted" if self.rep.ad_gate.accepted else "rejected"
        if dedup_plan:
            self.rep.dedup_gate = self._gate_dedup(dedup_plan)
            self.rep.dedup_status = "accepted" if self.rep.dedup_gate.accepted else "rejected"

        self._apply(
            ad_plan if self.rep.ad_gate and self.rep.ad_gate.accepted else None,
            dedup_plan if self.rep.dedup_gate and self.rep.dedup_gate.accepted else None,
        )

    def _select_ads(self, ads: list[AdRecord]) -> list[AdRecord]:
        judged = [r for r in ads if r.verdict != "heuristic"]
        self.rep.heuristic_skipped = len(ads) - len(judged)
        ordered = [r for r in judged if r.verdict == "ad"] + [r for r in judged if r.verdict != "ad"]
        cap = max(self.tuning.max_items_per_run, 0)
        self.rep.over_cap = max(len(ordered) - cap, 0)
        self.rep.judged_ads = min(len(ordered), cap)
        return ordered[:cap]

    def _second_ad(self, record: AdRecord) -> bool:
        self.rep.second_calls += 1
        agreed, cost = second_opinion_ad(record, self.tuning)
        self.rep.cost += cost
        return agreed

    def _second_opinions(self, records: list[AdRecord], judgement: AdJudgement) -> list[_Mistake]:
        by_key = {r.key: r for r in records}
        mistakes = []
        for v in judgement.verdicts:
            if v.correct:
                continue
            record = by_key[v.key]
            # A mistake labeled as an ad teaches the gate to remove more, so it needs a second opinion.
            second = self._second_ad(record) if v.is_ad else None
            mistakes.append(_Mistake(record, v, second))
        self.rep.mistakes = mistakes
        return mistakes

    def _plan_ads(self, records: list[AdRecord], judgement: AdJudgement,
                  mistakes: list[_Mistake]) -> _AdPlan | None:
        stored = self.store.ad_examples()
        known_ids = {str(e.get("id")) for e in stored}
        cap = self.tuning.max_new_examples_per_day
        new: list[dict] = []
        lines = self.rep.example_lines

        def label(e: dict) -> str:
            return "ad" if e["is_ad"] else "not ad"

        for m in mistakes:
            if m.second is False:
                continue
            ex = dict(m.verdict.example or {}, added=self.date)
            if not m.verdict.visible_in_excerpt:
                # The summary would show the misleading excerpt under the opposite label.
                lines.append(f'- Example "{ex.get("title")}" not learned: the excerpt the gate reads '
                             "does not show the answer (see Needs a longer input).")
                continue
            problem = _modelfile_problem(str(ex.get("title", "")), str(ex.get("summary", "")),
                                         str(ex.get("notes", "")))
            if problem:
                lines.append(f'- Example "{ex.get("title")}" dropped: {problem}.')
            elif str(ex.get("id")) in known_ids:
                lines.append(f'- Example "{ex["title"]}" dropped: id {ex["id"]} is already learned.')
            elif len(new) >= cap:
                lines.append(f'- Example "{ex["title"]}" dropped: over the cap of {cap} new examples per day.')
            else:
                new.append(ex)
                known_ids.add(str(ex["id"]))
                lines.append(f'- New example "{ex["title"]}" ({label(ex)}, id {ex["id"]}) from a mistake.')

        self._add_balancing_example(records, judgement, new, known_ids, cap)
        by_key = {r.key: r for r in records}
        news = [by_key[v.key] for v in judgement.verdicts if not v.is_ad]
        rules = self._plan_rules(judgement.proposed_rules, news)
        if not new and rules is None:
            return None
        return _AdPlan(examples=stored + new,
                       rules=rules if rules is not None else self.store.ad_rules().strip(),
                       new_examples=new)

    def _add_balancing_example(self, records: list[AdRecord], judgement: AdJudgement,
                               new: list[dict], known_ids: set[str], cap: int) -> None:
        """Add one correctly handled item of the minority label, so a day's
        examples don't all push the gate the same way."""
        if not new or len(new) >= cap:
            return
        n_ads = sum(1 for e in new if e["is_ad"])
        if n_ads * 2 == len(new):
            return
        want_ad = n_ads * 2 < len(new)
        by_key = {r.key: r for r in records}
        for v in judgement.verdicts:
            if not v.correct or v.is_ad != want_ad:
                continue
            record = by_key[v.key]
            ex = _make_example(record, v, {"summary": record.text_sent, "notes": v.reason})
            ex["added"] = self.date
            if str(ex["id"]) in known_ids or _modelfile_problem(ex["title"], ex["summary"], ex["notes"]):
                continue
            kind = "ad" if want_ad else "not ad"
            if want_ad and not self._second_ad(record):
                self.rep.example_lines.append(
                    f'- Balancing example "{ex["title"]}" ({kind}) dropped: the second opinion disagreed.')
                return
            new.append(ex)
            self.rep.example_lines.append(
                f'- Balancing example "{ex["title"]}" ({kind}, id {ex["id"]}), handled correctly today.')
            return
        self.rep.example_lines.append(
            f"- No correctly handled {'ad' if want_ad else 'non-ad'} from today to balance the new examples.")

    def _plan_rules(self, proposed: list[ProposedRule], news: list[AdRecord]) -> str | None:
        """Return the candidate rules text, or None when no rule was added.
        news is today's judged items the judge labeled not ads."""
        rules = self.store.ad_rules().strip()
        limit = self.tuning.max_rules_chars
        added = False
        for rule in proposed:
            text = rule.text.strip()
            where = f'- Rule ({rule.leaning}) "{text}"'
            problem = _modelfile_problem(text)
            if problem:
                self.rep.rule_lines.append(f"{where} dropped: {problem}.")
                continue
            if text in rules:
                self.rep.rule_lines.append(f"{where} dropped: already learned.")
                continue
            joined = f"{rules}\n{text}" if rules else text
            if len(joined) > limit:
                self.rep.rule_lines.append(
                    f"{where} dropped: the learned rules would reach {len(joined)} characters, over the "
                    f"limit of {limit}.")
                continue
            if rule.leaning == "remove":
                self.rep.second_calls += 1
                approved, cost = second_opinion_rule(text, news, self.tuning)
                self.rep.cost += cost
                if not approved:
                    self.rep.rule_lines.append(
                        f"{where} dropped: it makes the gate remove more, and the second opinion did not "
                        f"approve it against {min(len(news), RULE_CHECK_ITEMS)} of today's news items.")
                    continue
                where += " (second opinion approved)"
            rules = joined
            added = True
            self.rep.rule_lines.append(f"{where} proposed.")
        return rules if added else None

    def _plan_dedup(self, cases: list[DedupCase]) -> _DedupPlan | None:
        stored = self.store.dedup_cases()
        known = {tuple(c.get("titles", [])) for c in stored}
        new: list[dict] = []
        for case in cases:
            note = f" {case.note.strip().rstrip('.')}." if case.note.strip() else ""
            where = f"- {case.kind}: {case.titles}, expected groups {case.expected_groups}.{note}"
            if case.kind == "missed_duplicate":
                self.rep.second_calls += 1
                agreed, cost = second_opinion_dedup(case, self.tuning)
                self.rep.cost += cost
                if not agreed:
                    self.rep.dedup_lines.append(f"{where} Second opinion: disagreed, so it was dropped.")
                    continue
                second = "agreed"
            else:
                second = "not needed, the change keeps more"
            if tuple(case.titles) in known:
                self.rep.dedup_lines.append(f"{where} Dropped: the same titles are already a learned case.")
                continue
            known.add(tuple(case.titles))
            new.append({"titles": list(case.titles), "expected_groups": [list(g) for g in case.expected_groups],
                        "kind": case.kind, "note": case.note, "added": self.date})
            self.rep.dedup_lines.append(f"{where} Second opinion: {second}.")
        if not new:
            return None
        return _DedupPlan(cases=stored + new, new_count=len(new))

    def _install_candidate(self, plan: _AdPlan) -> None:
        text = _build_text(self.config, plan.examples, plan.rules)
        self.candidate_attempted = True
        _install(self.config, text, self.candidate)

    def _gate_ads(self, records: list[AdRecord], judgement: AdJudgement, mistakes: list[_Mistake],
                  plan: _AdPlan) -> GateResult:
        self._install_candidate(plan)
        cases: list[Case] = []
        for prefix, examples in (("base", load_examples(BASE_EXAMPLES)), ("learned", self.store.ad_examples())):
            for e in examples:
                cases.append(Case(id=f"{prefix}:{e['id']}", title=e["title"], text=e["summary"],
                                  expected=bool(e["is_ad"]), today_mistake=False, countable=True))
        disputed = {m.record.key for m in mistakes if m.second is False}
        by_key = {r.key: r for r in records}
        for v in judgement.verdicts:
            record = by_key[v.key]
            if record.key in disputed:
                # The two judges disagree, so trust the gate: a candidate that
                # changes this item's verdict counts as a regression.
                expected, mistake = record.verdict == "ad", False
            else:
                expected, mistake = v.is_ad, not v.correct
            cases.append(Case(id=f"today:{record.key[:8]}", title=record.title, text=record.text_sent,
                              expected=expected, today_mistake=mistake,
                              countable=v.visible_in_excerpt if mistake else True))
        self.rep.case_titles.update({c.id: c.title for c in cases})

        current_ok = [p == c.expected for p, c in zip(predict_ads(self.live, cases), cases, strict=True)]
        candidate_ok = [p == c.expected for p, c in zip(predict_ads(self.candidate, cases), cases, strict=True)]

        def rerun(i: int) -> tuple[bool, bool]:
            c = cases[i]
            return (predict_ads(self.live, [c])[0] == c.expected,
                    predict_ads(self.candidate, [c])[0] == c.expected)

        candidate_ok = stabilize(current_ok, candidate_ok, rerun)
        return decide(cases, current_ok, candidate_ok)

    def _gate_dedup(self, plan: _DedupPlan) -> GateResult:
        cases = [DedupCase(titles=list(c["titles"]), expected_groups=[list(g) for g in c["expected_groups"]],
                           kind=str(c.get("kind", "")), note=str(c.get("note", ""))) for c in plan.cases]
        first_new = len(cases) - plan.new_count
        meta = [Case(id=f"dedup:{i}", title=c.titles[0] if c.titles else "", text="", expected=True,
                     today_mistake=i >= first_new, countable=True) for i, c in enumerate(cases)]
        self.rep.case_titles.update({m.id: m.title for m in meta})
        model = self.config.models.dedup_model
        current_block = self.store.dedup_examples_block()
        candidate_block = TuningStore.dedup_block_for(plan.cases)

        def ok(block: str, subset: list[DedupCase]) -> list[bool]:
            return [dedup_case_ok(c, d) for c, d in zip(subset, predict_dedup(subset, model, block), strict=True)]

        current_ok = ok(current_block, cases)
        candidate_ok = ok(candidate_block, cases)

        def rerun(i: int) -> tuple[bool, bool]:
            return ok(current_block, [cases[i]])[0], ok(candidate_block, [cases[i]])[0]

        candidate_ok = stabilize(current_ok, candidate_ok, rerun)
        return decide(meta, current_ok, candidate_ok)

    def _undo(self, snap: Path, rebuild: bool) -> None:
        """Put the learned files (and, with rebuild, the live model) back as the snapshot has them."""
        lines = self.rep.apply_lines
        rollback_cmd = f"newsradar tune --config {self.config.config_path} --rollback {self.date}"
        try:
            self.store.restore(snap)
        except Exception as exc:
            self.rep.undo_status = "restoring the snapshot failed too"
            lines.append(f"- Restoring the learned files from {snap} failed: {type(exc).__name__}: "
                         f"{_text(exc)}. The learned files may be half-saved. Run `{rollback_cmd}`.")
            return
        self.rep.undo_status = "was undone"
        lines.append(f"- Learned files restored from {snap}; nothing from this run is kept.")
        if not rebuild:
            return
        try:
            _install(self.config, (snap / "Modelfile").read_text(encoding="utf-8"), self.live)
            lines.append(f"- Rebuilt {self.live} from the snapshot.")
        except Exception as exc:
            lines.append(f"- Rebuilding {self.live} from the snapshot also failed: "
                         f"{type(exc).__name__}: {_text(exc)}. The learned files are restored, "
                         "so run the --rebuild command below once Ollama works.")

    def _apply(self, ad_plan: _AdPlan | None, dedup_plan: _DedupPlan | None) -> None:
        if not ad_plan and not dedup_plan:
            return
        # One snapshot covers both parts, so --rollback DATE undoes the whole run.
        current_text = _build_text(self.config, self.store.ad_examples(), self.store.ad_rules())
        snap = self.store.snapshot(self.date, current_text)
        lines = self.rep.apply_lines
        lines.append(f"- Snapshot of the learned files and built Modelfile: {snap}")
        try:
            if ad_plan:
                self.store.save_ad_examples(ad_plan.examples)
                self.store.save_ad_rules(ad_plan.rules + "\n" if ad_plan.rules else "")
                lines.append(f"- Saved {len(ad_plan.new_examples)} new ad examples and the learned rules.")
            if dedup_plan:
                self.store.save_dedup_cases(dedup_plan.cases)
                lines.append(f"- Saved {dedup_plan.new_count} new dedup cases.")
            if ad_plan:
                rebuild_live_model(self.config)
                lines.append(f"- Rebuilt {self.live} from base plus learned data.")
        except Exception as exc:
            lines.append(f"- Applying failed: {type(exc).__name__}: {_text(exc)}.")
            self._undo(snap, rebuild=ad_plan is not None)
            undone = self.rep.undo_status
            if ad_plan:
                self.rep.ads_status = f"accepted, but applying failed and {undone}"
            if dedup_plan:
                self.rep.dedup_status = f"accepted, but applying failed and {undone}"
            return
        self.outcome.ads_applied = ad_plan is not None
        self.outcome.dedup_applied = dedup_plan is not None
        if ad_plan:
            self.rep.ads_status = "applied"
        if dedup_plan:
            self.rep.dedup_status = "applied"


def run_tune(config: Config, date: str, log: Callable[[str], None] = print) -> TuneOutcome:
    """Run the daily tune for date. Never raises: every failure ends up in the report."""
    store = TuningStore(config.tuning.data_dir)
    report = _Report(date=date, config_path=config.config_path)
    outcome = TuneOutcome(False, False, "", store.report_path(date))
    tuner = _Tuner(config, date, report, outcome)
    try:
        tuner.run()
    except _RUN_ERRORS as exc:
        report.error = f"A call failed, so the run applied nothing: {type(exc).__name__}: {exc}"
    except Exception as exc:  # a bug; report it rather than crash the daily job
        report.error = f"Unexpected error, so the run applied nothing: {type(exc).__name__}: {exc}"
    if report.error:
        outcome.ads_applied = outcome.dedup_applied = False

    if tuner.candidate_attempted:
        try:
            remove_ollama_model(tuner.candidate)
            report.cleanup_lines.append(f"- Removed {tuner.candidate}.")
        except Exception as exc:
            report.cleanup_lines.append(f"- Could not remove {tuner.candidate}: {exc}")
    try:
        pruned = prune_evidence(config.tuning.data_dir, date, config.tuning.evidence_days)
        if pruned:
            report.cleanup_lines.append(f"- Deleted evidence older than {config.tuning.evidence_days} days: "
                                        f"{', '.join(pruned)}.")
    except Exception as exc:
        report.cleanup_lines.append(f"- Could not prune old evidence: {exc}")
    try:
        pruned = store.prune(date, config.tuning.evidence_days)
        if pruned:
            report.cleanup_lines.append(f"- Deleted snapshots and reports older than "
                                        f"{config.tuning.evidence_days} days: {', '.join(pruned)}.")
    except Exception as exc:
        report.cleanup_lines.append(f"- Could not prune old snapshots and reports: {exc}")
    shutil.rmtree(_build_dir(config), ignore_errors=True)

    try:
        outcome.report_path.parent.mkdir(parents=True, exist_ok=True)
        outcome.report_path.write_text(report.render(), encoding="utf-8")
    except Exception as exc:
        log(f"tune {date}: could not write the report to {outcome.report_path}: {exc}")
    outcome.summary = report.summary(outcome.report_path)
    log(outcome.summary)
    return outcome
