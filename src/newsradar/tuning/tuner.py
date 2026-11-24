"""`newsradar tune`: the daily run that judges the day's ad-gate and dedup
decisions, gates a candidate against the live versions, applies what passes,
and writes a report.

Both gates run before either part is applied. Any failed judge, second
opinion, candidate build or gate call means nothing is applied for either
part. The tuner writes only under `[tuning] data_dir` and to the local Ollama
models, and never runs git. Before the first judge call, a preflight checks
that the ollama command resolves and that the live model's Modelfile reads. If
either fails, the run stops there and spends nothing.

When the ads or dedup gate rejects only because it broke cases from today's
judge, each of those cases gets a blind second opinion. A case the second
opinion disputes is dropped from the gate and from the saved change, and the
gate decides again on the rest.

The ads gate lets a change keep ads from today's judged items, but only when
every broken case is one of those and the change fixes at least
KEPT_AD_FIX_RATIO cases per kept ad. Fixed seed examples don't count toward
that ratio. News the candidate now removes, and any break in a base, learned
or seed example, always block.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from newsradar.ad_detector.update_modelfile import (
    LEARNED_RULES_END,
    LEARNED_RULES_START,
    TRAINING_SECTION_START,
    ModelfileTooLargeError,
)
from newsradar.claude_cli import ClaudeCallError
from newsradar.config import Config
from newsradar.ollama_cli import OllamaNotFoundError, ollama_bin
from newsradar.tuning.evidence import AdRecord, load_day, prune_evidence
from newsradar.tuning.gate import KEPT_AD_FIX_RATIO, GateError
from newsradar.tuning.gating import MAX_BLOCKERS_FOR_SECOND_OPINION, _GateSteps
from newsradar.tuning.judge import (
    AD_TYPES,
    RULE_CHECK_ITEMS,
    SUMMARY_CHARS,
    AdJudgement,
    DedupCase,
    JudgeError,
    ProposedRule,
    _make_example,
    judge_ads,
    judge_dedup,
    second_opinion_ad,
    second_opinion_dedup,
    second_opinion_dedup_groups,
    second_opinion_rule,
)
from newsradar.tuning.models import (
    SnapshotError,
    _build_dir,
    _build_text,
    _install,
    candidate_model_name,
    installed_modelfile,
    live_model_name,
    remove_ollama_model,
)
from newsradar.tuning.report import TuneOutcome, _AdPlan, _DedupPlan, _Mistake, _Report
from newsradar.tuning.store import AD_EXAMPLES_FILE, AD_RULES_FILE, TuningStore, domain_only

__all__ = [
    "KEPT_AD_FIX_RATIO", "MAX_BLOCKERS_FOR_SECOND_OPINION", "ModelfileTooLargeError", "OllamaNotFoundError",
    "PreflightError", "SeedError", "SnapshotError", "TuneOutcome", "candidate_model_name", "installed_modelfile",
    "live_model_name", "load_seed", "rebuild_live_model", "remove_ollama_model", "rollback", "run_tune",
]

# Text that would end the SYSTEM block or confuse the marker search in
# build_live_modelfile, so it must never reach the learned files.
_FORBIDDEN = ('"""', LEARNED_RULES_START.strip(), LEARNED_RULES_END.strip(), TRAINING_SECTION_START.strip())

# A rule that tells the gate to output is_ad: true makes it remove more, whatever leaning it claims.
_SAYS_AD = re.compile(r"is_ad[\"'`]?\s*[:=]\s*true", re.IGNORECASE)


class SeedError(RuntimeError):
    """The --include-seed folder is missing or holds files the tuner can't read."""



class PreflightError(RuntimeError):
    """The ollama command or the live model's Modelfile is out of reach, so the
    run stops before any paid judge call."""


# A failure of any of these means the run can't trust its results and applies nothing.
_RUN_ERRORS = (JudgeError, ClaudeCallError, GateError, SeedError, SnapshotError,
               ModelfileTooLargeError, subprocess.CalledProcessError, OSError)



def rebuild_live_model(config: Config, model_name: str | None = None) -> str:
    """Build the live ad-detector from the committed base plus the learned data.
    Returns the Modelfile text. Raises CalledProcessError when `ollama create` fails."""
    store = TuningStore(config.tuning.data_dir)
    text = _build_text(config, store.ad_examples(), store.ad_rules())
    _install(config, text, model_name or live_model_name(config))
    return text


def rollback(config: Config, date: str) -> bool:
    """Restore the parts the latest snapshot on date applied. When that run
    changed the ads, reinstall the live model from the Modelfile the snapshot
    took from the model that was live then. Returns True when it reinstalled.

    Raises FileNotFoundError when there is no snapshot, or when an ads
    snapshot holds no Modelfile, OllamaNotFoundError when an ads snapshot
    needs a reinstall and there is no ollama, and BadSnapshotError when its
    applied.json is damaged. These checks run before any file changes. A
    failed reinstall raises CalledProcessError after the files are restored."""
    store = TuningStore(config.tuning.data_dir)
    snap = store.latest_snapshot(date)
    if snap is None:
        raise FileNotFoundError(f"no snapshot for {date} in {store.root / 'snapshots'}")
    parts = store.snapshot_parts(snap)
    if "ads" in parts and not store.has_modelfile(snap):
        raise FileNotFoundError(f"snapshot {snap} changed the ads but holds no Modelfile; nothing was restored")
    if "ads" in parts:
        ollama_bin()  # the reinstall needs it, so find it before any file changes
    text = store.restore(snap, parts)
    if "ads" not in parts:
        return False
    if text is None:  # has_modelfile saw it, so only a file removed since then gets here
        raise FileNotFoundError(f"snapshot {snap} lost its Modelfile during the rollback")
    _install(config, text, live_model_name(config))
    return True


def _text(exc: BaseException) -> str:
    return str(exc).rstrip(".")


def load_seed(seed_dir: Path) -> tuple[list, list[str]]:
    """Read a seed folder in the store's format. Returns (raw examples, rule
    lines). Raises SeedError when the folder or its files can't be read."""
    seed_dir = Path(seed_dir)
    examples_path, rules_path = seed_dir / AD_EXAMPLES_FILE, seed_dir / AD_RULES_FILE
    if not seed_dir.is_dir():
        raise SeedError(f"seed folder {seed_dir} does not exist")
    if not examples_path.exists() and not rules_path.exists():
        raise SeedError(f"seed folder {seed_dir} holds neither {AD_EXAMPLES_FILE} nor {AD_RULES_FILE}")
    examples: list = []
    if examples_path.exists():
        try:
            examples = json.loads(examples_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SeedError(f"can't read {examples_path}: {exc}") from exc
        if not isinstance(examples, list):
            raise SeedError(f"{examples_path} must hold a JSON list of examples")
    rules: list[str] = []
    if rules_path.exists():
        try:
            rules = [line.strip() for line in rules_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            raise SeedError(f"can't read {rules_path}: {exc}") from exc
    return examples, rules


def _seed_example(raw, date: str) -> tuple[dict | None, str]:
    """Sanitize one seed example like judge output. Returns (example, "") or (None, why)."""
    if not isinstance(raw, dict):
        return None, "it is not a JSON object"
    title = str(raw.get("title") or "").strip()
    is_ad = raw.get("is_ad")
    ad_type = raw.get("ad_type") or ("not_ad" if is_ad is False else "")
    if not title or not isinstance(is_ad, bool) or ad_type not in AD_TYPES or is_ad != (ad_type != "not_ad"):
        return None, "it needs a title, a boolean is_ad and an ad_type that fits it"
    ex = {
        "id": str(raw.get("id") or "seed-" + (re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") or "item")),
        "title": title,
        "summary": str(raw.get("summary") or "")[:SUMMARY_CHARS],
        "source": str(raw.get("source") or ""),
        "link": domain_only(str(raw.get("link") or "")),
        "is_ad": is_ad,
        "ad_type": ad_type,
        "notes": str(raw.get("notes") or ""),
        "added": str(raw.get("added") or date),
    }
    problem = _modelfile_problem(ex["title"], ex["summary"], ex["notes"])
    return (None, problem) if problem else (ex, "")


def _modelfile_problem(*texts: str) -> str | None:
    for text in texts:
        for bad in _FORBIDDEN:
            if bad in text:
                return f"it contains {bad!r}, which would break the Modelfile"
    return None


# ---------- the run ----------

class _Tuner(_GateSteps):
    def __init__(self, config: Config, date: str, report: _Report, outcome: TuneOutcome,
                 log: Callable[[str], None] = lambda s: None, seed_dir: Path | None = None):
        self.config = config
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.seed_examples: list[dict] = []  # sanitized seed examples in the candidate
        self.tuning = config.tuning
        self.date = date
        self.store = TuningStore(config.tuning.data_dir)
        self.rep = report
        self.outcome = outcome
        self.log = log
        self.live = live_model_name(config)
        self.candidate = candidate_model_name(config)
        self.candidate_attempted = False

    def run(self) -> None:
        seed_examples, seed_rules = load_seed(self.seed_dir) if self.seed_dir else ([], [])
        ads, dedups, bad = load_day(self.tuning.data_dir, self.date)
        self.rep.bad_lines = bad
        if not ads and not dedups:
            self.rep.no_evidence = True
            self.log(f"tune {self.date}: no evidence for this date")
            return
        self.log(f"tune {self.date}: loaded {len(ads)} ad records and {len(dedups)} dedup batches")
        self._preflight()

        records = self._select_ads(ads)
        if records:
            self.log(f"tune {self.date}: judging {len(records)} ad records in {-(-len(records) // 20)} batches")
        judgement = judge_ads(records, self.tuning) if records else AdJudgement([], [], 0.0)
        self.rep.cost += judgement.cost_usd
        self.rep.ad_skipped = judgement.skipped
        self.rep.ad_shifted = judgement.shifted
        self.rep.judged_ads -= judgement.skipped
        cases: list[DedupCase] = []
        if dedups:
            self.log(f"tune {self.date}: judging {len(dedups)} dedup batches")
            cases, cost, skipped = judge_dedup(dedups, self.tuning)
            self.rep.cost += cost
            self.rep.dedup_skipped = skipped
        self.rep.dedup_batches = len(dedups)

        self.log(f"tune {self.date}: second opinions for {sum(1 for v in judgement.verdicts if not v.correct)} "
                 f"ad mistakes")
        mistakes = self._second_opinions(records, judgement)
        ad_plan = self._plan_ads(records, judgement, mistakes, seed_examples, seed_rules)
        dedup_plan = self._plan_dedup(cases)

        # Both gates run before either part is applied.
        if ad_plan:
            self.rep.ad_gate = self._gate_ads(records, judgement, mistakes, ad_plan)
            self.rep.ads_status = "accepted" if self.rep.ad_gate.accepted else "rejected"
            g = self.rep.ad_gate
            self.log(f"tune {self.date}: ads gate {self.rep.ads_status}, current errors {g.current_errors}, "
                     f"candidate errors {g.candidate_errors}")
        if dedup_plan:
            self.log(f"tune {self.date}: running the dedup gate")
            self.rep.dedup_gate = self._gate_dedup(dedup_plan)
            self.rep.dedup_status = "accepted" if self.rep.dedup_gate.accepted else "rejected"
            g = self.rep.dedup_gate
            self.log(f"tune {self.date}: dedup gate {self.rep.dedup_status}, current errors {g.current_errors}, "
                     f"candidate errors {g.candidate_errors}")

        self.log(f"tune {self.date}: applying accepted changes, if any")
        self._apply(
            ad_plan if self.rep.ad_gate and self.rep.ad_gate.accepted else None,
            dedup_plan if self.rep.dedup_gate and self.rep.dedup_gate.accepted else None,
        )

    def _preflight(self) -> None:
        """Check, before any paid judge call, what the gate and the snapshot need
        later: the ollama command resolves and the live model's Modelfile reads."""
        try:
            ollama_bin()
            installed_modelfile(self.live)
        except (OllamaNotFoundError, SnapshotError) as exc:
            raise PreflightError(f"{type(exc).__name__}: {exc}") from exc

    def _select_ads(self, ads: list[AdRecord]) -> list[AdRecord]:
        ordered = [r for r in ads if r.verdict == "ad"] + [r for r in ads if r.verdict != "ad"]
        cap = max(self.tuning.max_items_per_run, 0)
        self.rep.over_cap = max(len(ordered) - cap, 0)
        self.rep.judged_ads = min(len(ordered), cap)
        return ordered[:cap]

    def _second_ad(self, record: AdRecord) -> bool | None:
        """The blind second opinion's is_ad, or None when its reply was malformed."""
        self.rep.second_calls += 1
        is_ad, cost = second_opinion_ad(record, self.tuning)
        self.rep.cost += cost
        return is_ad

    def _second_dedup(self, case: DedupCase) -> bool | None:
        self.rep.second_calls += 1
        agrees, cost = second_opinion_dedup(case, self.tuning)
        self.rep.cost += cost
        return agrees

    def _second_dedup_groups(self, case: DedupCase) -> list[list[int]] | None:
        self.rep.second_calls += 1
        groups, cost = second_opinion_dedup_groups(case, self.tuning)
        self.rep.cost += cost
        return groups

    def _second_opinions(self, records: list[AdRecord], judgement: AdJudgement) -> list[_Mistake]:
        by_key = {r.key: r for r in records}
        mistakes = []
        for v in judgement.verdicts:
            if v.correct:
                continue
            record = by_key[v.key]
            # A mistake labeled as an ad teaches the gate to remove more, so it needs a second opinion.
            if v.is_ad:
                is_ad = self._second_ad(record)
                mistakes.append(_Mistake(record, v, is_ad is True, unusable=is_ad is None))
            else:
                mistakes.append(_Mistake(record, v, None))
        self.rep.mistakes = mistakes
        return mistakes

    def _plan_ads(self, records: list[AdRecord], judgement: AdJudgement, mistakes: list[_Mistake],
                  seed_examples: list | None = None, seed_rules: list[str] | None = None) -> _AdPlan | None:
        stored = self.store.ad_examples()
        known_ids = {str(e.get("id")) for e in stored}
        cap = self.tuning.max_new_examples_per_day
        new: list[dict] = []
        example_keys: dict[str, str] = {}
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
                example_keys[str(ex["id"])] = m.record.key
                lines.append(f'- New example "{ex["title"]}" ({label(ex)}, id {ex["id"]}) from a mistake.')

        self._add_balancing_example(records, judgement, new, known_ids, cap, example_keys)

        # The seed is not subject to the per-day cap and keeps its own added date.
        seed: list[dict] = []
        for raw in seed_examples or []:
            ex, problem = _seed_example(raw, self.date)
            title = ex["title"] if ex else (raw.get("title") if isinstance(raw, dict) else raw)
            if ex is None:
                self.rep.seed_lines.append(f'- Example "{title}" dropped: {problem}.')
            elif ex["id"] in known_ids:
                self.rep.seed_lines.append(f'- Example "{title}" dropped: id {ex["id"]} is already learned.')
            else:
                seed.append(ex)
                known_ids.add(ex["id"])
                self.rep.seed_lines.append(f'- Example "{title}" ({label(ex)}, id {ex["id"]}) included.')
        self.seed_examples = seed

        by_key = {r.key: r for r in records}
        news = [by_key[v.key] for v in judgement.verdicts if not v.is_ad]
        proposed = [(rule, self.rep.rule_lines) for rule in judgement.proposed_rules]
        # Seed rules have no stated leaning, so each gets the rule second opinion.
        proposed += [(ProposedRule(text=t, leaning="remove"), self.rep.seed_lines) for t in seed_rules or []]
        rules = self._plan_rules(proposed, news)
        if not new and not seed and rules is None:
            return None
        return _AdPlan(examples=stored + new + seed,
                       rules=rules if rules is not None else self.store.ad_rules().strip(),
                       new_examples=new + seed, example_keys=example_keys, rules_changed=rules is not None)

    def _add_balancing_example(self, records: list[AdRecord], judgement: AdJudgement,
                               new: list[dict], known_ids: set[str], cap: int,
                               example_keys: dict[str, str]) -> None:
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
            if want_ad and self._second_ad(record) is not True:
                self.rep.example_lines.append(
                    f'- Balancing example "{ex["title"]}" ({kind}) dropped: the second opinion disagreed '
                    "or gave no usable reply.")
                return
            new.append(ex)
            example_keys[str(ex["id"])] = record.key
            self.rep.example_lines.append(
                f'- Balancing example "{ex["title"]}" ({kind}, id {ex["id"]}), handled correctly today.')
            return
        self.rep.example_lines.append(
            f"- No correctly handled {'ad' if want_ad else 'non-ad'} from today to balance the new examples.")

    def _plan_rules(self, proposed: list[tuple[ProposedRule, list[str]]], news: list[AdRecord]) -> str | None:
        """Return the candidate rules text, or None when no rule was added.
        Each rule comes with the report list its outcome goes to. news is
        today's judged items the judge labeled not ads."""
        rules = self.store.ad_rules().strip()
        limit = self.tuning.max_rules_chars
        added = False
        for rule, lines in proposed:
            text = rule.text.strip()
            leaning = "remove" if _SAYS_AD.search(text) else rule.leaning
            where = f'- Rule ({leaning}) "{text}"'
            if leaning != rule.leaning:
                where += " (treated as remove-leaning: it says is_ad: true)"
            problem = _modelfile_problem(text)
            if problem:
                lines.append(f"{where} dropped: {problem}.")
                continue
            if text in rules:
                lines.append(f"{where} dropped: already learned.")
                continue
            joined = f"{rules}\n{text}" if rules else text
            if len(joined) > limit:
                lines.append(f"{where} dropped: the learned rules would reach {len(joined)} characters, "
                             f"over the limit of {limit}.")
                continue
            if leaning == "remove":
                if not news:
                    lines.append(f"{where} dropped: it makes the gate remove more, and no item from today "
                                 "was labeled not an ad, so there is nothing to check it against.")
                    continue
                self.rep.second_calls += 1
                approved, cost = second_opinion_rule(text, news, self.tuning)
                self.rep.cost += cost
                if not approved:
                    lines.append(f"{where} dropped: it makes the gate remove more, and the second opinion "
                                 f"did not approve it against {min(len(news), RULE_CHECK_ITEMS)} of today's "
                                 "news items.")
                    continue
                where += " (second opinion approved)"
            rules = joined
            added = True
            lines.append(f"{where} proposed.")
        return rules if added else None

    def _plan_dedup(self, cases: list[DedupCase]) -> _DedupPlan | None:
        stored = self.store.dedup_cases()
        known = {tuple(c.get("titles", [])) for c in stored}
        new: list[dict] = []
        for case in cases:
            note = f" {case.note.strip().rstrip('.')}." if case.note.strip() else ""
            where = f"- {case.kind}: {case.titles}, expected groups {case.expected_groups}.{note}"
            # A case with expected groups teaches a merge, which removes more.
            if case.kind == "missed_duplicate" or case.expected_groups:
                agreed = self._second_dedup(case)
                if agreed is not True:
                    why = "disagreed" if agreed is False else "gave no usable reply"
                    self.rep.dedup_lines.append(f"{where} Second opinion: {why}, so it was dropped.")
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

    def _undo(self, snap: Path, rebuild: bool) -> None:
        """Put the learned files back as the snapshot has them. With rebuild,
        also reinstall the live model from the Modelfile the snapshot took."""
        lines = self.rep.apply_lines
        rollback_cmd = f"newsradar tune --config {self.config.config_path} --rollback {self.date}"
        try:
            text = self.store.restore(snap)
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
            if text is None:
                raise FileNotFoundError(f"{snap} holds no Modelfile")
            _install(self.config, text, self.live)
            lines.append(f"- Reinstalled {self.live} from the Modelfile the snapshot took of the live model.")
        except Exception as exc:
            lines.append(f"- Rebuilding {self.live} from the snapshot also failed: "
                         f"{type(exc).__name__}: {_text(exc)}. The learned files are restored, "
                         "so run the --rebuild command below once Ollama works.")

    def _apply(self, ad_plan: _AdPlan | None, dedup_plan: _DedupPlan | None) -> None:
        if not ad_plan and not dedup_plan:
            return
        # One snapshot covers both parts, so --rollback DATE undoes the whole run.
        # It keeps the Modelfile of the model actually live now, read back from
        # Ollama, so a rollback reinstalls that model and not a rebuild.
        parts = [name for name, plan in (("ads", ad_plan), ("dedup", dedup_plan)) if plan]
        live_text = installed_modelfile(self.live) if ad_plan else None
        snap = self.store.snapshot(self.date, live_text, parts)
        lines = self.rep.apply_lines
        what = "the learned files and the live Modelfile" if ad_plan else "the learned files"
        lines.append(f"- Snapshot of {what}: {snap}")
        live_changed = False
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
                live_changed = True
                lines.append(f"- Rebuilt {self.live} from base plus learned data.")
        except Exception as exc:
            lines.append(f"- Applying failed: {type(exc).__name__}: {_text(exc)}.")
            # A failed `ollama create` leaves the old model in place, so only a
            # model this run did replace needs reinstalling.
            if ad_plan and not live_changed:
                lines.append(f"- {self.live} was not replaced, so it is still the model from before this run.")
            self._undo(snap, rebuild=live_changed)
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


def run_tune(config: Config, date: str, log: Callable[[str], None] | None = None,
             seed_dir: Path | None = None) -> TuneOutcome:
    """Run the daily tune for date. Never raises: every failure ends up in the report.

    seed_dir holds learned_ad_examples.json and/or learned_ad_rules.txt to
    propose alongside the judge's findings. The seed goes through the gate and
    is saved only if the ads change is accepted.

    The default log is the scan's timestamped, flushing log line, so `tail -f` follows the run.
    """
    if log is None:
        from newsradar.cli import log as scan_log
        log = scan_log
    store = TuningStore(config.tuning.data_dir)
    report = _Report(date=date, config_path=config.config_path, seed_dir=Path(seed_dir) if seed_dir else None)
    outcome = TuneOutcome(False, False, "", store.report_path(date))
    tuner = _Tuner(config, date, report, outcome, log, seed_dir)
    try:
        tuner.run()
    except PreflightError as exc:
        report.error = f"Preflight failed, so the run made no judge calls and applied nothing: {exc}"
    except ModelfileTooLargeError as exc:
        report.error = (f"The candidate Modelfile is too large for num_ctx, so the run applied nothing: "
                        f"{type(exc).__name__}: {exc}")
    except _RUN_ERRORS as exc:
        if isinstance(exc, JudgeError):
            report.cost += exc.cost_usd  # calls that ran before the failure still cost
        report.error = f"A call failed, so the run applied nothing: {type(exc).__name__}: {exc}"
    except Exception as exc:  # a bug; report it rather than crash the daily job
        report.error = f"Unexpected error, so the run applied nothing: {type(exc).__name__}: {exc}"
    if report.error:
        outcome.ads_applied = outcome.dedup_applied = False

    log(f"tune {date}: cleaning up and pruning old evidence")
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
