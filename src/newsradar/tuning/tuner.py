"""`newsradar tune`: the daily run that judges the day's ad-gate and dedup
decisions, gates a candidate against the live versions, applies what passes,
and writes a report.

Both gates run before either part is applied. Any failed judge, second
opinion, candidate build or gate call means nothing is applied for either
part. The tuner writes only under `[tuning] data_dir` and to the local Ollama
models, and never runs git.

When the ads or dedup gate rejects only because it broke cases from today's
judge, each of those cases gets a blind second opinion. A case the second
opinion disputes is dropped from the gate and from the saved change, and the
gate decides again on the rest.

The ads gate lets a change keep ads from today's judged items, but only when
every broken case is one of those and the change fixes at least
KEPT_AD_FIX_RATIO cases per kept ad. News the candidate now removes, and any
break in a base, learned or seed example, always block.
"""

from __future__ import annotations

import json
import re
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
    AD_TYPES,
    RULE_CHECK_ITEMS,
    SUMMARY_CHARS,
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
    second_opinion_dedup_groups,
    second_opinion_rule,
)
from newsradar.tuning.store import AD_EXAMPLES_FILE, AD_RULES_FILE, TuningStore, domain_only

BASE_MODELFILE = DEFAULT_MODELFILE_PATH
BASE_EXAMPLES = DEFAULT_DATA_PATH

# Text that would end the SYSTEM block or confuse the marker search in
# build_live_modelfile, so it must never reach the learned files.
_FORBIDDEN = ('"""', LEARNED_RULES_START.strip(), LEARNED_RULES_END.strip(), TRAINING_SECTION_START.strip())

# A rule that tells the gate to output is_ad: true makes it remove more, whatever leaning it claims.
_SAYS_AD = re.compile(r"is_ad[\"'`]?\s*[:=]\s*true", re.IGNORECASE)

# Second opinions on gate blockers are asked only when this many cases or fewer broke.
MAX_BLOCKERS_FOR_SECOND_OPINION = 2

# The ads gate accepts a change that now keeps some of today's ads only when it
# fixes at least this many cases per kept ad.
KEPT_AD_FIX_RATIO = 5


class SeedError(RuntimeError):
    """The --include-seed folder is missing or holds files the tuner can't read."""


class SnapshotError(RuntimeError):
    """`ollama show --modelfile` failed, so the live model can't be snapshotted."""


# A failure of any of these means the run can't trust its results and applies nothing.
_RUN_ERRORS = (JudgeError, ClaudeCallError, GateError, SeedError, SnapshotError,
               subprocess.CalledProcessError, OSError)


@dataclass
class TuneOutcome:
    ads_applied: bool
    dedup_applied: bool
    summary: str
    report_path: Path


def live_model_name(config: Config) -> str:
    return config.models.ad_detector_model.removeprefix("ollama/")


def candidate_model_name(config: Config) -> str:
    """`<live>-candidate`, with any `:tag` on the live name dropped first."""
    name = live_model_name(config)
    base, sep, tag = name.rpartition(":")
    if sep and "/" not in tag:
        name = base
    return f"{name}-candidate"


def installed_modelfile(name: str) -> str:
    """The Modelfile of the installed model, from `ollama show --modelfile`.
    Raises SnapshotError when the call fails or prints nothing."""
    try:
        proc = subprocess.run(["ollama", "show", "--modelfile", name], capture_output=True, text=True,
                              check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise SnapshotError(f"ollama show --modelfile {name} exited {exc.returncode}: "
                            f"{(exc.stderr or '').strip()[:300]}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise SnapshotError(f"ollama show --modelfile {name} failed: {exc}") from exc
    if not proc.stdout.strip():
        raise SnapshotError(f"ollama show --modelfile {name} printed nothing")
    return proc.stdout


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


def rollback(config: Config, date: str) -> bool:
    """Restore the parts the latest snapshot on date applied. When that run
    changed the ads, reinstall the live model from the Modelfile the snapshot
    took from the model that was live then. Returns True when it reinstalled."""
    store = TuningStore(config.tuning.data_dir)
    snap = store.latest_snapshot(date)
    if snap is None:
        raise FileNotFoundError(f"no snapshot for {date} in {store.root / 'snapshots'}")
    parts = store.snapshot_parts(snap)
    text = store.restore(snap, parts)
    if "ads" not in parts:
        return False
    if text is None:
        raise FileNotFoundError(f"snapshot {snap} changed the ads but holds no Modelfile")
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


# ---------- run state and report ----------

@dataclass
class _Mistake:
    record: AdRecord
    verdict: AdVerdict
    second: bool | None  # None when no second opinion was needed
    unusable: bool = False  # the second opinion's reply was malformed, counted as a disagreement


@dataclass
class _AdPlan:
    examples: list[dict]  # the full candidate learned list
    rules: str
    new_examples: list[dict]
    example_keys: dict[str, str] = field(default_factory=dict)  # new example id -> its record key
    rules_changed: bool = False


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
    over_cap: int = 0
    dedup_batches: int = 0
    dedup_skipped: int = 0
    ad_skipped: int = 0
    ad_shifted: int = 0  # part of ad_skipped: the reply's correct contradicted the gate's record
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
    blocker_lines: list[str] = field(default_factory=list)
    seed_dir: Path | None = None
    seed_lines: list[str] = field(default_factory=list)
    ads_status: str = "no change proposed"
    live_untuned: bool = False  # no snapshot yet, so the live model wasn't built by the tuner
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
        kept = set(result.kept_ads)
        for label, ids in (("Fixed", result.fixed), ("Broken", [b for b in result.broken if b not in kept])):
            for case_id in ids:
                lines.append(f"- {label}: {case_id} ({self.case_titles.get(case_id, '')})")
        if result.kept_ads:
            lines += ["", f"Kept ads (allowed: fixes outnumber them {KEPT_AD_FIX_RATIO} to 1 or more):", ""]
            lines += [f"- Kept ad: {case_id} ({self.case_titles.get(case_id, '')})" for case_id in result.kept_ads]
        return lines

    def render(self) -> str:
        if self.no_evidence:
            extra = f" ({self.bad_lines} bad evidence lines skipped)" if self.bad_lines else ""
            seed = f" The seed in {self.seed_dir} was not used." if self.seed_dir else ""
            return f"Tuning report {self.date}: no evidence for this day, nothing to tune{extra}.{seed}\n"

        out = [f"# Tuning report {self.date}", ""]
        if self.error:
            out += [f"Applied nothing. {self.error}", ""]
        else:
            out += [f"Ads: {self.ads_status}. Dedup: {self.dedup_status}.", ""]

        out += ["## Evidence", "",
                f"- Ad-gate decisions judged: {self.judged_ads}.",
                f"- Decisions over the cap of items per run: {self.over_cap}.",
                f"- Dedup batches judged: {self.dedup_batches}.",
                f"- Bad evidence lines skipped: {self.bad_lines}.", ""]

        out += ["## Ad gate mistakes", ""]
        if not self.mistakes:
            out.append("None found.")
        for m in self.mistakes:
            label = "an ad" if m.verdict.is_ad else "not an ad"
            if m.unusable:
                second = "no usable reply, counted as a disagreement, so the change was dropped"
            elif m.second is None:
                second = "not needed, the change keeps more"
            else:
                second = "agreed" if m.second else "disagreed, so the change was dropped"
            out.append(f'- "{m.record.title}" ({m.record.source}). Gate verdict: {m.record.verdict}. '
                       f"Judge: {label} ({m.verdict.ad_type}), {m.verdict.reason}. Second opinion: {second}.")
        out += [f"- Records skipped: {self.ad_skipped} (no usable verdict: {self.ad_skipped - self.ad_shifted}; "
                f"label didn't match the gate's record: {self.ad_shifted}). They were not judged.", ""]

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

        if self.seed_dir:
            if self.ads_status == "applied":
                outcome = "Saved with the rest of the ads change."
            else:
                outcome = f"Not saved: the ads change was {self.ads_status}."
            out += ["## Seed", "", f"From {self.seed_dir}. {outcome}", ""]
            out += (self.seed_lines or ["- The seed held nothing usable."]) + [""]

        out += ["## Dedup cases", ""]
        out += self.dedup_lines or ["None."]
        out += [f"- Cases skipped: {self.dedup_skipped} (malformed: a bad kind, too few or unknown ids, "
                "or bad groups).", ""]

        out += ["## Gate results", ""]
        out += self._gate_lines("Ads", self.ad_gate, self.ads_status)
        if self.ad_gate is not None and self.live_untuned:
            out.append("The live model was not built by the tuner (no snapshot exists yet). The candidate "
                       "may also differ from it in num_ctx, the example order and the learned-rule markers, "
                       "so not every fix or break comes from the new examples.")
        out += self._gate_lines("Dedup", self.dedup_gate, self.dedup_status)
        out.append("")
        if self.blocker_lines:
            out += ["Second opinions on the cases that blocked a gate:", ""] + self.blocker_lines + [""]

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

    def _install_candidate(self, plan: _AdPlan) -> None:
        self.log(f"tune {self.date}: building and installing the candidate model")
        text = _build_text(self.config, plan.examples, plan.rules)
        self.candidate_attempted = True
        _install(self.config, text, self.candidate)

    def _ad_cases(self, records: list[AdRecord], judgement: AdJudgement,
                  mistakes: list[_Mistake]) -> tuple[list[Case], dict[str, AdRecord], set[str]]:
        """Gate cases, today's case ids mapped to their records, and the ids of
        today cases whose label the planning second opinion already disputed."""
        cases: list[Case] = []
        for prefix, examples in (("base", load_examples(BASE_EXAMPLES)), ("learned", self.store.ad_examples())):
            for e in examples:
                cases.append(Case(id=f"{prefix}:{e['id']}", title=e["title"], text=e["summary"],
                                  expected=bool(e["is_ad"]), today_mistake=False, countable=True))
        for e in self.seed_examples:
            cases.append(Case(id=f"seed:{e['id']}", title=e["title"], text=e["summary"],
                              expected=bool(e["is_ad"]), today_mistake=False, countable=True))
        disputed = {m.record.key for m in mistakes if m.second is False}
        by_key = {r.key: r for r in records}
        today: dict[str, AdRecord] = {}
        settled: set[str] = set()
        for v in judgement.verdicts:
            record = by_key[v.key]
            case_id = f"today:{record.key[:8]}"
            if record.key in disputed:
                # The two judges disagree, so trust the gate: a candidate that
                # changes this item's verdict counts as a regression.
                expected, mistake = record.verdict == "ad", False
                settled.add(case_id)
            else:
                expected, mistake = v.is_ad, not v.correct
            cases.append(Case(id=case_id, title=record.title, text=record.text_sent,
                              expected=expected, today_mistake=mistake,
                              countable=v.visible_in_excerpt if mistake else True, today=True))
            today[case_id] = record
        self.rep.case_titles.update({c.id: c.title for c in cases})
        return cases, today, settled

    def _score_ads(self, cases: list[Case]) -> tuple[list[bool], list[bool]]:
        """(current_ok, candidate_ok) for the installed live and candidate models."""
        self.log(f"tune {self.date}: ads gate, {len(cases)} cases")
        current_ok = [p == c.expected for p, c in zip(predict_ads(self.live, cases), cases, strict=True)]
        candidate_ok = [p == c.expected for p, c in zip(predict_ads(self.candidate, cases), cases, strict=True)]

        def rerun(i: int) -> tuple[bool, bool]:
            c = cases[i]
            return (predict_ads(self.live, [c])[0] == c.expected,
                    predict_ads(self.candidate, [c])[0] == c.expected)

        return current_ok, stabilize(current_ok, candidate_ok, rerun)

    def _gate_ads(self, records: list[AdRecord], judgement: AdJudgement, mistakes: list[_Mistake],
                  plan: _AdPlan) -> GateResult:
        self.rep.live_untuned = not self.store.has_snapshot()
        self._install_candidate(plan)
        cases, today, settled = self._ad_cases(records, judgement, mistakes)
        current_ok, candidate_ok = self._score_ads(cases)
        result = decide(cases, current_ok, candidate_ok, allow_keep_side_ratio=KEPT_AD_FIX_RATIO)
        expected = {c.id: c.expected for c in cases}

        def disputes(case_id: str) -> bool | None:
            is_ad = self._second_ad(today[case_id])
            return None if is_ad is None else is_ad != expected[case_id]

        # A case whose dispute the planning second opinion already settled always blocks.
        checks = {cid: disputes for cid in today if cid not in settled}
        result, gone = self._settle_blockers("Ads", result, cases, current_ok, candidate_ok, checks,
                                             keep_side_ratio=KEPT_AD_FIX_RATIO)
        if not gone:
            return result
        keys = {today[cid].key for cid in gone}
        drop = {eid for eid, key in plan.example_keys.items() if key in keys}
        if not drop or not result.accepted:
            return result

        # The applied change must be the gated one: rebuild without the disputed
        # examples and gate once more, with no second opinions this time.
        plan.examples = [e for e in plan.examples if str(e.get("id")) not in drop]
        plan.new_examples = [e for e in plan.new_examples if str(e.get("id")) not in drop]
        self.rep.blocker_lines.append(
            f"- Ads: dropped the disputed examples {', '.join(sorted(drop))} and gated the change again.")
        if not plan.new_examples and not plan.rules_changed:
            self.rep.blocker_lines.append("- Ads: nothing new is left after the drops, so nothing is applied.")
            return GateResult(False, "nothing new is left after dropping disputed examples",
                              result.current_errors, result.candidate_errors, [], [])
        self._install_candidate(plan)
        remaining = [c for c in cases if c.id not in gone]
        current_ok, candidate_ok = self._score_ads(remaining)
        return decide(remaining, current_ok, candidate_ok, allow_keep_side_ratio=KEPT_AD_FIX_RATIO)

    def _settle_blockers(self, name: str, result: GateResult, cases: list[Case], current_ok: list[bool],
                         candidate_ok: list[bool],
                         checks: dict[str, Callable[[str], bool | None]],
                         keep_side_ratio: int | None = None) -> tuple[GateResult, set[str]]:
        """When a gate rejects and every broken case is in checks, ask a blind
        second opinion about each. checks maps a case id to a call returning
        True when the second opinion disputes the case, False when it supports
        the case's label, and None for a malformed reply. A disputed case is
        dropped and decide runs again on the rest, reusing the ok lists and
        keep_side_ratio. Returns (result, disputed ids)."""
        if result.accepted or not result.broken or not all(b in checks for b in result.broken):
            return result, set()
        if len(result.broken) > MAX_BLOCKERS_FOR_SECOND_OPINION:
            self.rep.blocker_lines.append(
                f"- {name}: {len(result.broken)} cases broke, more than {MAX_BLOCKERS_FOR_SECOND_OPINION}, "
                "so the gate rejects without second opinions.")
            return result, set()
        self.log(f"tune {self.date}: second opinions on {len(result.broken)} {name.lower()} gate blockers")
        disputed: set[str] = set()
        for case_id in result.broken:
            verdict = checks[case_id](case_id)
            where = f"- {name}: {case_id} ({self.rep.case_titles.get(case_id, '')})"
            if verdict is True:
                disputed.add(case_id)
                self.rep.blocker_lines.append(f"{where}: disputed, not counted.")
            elif verdict is None:
                self.rep.blocker_lines.append(f"{where}: no usable second opinion, so it still blocks.")
            else:
                self.rep.blocker_lines.append(f"{where}: the second opinion supports the first judge's label, "
                                              "so it still blocks.")
        if not disputed:
            return result, set()
        keep = [i for i, c in enumerate(cases) if c.id not in disputed]
        result = decide([cases[i] for i in keep], [current_ok[i] for i in keep], [candidate_ok[i] for i in keep],
                        allow_keep_side_ratio=keep_side_ratio)
        return result, disputed

    def _score_dedup(self, plan: _DedupPlan) -> tuple[list[DedupCase], list[Case], list[bool], list[bool],
                                                      list[set[int]]]:
        """Score the current block and the plan's block on every case in the plan.
        Returns (cases, gate cases, current_ok, candidate_ok, candidate drops)."""
        cases = [DedupCase(titles=list(c["titles"]), expected_groups=[list(g) for g in c["expected_groups"]],
                           kind=str(c.get("kind", "")), note=str(c.get("note", ""))) for c in plan.cases]
        first_new = len(cases) - plan.new_count
        meta = [Case(id=f"dedup:{i}", title=c.titles[0] if c.titles else "", text="", expected=True,
                     today_mistake=i >= first_new, countable=True) for i, c in enumerate(cases)]
        self.rep.case_titles.update({m.id: m.title for m in meta})
        model = self.config.models.dedup_model
        current_block = self.store.dedup_examples_block()
        candidate_block = TuningStore.dedup_block_for(plan.cases)

        def ok(block: str, subset: list[DedupCase]) -> tuple[list[bool], list[set[int]]]:
            drops = predict_dedup(subset, model, block)
            return [dedup_case_ok(c, d) for c, d in zip(subset, drops, strict=True)], drops

        current_ok, _ = ok(current_block, cases)
        candidate_ok, candidate_drops = ok(candidate_block, cases)

        def rerun(i: int) -> tuple[bool, bool]:
            return ok(current_block, [cases[i]])[0][0], ok(candidate_block, [cases[i]])[0][0]

        return cases, meta, current_ok, stabilize(current_ok, candidate_ok, rerun), candidate_drops

    def _gate_dedup(self, plan: _DedupPlan) -> GateResult:
        cases, meta, current_ok, candidate_ok, candidate_drops = self._score_dedup(plan)
        result = decide(meta, current_ok, candidate_ok)

        def disputes(case_id: str) -> bool | None:
            # Disputed only when the candidate's drops are right under the second
            # opinion's own groups; any other mismatch leaves the case blocking.
            i = int(case_id.split(":")[1])
            groups = self._second_dedup_groups(cases[i])
            if groups is None:
                return None
            theirs = DedupCase(titles=cases[i].titles, expected_groups=groups, kind=cases[i].kind, note="")
            return dedup_case_ok(theirs, candidate_drops[i])

        checks = {m.id: disputes for m in meta if m.today_mistake}
        result, gone = self._settle_blockers("Dedup", result, meta, current_ok, candidate_ok, checks)
        if not gone or not result.accepted:
            return result
        # The saved block must be the gated one: drop the disputed cases and gate once more.
        drop = {int(cid.split(":")[1]) for cid in gone}
        plan.cases = [c for i, c in enumerate(plan.cases) if i not in drop]
        plan.new_count -= len(drop)
        self.rep.blocker_lines.append(f"- Dedup: dropped {len(drop)} disputed cases and gated the change again.")
        if plan.new_count <= 0:
            self.rep.blocker_lines.append("- Dedup: nothing new is left after the drops, so nothing is applied.")
            return GateResult(False, "nothing new is left after dropping disputed cases",
                              result.current_errors, result.candidate_errors, [], [])
        _, meta, current_ok, candidate_ok, _ = self._score_dedup(plan)
        return decide(meta, current_ok, candidate_ok)

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
