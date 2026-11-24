"""The tuner's run state and the daily report it writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from newsradar.tuning.evidence import AdRecord
from newsradar.tuning.gate import KEPT_AD_FIX_RATIO, GateResult
from newsradar.tuning.judge import AdVerdict


@dataclass
class TuneOutcome:
    ads_applied: bool
    dedup_applied: bool
    summary: str
    report_path: Path



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
    sample_line: str = ""  # how many of today's passed records the ads gate left out
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
            lines += ["", f"Kept ads (allowed: fixes outside the seed outnumber them {KEPT_AD_FIX_RATIO} to 1 "
                      "or more):", ""]
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
        if self.ad_gate is not None and self.sample_line:
            out.append(self.sample_line)
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
