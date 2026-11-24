"""The gate and D1 orchestration for a tune run: build the candidate, score
both models, ask second opinions on the cases that block, and gate again
when a disputed case is dropped from the change.

_GateSteps is a mixin for the run class in tuner.py, which supplies config,
tuning, date, store, rep, log, live, candidate, candidate_attempted,
seed_examples and the _second_ad and _second_dedup_groups calls.
"""

from __future__ import annotations

import hashlib
from typing import Callable

from newsradar.ad_detector.update_modelfile import load_examples
from newsradar.tuning.evidence import AdRecord
from newsradar.tuning.gate import (
    KEPT_AD_FIX_RATIO,
    Case,
    GateResult,
    decide,
    dedup_case_ok,
    predict_ads,
    predict_dedup,
    stabilize,
)
from newsradar.tuning.judge import AdJudgement, DedupCase
from newsradar.tuning.models import BASE_EXAMPLES, _build_text, _install
from newsradar.tuning.report import _AdPlan, _DedupPlan, _Mistake
from newsradar.tuning.store import TuningStore

# Second opinions on gate blockers are asked only when this many cases or fewer broke.
MAX_BLOCKERS_FOR_SECOND_OPINION = 2

# However full max_gate_today_cases is, the ads gate scores at least this many
# of today's passed records (or all of them, when there are fewer).
MIN_PASSED_SCORED = 20


def _sample_passed(keys: list[str], n: int, date: str) -> set[str]:
    """n of the keys, picked by a hash seeded with date, so every run on that
    date picks the same ones whatever order the keys come in."""
    order = sorted(keys, key=lambda k: hashlib.sha256(f"{date}:{k}".encode("utf-8")).hexdigest())
    return set(order[:max(n, 0)])



class _GateSteps:
    def _install_candidate(self, plan: _AdPlan) -> None:
        self.log(f"tune {self.date}: building and installing the candidate model")
        text = _build_text(self.config, plan.examples, plan.rules)
        self.candidate_attempted = True
        _install(self.config, text, self.candidate)

    def _ad_cases(self, records: list[AdRecord], judgement: AdJudgement, mistakes: list[_Mistake],
                  plan: _AdPlan) -> tuple[list[Case], dict[str, AdRecord], set[str]]:
        """Gate cases, today's case ids mapped to their records, and the ids of
        today cases whose label the planning second opinion already disputed.

        Every base, learned and seed example is a case. Of today's records,
        every mistake, removal, settled case and record behind a new example
        is a case. Records that passed fill what is left of
        max_gate_today_cases, picked by _sample_passed, and never fewer than
        MIN_PASSED_SCORED of them."""
        cases: list[Case] = []
        for prefix, examples in (("base", load_examples(BASE_EXAMPLES)), ("learned", self.store.ad_examples())):
            for e in examples:
                cases.append(Case(id=f"{prefix}:{e['id']}", title=e["title"], text=e["summary"],
                                  expected=bool(e["is_ad"]), today_mistake=False, countable=True))
        for e in self.seed_examples:
            cases.append(Case(id=f"seed:{e['id']}", title=e["title"], text=e["summary"],
                              expected=bool(e["is_ad"]), today_mistake=False, countable=True))
        disputed = {m.record.key for m in mistakes if m.second is False}
        example_keys = set(plan.example_keys.values())
        by_key = {r.key: r for r in records}
        today: dict[str, AdRecord] = {}
        settled: set[str] = set()
        passed = {v.key for v in judgement.verdicts
                  if v.correct and v.key not in disputed and v.key not in example_keys
                  and by_key[v.key].verdict != "ad"}
        always = len(judgement.verdicts) - len(passed)
        cap = self.tuning.max_gate_today_cases
        keep = _sample_passed(sorted(passed), max(cap - always, min(MIN_PASSED_SCORED, len(passed))), self.date)
        if len(keep) < len(passed):
            self.rep.sample_line = (
                f"Ads gate: {len(passed) - len(keep)} of today's {len(passed)} passed records were sampled "
                f"out (max_gate_today_cases {cap}; the {always} mistakes, removals and records behind new "
                f"examples always count, and at least {MIN_PASSED_SCORED} passed records are scored).")
        for v in judgement.verdicts:
            if v.key in passed and v.key not in keep:
                continue
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
        cases, today, settled = self._ad_cases(records, judgement, mistakes, plan)
        current_ok, candidate_ok = self._score_ads(cases)
        result = decide(cases, current_ok, candidate_ok, allow_keep_side_ratio=KEPT_AD_FIX_RATIO)
        expected = {c.id: c.expected for c in cases}

        def disputes(case_id: str) -> bool | None:
            is_ad = self._second_ad(today[case_id])
            return None if is_ad is None else is_ad != expected[case_id]

        # A case whose dispute the planning second opinion already settled gets no
        # second opinion here, so D1 never drops it. It can still pass on the keep
        # side under the kept-ad allowance; any other break of it blocks.
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
