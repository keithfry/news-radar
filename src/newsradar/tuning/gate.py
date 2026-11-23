"""Regression gate: score the live and candidate models on labeled cases and
accept a change only when it fixes a real mistake without breaking anything."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from newsradar import llm
from newsradar.llm import classify_ad
from newsradar.tuning.judge import DedupCase

# Returned in place of a drop set when the model call fails. No case expects
# index -1 to be dropped, so dedup_case_ok always scores it wrong.
_FAILED = {-1}


@dataclass
class Case:
    id: str
    title: str
    text: str
    expected: bool
    today_mistake: bool
    countable: bool  # false when the judge said the excerpt doesn't show the answer


@dataclass
class GateResult:
    accepted: bool
    reason: str
    current_errors: int
    candidate_errors: int
    fixed: list[str]
    broken: list[str]


def predict_ads(model: str, cases: list[Case]) -> list[bool]:
    """Call the model fresh for each case (no verdict cache). A call that
    raises returns the opposite of the expected answer, so it counts as wrong."""
    out: list[bool] = []
    for c in cases:
        try:
            is_ad, _reason = classify_ad(c.title, c.text[:1000], model)
            out.append(bool(is_ad))
        except Exception:
            out.append(not c.expected)
    return out


def stabilize(
    current: list[bool],
    candidate: list[bool],
    rerun: Callable[[int], tuple[bool, bool]],
) -> list[bool]:
    """Keep a candidate difference only when a rerun of both models repeats it.
    Unconfirmed differences, and reruns that raise, reset to the current value."""
    out = list(candidate)
    for i, (cur, cand) in enumerate(zip(current, candidate)):
        if cur == cand:
            continue
        try:
            cur2, cand2 = rerun(i)
        except Exception:
            out[i] = cur
            continue
        if not (cur2 == cur and cand2 == cand):
            out[i] = cur
    return out


def decide(cases: list[Case], current_ok: list[bool], candidate_ok: list[bool]) -> GateResult:
    fixed: list[str] = []
    broken: list[str] = []
    fixed_today = False
    for c, cur, cand in zip(cases, current_ok, candidate_ok):
        if not cur and cand:
            fixed.append(c.id)
            if c.today_mistake and c.countable:
                fixed_today = True
        elif cur and not cand:
            broken.append(c.id)
    current_errors = sum(1 for ok in current_ok if not ok)
    candidate_errors = sum(1 for ok in candidate_ok if not ok)

    if broken:
        accepted, reason = False, f"regression on {len(broken)} case(s): {', '.join(broken)}"
    elif not fixed_today:
        accepted, reason = False, "no fix for a countable mistake from today"
    elif candidate_errors >= current_errors:
        accepted, reason = False, "candidate does not have fewer errors than the current model"
    else:
        accepted, reason = True, f"fixed {len(fixed)} case(s), broke none"
    return GateResult(accepted, reason, current_errors, candidate_errors, fixed, broken)


def dedup_case_ok(case: DedupCase, dropped: set[int]) -> bool:
    """True when each expected group keeps exactly one member and nothing else drops."""
    expected_drops: set[int] = set()
    for group in case.expected_groups:
        if len(group) < 2:
            continue
        kept = [i for i in group if i not in dropped]
        if len(kept) != 1:
            return False
        expected_drops.update(i for i in group if i in dropped)
    return dropped == expected_drops


def predict_dedup(cases: list[DedupCase], model: str, examples_block: str) -> list[set[int]]:
    """Run the voted dedup per case with the given learned block, then restore
    the previous block. A failed call returns a drop set that scores wrong."""
    previous = llm.learned_dedup_examples()
    llm.set_learned_dedup_examples(examples_block)
    try:
        out: list[set[int]] = []
        for c in cases:
            batch = [(i, {"title": t}) for i, t in enumerate(c.titles)]
            try:
                out.append(set(llm._dedup_voted(batch, model)))
            except Exception:
                out.append(set(_FAILED))
        return out
    finally:
        llm.set_learned_dedup_examples(previous)
