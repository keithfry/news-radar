"""Regression gate: score the live and candidate models on labeled cases and
accept a change only when it fixes a real mistake without breaking anything."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from newsradar import llm
from newsradar.llm import classify_ad
from newsradar.tuning.judge import DedupCase



class GateError(Exception):
    """A model call failed, so the gate cannot decide. The caller applies nothing."""


@dataclass
class Case:
    id: str
    title: str
    text: str
    expected: bool
    today_mistake: bool
    countable: bool  # false when the judge said the excerpt doesn't show the answer
    today: bool = False  # one of today's judged items, not a base, learned or seed example


@dataclass
class GateResult:
    accepted: bool
    reason: str
    current_errors: int
    candidate_errors: int
    fixed: list[str]
    broken: list[str]
    kept_ads: list[str] = field(default_factory=list)  # broken ads the gate allowed through


def predict_ads(model: str, cases: list[Case]) -> list[bool]:
    """Call the model fresh for each case (no verdict cache). Raises GateError
    when a call raises or classify_ad fails open with a parse error."""
    out: list[bool] = []
    for c in cases:
        try:
            is_ad, reason = classify_ad(c.title, c.text[:1000], model)
        except Exception as exc:
            raise GateError(f"ad call failed for {c.id}: {exc}") from exc
        if reason == "parse error":
            raise GateError(f"ad call returned a parse error for {c.id}")
        out.append(bool(is_ad))
    return out


def stabilize(
    current: list[bool],
    candidate: list[bool],
    rerun: Callable[[int], tuple[bool, bool]],
) -> list[bool]:
    """Keep a candidate difference only when a rerun of both models repeats it.
    Unconfirmed differences reset to the current value. A rerun that raises
    propagates as GateError."""
    out = list(candidate)
    for i, (cur, cand) in enumerate(zip(current, candidate, strict=True)):
        if cur == cand:
            continue
        try:
            cur2, cand2 = rerun(i)
        except GateError:
            raise
        except Exception as exc:
            raise GateError(f"rerun failed for case {i}: {exc}") from exc
        if not (cur2 == cur and cand2 == cand):
            out[i] = cur
    return out


def decide(cases: list[Case], current_ok: list[bool], candidate_ok: list[bool],
           allow_keep_side_ratio: int | None = None) -> GateResult:
    """Accept only when nothing breaks, a countable mistake from today is fixed,
    and the candidate has fewer errors.

    With allow_keep_side_ratio set, a today case labeled an ad that the
    candidate now keeps doesn't block, as long as every broken case is one of
    those and there are at least that many fixes per kept ad. Any other break
    blocks: news the candidate now removes, or any base, learned or seed case."""
    fixed: list[str] = []
    broken: list[Case] = []
    fixed_today = False
    for c, cur, cand in zip(cases, current_ok, candidate_ok, strict=True):
        if not cur and cand:
            fixed.append(c.id)
            if c.today_mistake and c.countable:
                fixed_today = True
        elif cur and not cand:
            broken.append(c)
    broken_ids = [c.id for c in broken]
    current_errors = sum(1 for ok in current_ok if not ok)
    candidate_errors = sum(1 for ok in candidate_ok if not ok)

    # A broken case labeled an ad means the candidate now keeps it.
    kept = [c for c in broken if c.today and c.expected] if allow_keep_side_ratio is not None else []
    only_kept = bool(broken) and len(kept) == len(broken)
    if broken and not (only_kept and len(fixed) >= allow_keep_side_ratio * len(kept)):
        reason = f"regression on {len(broken)} case(s): {', '.join(broken_ids)}"
        if only_kept:
            reason += (f"; kept ads pass only when fixes outnumber them {allow_keep_side_ratio} to 1 "
                       f"({len(fixed)} fixed, {len(kept)} kept)")
        accepted = False
    elif not fixed_today:
        accepted, reason = False, "no fix for a countable mistake from today"
    elif candidate_errors >= current_errors:
        accepted, reason = False, "candidate does not have fewer errors than the current model"
    elif kept:
        names = ", ".join(f"{c.id} ({c.title})" for c in kept)
        noun = "kept ad" if len(kept) == 1 else "kept ads"
        accepted, reason = True, f"fixed {len(fixed)} case(s), with {len(kept)} {noun}: {names}"
    else:
        accepted, reason = True, f"fixed {len(fixed)} case(s), broke none"
    return GateResult(accepted, reason, current_errors, candidate_errors, fixed, broken_ids,
                      [c.id for c in kept] if accepted else [])


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
    the previous block. Raises GateError when a call fails."""
    previous = llm.learned_dedup_examples()
    llm.set_learned_dedup_examples(examples_block)
    try:
        out: list[set[int]] = []
        for c in cases:
            batch = [(i, {"title": t}) for i, t in enumerate(c.titles)]
            try:
                out.append(set(llm._dedup_voted(batch, model)))
            except Exception as exc:
                raise GateError(f"dedup call failed: {exc}") from exc
        return out
    finally:
        llm.set_learned_dedup_examples(previous)
