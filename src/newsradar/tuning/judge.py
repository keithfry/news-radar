"""The Opus judge: reviews the ad gate's and the dedup check's decisions.

The judge returns data only. Every reply is checked against the batch it
answers. A failed call or a broken reply raises JudgeError, so the tuner
applies nothing from a run it can't trust. A record the judge skipped or
named with a wrong id is left unjudged and counted. JudgeError carries the
cost of every call made before the failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources

from newsradar.claude_cli import ClaudeCallError, run_claude
from newsradar.config import TuningConfig
from newsradar.tuning.evidence import AdRecord, DedupRecord
from newsradar.tuning.store import domain_only

SUMMARY_CHARS = 200
AD_TYPES = [
    "referral_program", "lead_gen_offer", "newsletter_cta", "subscription_pitch",
    "sponsor_content", "product_feature_disguised", "not_ad",
]
LEANINGS = ["keep", "remove"]
CASE_KINDS = ["wrong_merge", "missed_duplicate"]


class JudgeError(RuntimeError):
    """A judge call failed, or its reply doesn't fit the batch it answers.

    cost_usd is what the calls made before the failure cost, including a call
    whose reply then failed the checks."""

    def __init__(self, message: str, cost_usd: float = 0.0):
        super().__init__(message)
        self.cost_usd = cost_usd


@dataclass
class AdVerdict:
    key: str
    correct: bool
    is_ad: bool
    ad_type: str
    reason: str
    visible_in_excerpt: bool
    example: dict | None


@dataclass
class ProposedRule:
    text: str
    leaning: str  # "keep" makes the gate keep more, "remove" makes it remove more


@dataclass
class AdJudgement:
    verdicts: list[AdVerdict]
    proposed_rules: list[ProposedRule]
    cost_usd: float
    skipped: int = 0  # records left unjudged, including the shifted ones
    shifted: int = 0  # skipped because the reply's correct contradicted the gate's record


@dataclass
class DedupCase:
    titles: list[str]
    expected_groups: list[list[int]]
    kind: str  # "wrong_merge" or "missed_duplicate"
    note: str


# ---------- JSON schemas for --json-schema ----------

def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_AD_TYPE = {"type": "string", "enum": AD_TYPES}
_GROUPS = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}

# Title, source and link come from the record, so the model writes only these.
_EXAMPLE_SCHEMA = _obj({"summary": _STR, "notes": _STR})

AD_SCHEMA = _obj({
    "verdicts": {"type": "array", "items": _obj({
        "id": _STR,
        "correct": _BOOL,
        "is_ad": _BOOL,
        "ad_type": _AD_TYPE,
        "reason": _STR,
        "visible_in_excerpt": _BOOL,
        "example": {"anyOf": [_EXAMPLE_SCHEMA, {"type": "null"}]},
    })},
    "proposed_rules": {"type": "array", "items": _obj({
        "text": _STR,
        "leaning": {"type": "string", "enum": LEANINGS},
    })},
})

DEDUP_SCHEMA = _obj({
    "cases": {"type": "array", "items": _obj({
        "kind": {"type": "string", "enum": CASE_KINDS},
        "ids": {"type": "array", "items": {"type": "integer"}},
        "expected_groups": _GROUPS,
        "note": _STR,
    })},
})

SECOND_AD_SCHEMA = _obj({"is_ad": _BOOL, "ad_type": _AD_TYPE, "reason": _STR})
SECOND_DEDUP_SCHEMA = _obj({"groups": _GROUPS, "reason": _STR})
SECOND_RULE_SCHEMA = _obj({"sound": _BOOL, "would_remove": {"type": "array", "items": {"type": "integer"}},
                           "reason": _STR})
RULE_CHECK_ITEMS = 20


# ---------- helpers ----------

def _prompt(name: str) -> str:
    return resources.files("newsradar.tuning").joinpath("prompts", name).read_text(encoding="utf-8")


class _Meter:
    """Running cost of one public call, so a JudgeError can report it."""

    def __init__(self):
        self.cost = 0.0


@contextmanager
def _metered() -> Iterator[_Meter]:
    meter = _Meter()
    try:
        yield meter
    except JudgeError as exc:
        exc.cost_usd += meter.cost
        raise


def _call(prompt: str, system_file: str, schema: dict, tuning: TuningConfig, what: str,
          meter: _Meter) -> dict:
    """One judge call. Its cost goes into meter before the reply is checked."""
    try:
        result = run_claude(prompt, tuning.judge_model, system_prompt=_prompt(system_file),
                            json_schema=schema, effort=tuning.judge_effort)
    except ClaudeCallError as exc:
        raise JudgeError(f"{what}: {exc}") from exc
    meter.cost += result.cost_usd
    if not isinstance(result.structured, dict):
        raise JudgeError(f"{what}: reply had no structured output")
    return result.structured


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or "item"


def _record_json(record: AdRecord, with_verdict: bool, id: str | None = None) -> dict:
    out = {"id": id} if id is not None else {}
    out |= {"title": record.title, "source": record.source,
            "source_type": record.source_type, "link": record.link}
    if with_verdict:
        out["verdict"] = record.verdict
        out["reason"] = record.reason
    out["text_sent"] = record.text_sent
    out["text_full"] = record.text_full
    return out


def _label_ok(is_ad, ad_type) -> bool:
    return isinstance(is_ad, bool) and ad_type in AD_TYPES and is_ad == (ad_type != "not_ad")


def _check_label(is_ad, ad_type, what: str) -> None:
    if not _label_ok(is_ad, ad_type):
        raise JudgeError(f"{what}: is_ad={is_ad!r} doesn't fit ad_type={ad_type!r}")


def _check_groups(groups, size: int, what: str) -> list[list[int]]:
    if not isinstance(groups, list):
        raise JudgeError(f"{what}: groups must be a list")
    seen: set[int] = set()
    out: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            raise JudgeError(f"{what}: group {group!r} needs at least 2 members")
        for i in group:
            if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < size:
                raise JudgeError(f"{what}: index {i!r} out of range for {size} titles")
            if i in seen:
                raise JudgeError(f"{what}: index {i} appears more than once")
            seen.add(i)
        out.append(list(group))
    return out


# ---------- ads ----------

def _make_example(record: AdRecord, verdict: AdVerdict, raw: dict) -> dict:
    """A labeled example for the gate. Title, source and link always come from
    the record; raw supplies only summary and notes. An empty summary falls
    back to the start of text_sent."""
    return {
        "id": f"{_slug(record.source)}-{record.date}-{record.key[:8]}",
        "title": record.title,
        "summary": str(raw.get("summary") or record.text_sent)[:SUMMARY_CHARS],
        "source": record.source,
        "link": domain_only(record.link),
        "is_ad": verdict.is_ad,
        "ad_type": verdict.ad_type,
        "notes": str(raw.get("notes") or ""),
    }


def _parse_ad_batch(batch: list[AdRecord], reply: dict,
                    what: str) -> tuple[list[AdVerdict], list[ProposedRule], int, int]:
    """Map a reply's r0..rN ids back to the batch. Returns (verdicts in batch
    order, rules, records skipped, of those skipped for a shifted verdict). A record is skipped when it has no verdict,
    its id is repeated, its is_ad and ad_type disagree, or the reply's
    `correct` contradicts the one derived from the label and the gate's
    verdict. That last check catches verdicts shifted onto a neighbouring
    record. A verdict with an unknown id is ignored. The derived `correct` is
    the one used. A mistake without an example gets one built from the record."""
    by_id = {f"r{i}": r for i, r in enumerate(batch)}
    raw_verdicts = reply.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise JudgeError(f"{what}: reply has no verdicts list")
    grouped: dict[str, list[dict]] = {}
    for raw in raw_verdicts:
        rid = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(rid, str) and rid in by_id:
            grouped.setdefault(rid, []).append(raw)

    found: dict[str, AdVerdict] = {}
    shifted = 0
    for rid, raws in grouped.items():
        if len(raws) > 1:
            continue
        raw, record = raws[0], by_id[rid]
        if not _label_ok(raw.get("is_ad"), raw.get("ad_type")):
            continue
        correct = raw["is_ad"] == (record.verdict == "ad")
        if isinstance(raw.get("correct"), bool) and raw["correct"] != correct:
            shifted += 1  # the verdict likely belongs to another record
            continue
        v = AdVerdict(key=record.key, correct=correct, is_ad=raw["is_ad"], ad_type=raw["ad_type"],
                      reason=str(raw.get("reason") or ""),
                      visible_in_excerpt=bool(raw.get("visible_in_excerpt")), example=None)
        if not correct:
            example = raw.get("example")
            v.example = _make_example(record, v, example if isinstance(example, dict) else {})
        found[rid] = v

    skipped = len(batch) - len(found)
    if skipped * 2 > len(batch):
        raise JudgeError(f"{what}: no usable verdict for {skipped} of {len(batch)} records")

    rules: list[ProposedRule] = []
    for raw in reply.get("proposed_rules") or []:
        if not isinstance(raw, dict) or raw.get("leaning") not in LEANINGS or not str(raw.get("text") or "").strip():
            raise JudgeError(f"{what}: bad proposed rule {raw!r}")
        rules.append(ProposedRule(text=str(raw["text"]).strip(), leaning=raw["leaning"]))
    return [found[rid] for rid in by_id if rid in found], rules, skipped, shifted


def judge_ads(records: list[AdRecord], tuning: TuningConfig, batch_size: int = 20) -> AdJudgement:
    verdicts: list[AdVerdict] = []
    rules: list[ProposedRule] = []
    skipped = shifted = 0
    starts = range(0, len(records), batch_size)
    with _metered() as meter:
        for n, start in enumerate(starts, 1):
            batch = records[start:start + batch_size]
            what = f"ad batch {n} of {len(starts)}"
            payload = json.dumps([_record_json(r, with_verdict=True, id=f"r{i}") for i, r in enumerate(batch)],
                                 indent=2, ensure_ascii=False)
            prompt = (f"Review these {len(batch)} ad-gate decisions. Return exactly one verdict per record, "
                      f"using its id.\n\nRecords:\n{payload}\n")
            reply = _call(prompt, "judge_ads.md", AD_SCHEMA, tuning, what, meter)
            batch_verdicts, batch_rules, batch_skipped, batch_shifted = _parse_ad_batch(batch, reply, what)
            verdicts += batch_verdicts
            rules += batch_rules
            skipped += batch_skipped
            shifted += batch_shifted
    return AdJudgement(verdicts=verdicts, proposed_rules=rules, cost_usd=meter.cost, skipped=skipped,
                       shifted=shifted)


def second_opinion_ad(record: AdRecord, tuning: TuningConfig) -> tuple[bool | None, float]:
    """Returns (is_ad, cost_usd) from a call that doesn't see the first verdict.
    is_ad is None when the reply's is_ad and ad_type disagree: the caller
    treats that as a disagreement. A failed call still raises JudgeError."""
    what = f"second opinion on ad {record.key[:8]}"
    payload = json.dumps(_record_json(record, with_verdict=False), indent=2, ensure_ascii=False)
    prompt = f"Decide whether this item is an advertisement.\n\nItem:\n{payload}\n"
    with _metered() as meter:
        reply = _call(prompt, "second_opinion.md", SECOND_AD_SCHEMA, tuning, what, meter)
    if not _label_ok(reply.get("is_ad"), reply.get("ad_type")):
        return None, meter.cost
    return reply["is_ad"], meter.cost


def second_opinion_rule(rule_text: str, records: list[AdRecord], tuning: TuningConfig) -> tuple[bool, float]:
    """Check a remove-leaning rule against today's news items (records the judge
    labeled not ads). Returns (approved, cost_usd). Approved only when the call
    finds the rule sound and says it would remove none of the items. The call
    sees neither the first judge's reasoning nor the gate's verdicts."""
    items = records[:RULE_CHECK_ITEMS]
    what = f"second opinion on rule {rule_text[:60]!r}"
    payload = json.dumps(
        {"rule": rule_text,
         "items": [{"id": i, "title": r.title, "source": r.source, "text_sent": r.text_sent}
                   for i, r in enumerate(items)]},
        indent=2, ensure_ascii=False)
    prompt = (f"Decide whether this rule is safe to add to the ad gate, checking it against "
              f"{len(items)} news items.\n\n{payload}\n")
    with _metered() as meter:
        reply = _call(prompt, "second_opinion_rule.md", SECOND_RULE_SCHEMA, tuning, what, meter)
        sound, would_remove = reply.get("sound"), reply.get("would_remove")
        if not isinstance(sound, bool) or not isinstance(would_remove, list):
            raise JudgeError(f"{what}: reply needs a boolean sound and a would_remove list")
    # Any entry in would_remove, even an id that matches no item, rejects the
    # rule: the cautious reading of a reply that names an item to remove.
    return sound and not would_remove, meter.cost


# ---------- dedup ----------

def _title_key(title: str) -> str:
    # Matches the scan's exact-title pre-pass (llm._exact_title_duplicates).
    return " ".join(title.split()).casefold()


def _dedup_payload(record: DedupRecord) -> list[dict]:
    dropped = set(record.dropped)
    surviving = {_title_key(t) for i, t in enumerate(record.titles) if i not in dropped}
    return [{"id": i, "title": t, "dropped": i in dropped,
             "exact_title_drop": i in dropped and bool(_title_key(t)) and _title_key(t) in surviving}
            for i, t in enumerate(record.titles)]


def _parse_case(raw, batch_titles: list[str], what: str) -> DedupCase | None:
    """Build a case from batch ids. Returns None when an id is unknown or repeated."""
    if not isinstance(raw, dict) or raw.get("kind") not in CASE_KINDS:
        raise JudgeError(f"{what}: bad case {raw!r}")
    ids = raw.get("ids")
    if not isinstance(ids, list) or len(ids) < 2:
        raise JudgeError(f"{what}: a case needs at least 2 ids")
    if (len(set(map(repr, ids))) != len(ids)
            or not all(type(i) is int and 0 <= i < len(batch_titles) for i in ids)):
        return None
    groups = _check_groups(raw.get("expected_groups"), len(ids), what)
    if raw["kind"] == "missed_duplicate" and not groups:
        raise JudgeError(f"{what}: a missed_duplicate case needs a duplicate group")
    return DedupCase(titles=[batch_titles[i] for i in ids], expected_groups=groups, kind=raw["kind"],
                     note=str(raw.get("note") or ""))


def judge_dedup(records: list[DedupRecord], tuning: TuningConfig) -> tuple[list[DedupCase], float, int]:
    """Returns (cases, cost_usd, skipped): skipped counts cases dropped for an unknown or repeated id."""
    cases: list[DedupCase] = []
    skipped = 0
    with _metered() as meter:
        for record in records:
            what = f"dedup batch {record.key} ({record.topic})"
            payload = json.dumps(_dedup_payload(record), indent=2, ensure_ascii=False)
            prompt = (f"Review this duplicate-check batch of {len(record.titles)} titles from the "
                      f"{record.topic} topic.\n\nItems:\n{payload}\n")
            reply = _call(prompt, "judge_dedup.md", DEDUP_SCHEMA, tuning, what, meter)
            raw_cases = reply.get("cases")
            if not isinstance(raw_cases, list):
                raise JudgeError(f"{what}: reply has no cases list")
            for raw in raw_cases:
                case = _parse_case(raw, record.titles, what)
                if case is None:
                    skipped += 1
                else:
                    cases.append(case)
    return cases, meter.cost, skipped


def second_opinion_dedup_groups(case: DedupCase, tuning: TuningConfig) -> tuple[list[list[int]] | None, float]:
    """Returns (groups, cost_usd): the duplicate groups a blind call finds among
    the case's titles. groups is None when the reply's groups are malformed (an
    index out of range or repeated). A failed call still raises JudgeError."""
    what = f"second opinion on dedup case {case.titles[:1]!r}"
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(case.titles)], indent=2, ensure_ascii=False)
    prompt = f"Group the duplicates among these {len(case.titles)} titles.\n\nTitles:\n{payload}\n"
    with _metered() as meter:
        reply = _call(prompt, "second_opinion.md", SECOND_DEDUP_SCHEMA, tuning, what, meter)
    try:
        return _check_groups(reply.get("groups"), len(case.titles), what), meter.cost
    except JudgeError:
        return None, meter.cost


def second_opinion_dedup(case: DedupCase, tuning: TuningConfig) -> tuple[bool | None, float]:
    """Returns (agrees, cost_usd): agrees when the call finds exactly the case's
    groups. agrees is None when the reply's groups are malformed: the caller
    treats that as a disagreement. A failed call still raises JudgeError."""
    groups, cost = second_opinion_dedup_groups(case, tuning)
    if groups is None:
        return None, cost
    return {frozenset(g) for g in groups} == {frozenset(g) for g in case.expected_groups}, cost
