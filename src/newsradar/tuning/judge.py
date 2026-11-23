"""The Opus judge: reviews the ad gate's and the dedup check's decisions.

The judge returns data only. Every reply is checked against the batch it
answers, and any failed call or bad reply raises JudgeError, so the tuner
applies nothing from a run it can't trust.
"""

from __future__ import annotations

import json
import re
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
    """A judge call failed, or its reply doesn't fit the batch it answers."""


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

_EXAMPLE_SCHEMA = _obj({"title": _STR, "summary": _STR, "source": _STR, "link": _STR, "notes": _STR})

AD_SCHEMA = _obj({
    "verdicts": {"type": "array", "items": _obj({
        "key": _STR,
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


# ---------- helpers ----------

def _prompt(name: str) -> str:
    return resources.files("newsradar.tuning").joinpath("prompts", name).read_text(encoding="utf-8")


def _call(prompt: str, system_file: str, schema: dict, tuning: TuningConfig, what: str) -> tuple[dict, float]:
    try:
        result = run_claude(prompt, tuning.judge_model, system_prompt=_prompt(system_file),
                            json_schema=schema, effort=tuning.judge_effort)
    except ClaudeCallError as exc:
        raise JudgeError(f"{what}: {exc}") from exc
    if not isinstance(result.structured, dict):
        raise JudgeError(f"{what}: reply had no structured output")
    return result.structured, result.cost_usd


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or "item"


def _record_json(record: AdRecord, with_verdict: bool) -> dict:
    out = {"key": record.key, "title": record.title, "source": record.source,
           "source_type": record.source_type, "link": record.link}
    if with_verdict:
        out["verdict"] = record.verdict
        out["reason"] = record.reason
    out["text_sent"] = record.text_sent
    out["text_full"] = record.text_full
    return out


def _check_label(is_ad, ad_type, what: str) -> None:
    if not isinstance(is_ad, bool) or ad_type not in AD_TYPES or is_ad != (ad_type != "not_ad"):
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
    return {
        "id": f"{_slug(record.source)}-{record.date}-{record.key[:8]}",
        "title": str(raw.get("title") or record.title),
        "summary": str(raw.get("summary") or "")[:SUMMARY_CHARS],
        "source": str(raw.get("source") or record.source),
        "link": domain_only(str(raw.get("link") or record.link)),
        "is_ad": verdict.is_ad,
        "ad_type": verdict.ad_type,
        "notes": str(raw.get("notes") or ""),
    }


def _parse_ad_batch(batch: list[AdRecord], reply: dict, what: str) -> tuple[list[AdVerdict], list[ProposedRule]]:
    by_key = {r.key: r for r in batch}
    raw_verdicts = reply.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise JudgeError(f"{what}: reply has no verdicts list")
    found: dict[str, AdVerdict] = {}
    for raw in raw_verdicts:
        key = raw.get("key") if isinstance(raw, dict) else None
        if key not in by_key:
            raise JudgeError(f"{what}: verdict for unknown key {key!r}")
        if key in found:
            raise JudgeError(f"{what}: more than one verdict for key {key}")
        record = by_key[key]
        _check_label(raw.get("is_ad"), raw.get("ad_type"), f"{what}, key {key}")
        correct = raw.get("correct")
        if correct is not ((record.verdict == "ad") == raw["is_ad"]):
            raise JudgeError(f"{what}, key {key}: correct={correct!r} contradicts is_ad={raw['is_ad']} "
                             f"for gate verdict {record.verdict!r}")
        v = AdVerdict(key=key, correct=correct, is_ad=raw["is_ad"], ad_type=raw["ad_type"],
                      reason=str(raw.get("reason") or ""),
                      visible_in_excerpt=bool(raw.get("visible_in_excerpt")), example=None)
        if not correct:
            if not isinstance(raw.get("example"), dict):
                raise JudgeError(f"{what}, key {key}: mistake has no example")
            v.example = _make_example(record, v, raw["example"])
        found[key] = v
    missing = [k for k in by_key if k not in found]
    if missing:
        raise JudgeError(f"{what}: no verdict for keys {', '.join(missing)}")

    rules: list[ProposedRule] = []
    for raw in reply.get("proposed_rules") or []:
        if not isinstance(raw, dict) or raw.get("leaning") not in LEANINGS or not str(raw.get("text") or "").strip():
            raise JudgeError(f"{what}: bad proposed rule {raw!r}")
        rules.append(ProposedRule(text=str(raw["text"]).strip(), leaning=raw["leaning"]))
    return [found[r.key] for r in batch], rules


def judge_ads(records: list[AdRecord], tuning: TuningConfig, batch_size: int = 20) -> AdJudgement:
    verdicts: list[AdVerdict] = []
    rules: list[ProposedRule] = []
    cost = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        what = f"ad batch (keys {', '.join(r.key for r in batch)})"
        payload = json.dumps([_record_json(r, with_verdict=True) for r in batch], indent=2, ensure_ascii=False)
        prompt = (f"Review these {len(batch)} ad-gate decisions. Return exactly one verdict per record, "
                  f"using its key.\n\nRecords:\n{payload}\n")
        reply, call_cost = _call(prompt, "judge_ads.md", AD_SCHEMA, tuning, what)
        cost += call_cost
        batch_verdicts, batch_rules = _parse_ad_batch(batch, reply, what)
        verdicts += batch_verdicts
        rules += batch_rules
    return AdJudgement(verdicts=verdicts, proposed_rules=rules, cost_usd=cost)


def second_opinion_ad(record: AdRecord, tuning: TuningConfig) -> bool:
    what = f"second opinion on ad key {record.key}"
    payload = json.dumps(_record_json(record, with_verdict=False), indent=2, ensure_ascii=False)
    prompt = f"Decide whether this item is an advertisement.\n\nItem:\n{payload}\n"
    reply, _ = _call(prompt, "second_opinion.md", SECOND_AD_SCHEMA, tuning, what)
    _check_label(reply.get("is_ad"), reply.get("ad_type"), what)
    return reply["is_ad"]


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
    cost = 0.0
    skipped = 0
    for record in records:
        what = f"dedup batch {record.key} ({record.topic})"
        payload = json.dumps(_dedup_payload(record), indent=2, ensure_ascii=False)
        prompt = (f"Review this duplicate-check batch of {len(record.titles)} titles from the "
                  f"{record.topic} topic.\n\nItems:\n{payload}\n")
        reply, call_cost = _call(prompt, "judge_dedup.md", DEDUP_SCHEMA, tuning, what)
        cost += call_cost
        raw_cases = reply.get("cases")
        if not isinstance(raw_cases, list):
            raise JudgeError(f"{what}: reply has no cases list")
        for raw in raw_cases:
            case = _parse_case(raw, record.titles, what)
            if case is None:
                skipped += 1
            else:
                cases.append(case)
    return cases, cost, skipped


def second_opinion_dedup(case: DedupCase, tuning: TuningConfig) -> bool:
    what = f"second opinion on dedup case {case.titles[:1]!r}"
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(case.titles)], indent=2, ensure_ascii=False)
    prompt = f"Group the duplicates among these {len(case.titles)} titles.\n\nTitles:\n{payload}\n"
    reply, _ = _call(prompt, "second_opinion.md", SECOND_DEDUP_SCHEMA, tuning, what)
    groups = _check_groups(reply.get("groups"), len(case.titles), what)
    return {frozenset(g) for g in groups} == {frozenset(g) for g in case.expected_groups}
