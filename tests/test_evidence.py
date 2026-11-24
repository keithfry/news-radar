"""Tests for the per-day ad and dedup evidence recorder."""

from __future__ import annotations

import json

import pytest

from newsradar import cli
from newsradar.config import load_config
from newsradar.tuning.evidence import EvidenceRecorder, load_day, prune_evidence

from conftest import EXAMPLE_CONFIG

DATE = "2026-11-23"


def test_records_ad_once_per_run_across_topics(tmp_path):
    rec = EvidenceRecorder(tmp_path, DATE)
    for topic in ("ai", "robotics"):
        rec.record_ad(topic=topic, item={"link": "https://x/1", "source": "S"}, title="T", text="body " * 500,
                      source_type="email", verdict="ad", reason="r", model="ad-detector")
    ads, dedups, bad = load_day(tmp_path, DATE)
    assert len(ads) == 1 and ads[0].topic == "ai"
    assert ads[0].text_sent == ("body " * 500)[:1000] and len(ads[0].text_full) <= 4000


def test_records_dedup_batch(tmp_path):
    rec = EvidenceRecorder(tmp_path, DATE)
    rec.record_dedup(topic="ai", titles=["A", "A again", "B"], dropped=[1])
    _, dedups, _ = load_day(tmp_path, DATE)
    assert dedups[0].titles == ["A", "A again", "B"] and dedups[0].dropped == [1]


def test_truncated_last_line_is_skipped_and_counted(tmp_path):
    rec = EvidenceRecorder(tmp_path, DATE)
    rec.record_ad(topic="ai", item={}, title="T", text="x", source_type="feed", verdict="not_ad", reason="", model="m")
    f = tmp_path / "evidence" / DATE / "ai.jsonl"
    f.write_text(f.read_text() + '{"kind": "ad", "title": "cut')
    ads, _, bad = load_day(tmp_path, DATE)
    assert len(ads) == 1 and bad == 1


def test_second_scan_same_day_keeps_last_record_per_key(tmp_path):
    for verdict in ("ad", "not_ad"):
        EvidenceRecorder(tmp_path, DATE).record_ad(topic="ai", item={}, title="T", text="x",
                                                   source_type="feed", verdict=verdict, reason="", model="m")
    ads, _, _ = load_day(tmp_path, DATE)
    assert [a.verdict for a in ads] == ["not_ad"]


def test_prune_removes_days_older_than_keep(tmp_path):
    for d in ("2026-10-20", "2026-10-25", DATE):
        (tmp_path / "evidence" / d).mkdir(parents=True)
    assert prune_evidence(tmp_path, DATE, 30) == ["2026-10-20"]


def test_write_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "evidence"
    blocker.write_text("a file, not a folder")
    EvidenceRecorder(tmp_path, DATE).record_dedup(topic="ai", titles=["A"], dropped=[])


@pytest.fixture
def _clean_cache():
    cli._ad_verdicts.clear()
    yield
    cli._ad_verdicts.clear()


def _scan(monkeypatch, tmp_path, item, source_type="rss"):
    config = load_config(EXAMPLE_CONFIG)
    monkeypatch.setattr(cli, "_evidence", EvidenceRecorder(tmp_path, DATE))
    removed: list[dict] = []
    cli._process_one(1, 1, item, source_type, config.topic("ai"), config, removed)
    return load_day(tmp_path, DATE)[0]


def test_scan_records_ad_decisions_when_tuning_enabled(monkeypatch, tmp_path, _clean_cache):
    monkeypatch.setattr(cli, "classify_ad", lambda *a: (True, "pitch"))
    item = {"title": "Acme platform news", "body": "Buy our thing", "link": "https://x/1", "source": "X"}
    ads = _scan(monkeypatch, tmp_path, item)
    assert [a.verdict for a in ads] == ["ad"]
    assert ads[0].source_type == "feed" and ads[0].reason == "pitch"


def test_ad_phrase_title_goes_to_the_model_and_linked_source_type(monkeypatch, tmp_path, _clean_cache):
    monkeypatch.setattr(cli, "classify_ad", lambda *a: (False, ""))
    item = {"title": "Gadgets: shop now", "link": "https://x/2", "source": "X", "_from_email_link": True}
    ads = _scan(monkeypatch, tmp_path, item)
    assert [(a.verdict, a.source_type) for a in ads] == [("not_ad", "linked_article")]


def test_load_day_skips_old_heuristic_records(tmp_path):
    day = tmp_path / "evidence" / "2026-11-20"
    day.mkdir(parents=True)
    old = {"kind": "ad", "key": "h", "date": "2026-11-20", "topic": "AI", "title": "Buy now", "link": "",
           "source": "X", "source_type": "feed", "text_sent": "", "text_full": "", "verdict": "heuristic",
           "reason": "title phrase", "model": ""}
    (day / "AI.jsonl").write_text(json.dumps(old) + "\n" + json.dumps({**old, "key": "k", "verdict": "ad"}) + "\n")
    ads, dedups, bad = load_day(tmp_path, "2026-11-20")
    assert [a.key for a in ads] == ["k"] and dedups == [] and bad == 0


def test_no_evidence_when_recorder_is_none(monkeypatch, tmp_path, _clean_cache):
    monkeypatch.setattr(cli, "classify_ad", lambda *a: (True, "pitch"))
    monkeypatch.setattr(cli, "_evidence", None)
    config = load_config(EXAMPLE_CONFIG)
    cli._process_one(1, 1, {"title": "T", "body": "b"}, "rss", config.topic("ai"), config, [])
    assert not (tmp_path / "evidence").exists()
