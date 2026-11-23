"""Tests for the learned data store."""

from __future__ import annotations

from newsradar.tuning.store import TuningStore, domain_only


def ex(title, is_ad=True, added="2026-11-01", id=None):
    return {"id": id or title, "title": title, "summary": "s", "source": "S", "link": "https://x/",
            "is_ad": is_ad, "ad_type": "not_ad", "notes": "", "added": added}


def test_store_round_trip_snapshot_and_restore(tmp_path):
    s = TuningStore(tmp_path)
    s.save_ad_examples([ex("a", True)]); s.save_ad_rules("r1")
    snap = s.snapshot("2026-11-23", "MODELFILE-1")
    s.save_ad_examples([]); s.save_ad_rules("r2")
    assert s.restore(snap) == "MODELFILE-1"
    assert [e["title"] for e in s.ad_examples()] == ["a"] and s.ad_rules() == "r1"


def test_missing_files_read_empty(tmp_path):
    s = TuningStore(tmp_path / "nope")
    assert s.ad_examples() == [] and s.ad_rules() == "" and s.dedup_cases() == []
    assert s.dedup_examples_block() == "" and s.latest_snapshot("2026-11-23") is None


def test_snapshot_same_day_gets_suffix_and_latest(tmp_path):
    s = TuningStore(tmp_path)
    a = s.snapshot("2026-11-23", "one")
    b = s.snapshot("2026-11-23", "two")
    c = s.snapshot("2026-11-23", "three")
    assert (a.name, b.name, c.name) == ("2026-11-23", "2026-11-23-2", "2026-11-23-3")
    assert s.latest_snapshot("2026-11-23") == c
    assert s.latest_snapshot("2026-11-24") is None


def test_restore_removes_files_absent_from_snapshot(tmp_path):
    s = TuningStore(tmp_path)
    snap = s.snapshot("2026-11-23", "M")
    s.save_ad_rules("later")
    s.save_dedup_cases([{"titles": ["a"], "expected_groups": []}])
    s.restore(snap)
    assert s.ad_rules() == "" and s.dedup_cases() == []


def test_report_path(tmp_path):
    assert TuningStore(tmp_path).report_path("2026-11-23") == tmp_path / "tuning" / "reports" / "2026-11-23.md"


def test_dedup_block_formats_newest_twelve(tmp_path):
    s = TuningStore(tmp_path)
    cases = [{"titles": [f"T{i}a", f"T{i}b", f"T{i}c"], "expected_groups": [[0, 1]], "added": f"2026-11-{i:02d}"}
             for i in range(1, 16)]
    s.save_dedup_cases(cases)
    block = s.dedup_examples_block()
    assert block.startswith("LEARNED EXAMPLES (past mistakes to avoid):\n")
    assert "T15a" in block and "T4a" in block and "T3a" not in block
    assert "0. T15a" in block and "[[0, 1]]" in block


def test_dedup_block_empty_without_cases(tmp_path):
    assert TuningStore(tmp_path).dedup_examples_block() == ""


def test_domain_only():
    assert domain_only("https://thehustle.co/join?ref=6693394292") == "https://thehustle.co/"
    assert domain_only("") == ""
    assert domain_only("not a url") == ""
