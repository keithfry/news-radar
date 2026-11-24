"""Tests for the learned data store."""

from __future__ import annotations

import json
import os

import pytest

from newsradar.tuning import store as store_module
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


def test_snapshot_records_parts_and_restore_touches_only_those(tmp_path):
    s = TuningStore(tmp_path)
    s.save_ad_rules("r1"); s.save_dedup_cases([{"titles": ["a", "b"], "expected_groups": []}])
    snap = s.snapshot("2026-11-23", None, ["dedup"])
    assert s.snapshot_parts(snap) == ["dedup"] and not (snap / "Modelfile").exists()
    s.save_ad_rules("r2"); s.save_dedup_cases([])
    assert s.restore(snap) is None
    assert s.ad_rules() == "r2" and [c["titles"] for c in s.dedup_cases()] == [["a", "b"]]


def test_snapshot_without_parts_file_counts_as_both(tmp_path):
    s = TuningStore(tmp_path)
    snap = s.root / "snapshots" / "2026-11-23"
    snap.mkdir(parents=True)
    assert s.snapshot_parts(snap) == ["ads", "dedup"] and s.has_snapshot()


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
    # The real batch that follows is headed "Articles:", so the examples use another label.
    assert "Example articles:\n0. T15a" in block and "\nArticles:" not in block


def test_dedup_block_for_unsaved_cases_matches_saved_block(tmp_path):
    s = TuningStore(tmp_path)
    cases = [{"titles": ["A", "A2", "B"], "expected_groups": [[0, 1]], "kind": "missed_duplicate", "added": "2026-11-23"}]
    block = TuningStore.dedup_block_for(cases)
    assert "0. A\n1. A2\n2. B" in block and s.dedup_cases() == []
    s.save_dedup_cases(cases)
    assert s.dedup_examples_block() == block


def test_dedup_block_empty_without_cases(tmp_path):
    assert TuningStore(tmp_path).dedup_examples_block() == ""


def test_domain_only():
    assert domain_only("https://thehustle.co/join?ref=1234567890") == "https://thehustle.co/"
    assert domain_only("") == ""
    assert domain_only("not a url") == ""


def test_domain_only_drops_port_and_never_raises():
    assert domain_only("http://host:8080/path?x=1") == "http://host/"
    assert domain_only("http://host:abc/") == "http://host/"
    assert domain_only("http://[::1]:8080/a") == "http://[::1]/"
    assert domain_only("http://[::1/") == ""
    assert domain_only(None) == ""


def _snap(s, name):
    path = s.root / "snapshots" / name
    path.mkdir(parents=True)
    (path / "Modelfile").write_text("m")
    return path


def _report(s, date):
    path = s.report_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("r")
    return path


def test_prune_removes_old_snapshots_and_reports(tmp_path):
    s = TuningStore(tmp_path)
    old_snap, new_snap = _snap(s, "2026-10-01"), _snap(s, "2026-11-20")
    old_report, new_report = _report(s, "2026-10-01"), _report(s, "2026-11-20")
    removed = s.prune("2026-11-23", 30)
    assert not old_snap.exists() and not old_report.exists()
    assert new_snap.exists() and new_report.exists()
    assert sorted(removed) == ["reports/2026-10-01.md", "snapshots/2026-10-01"]


def test_prune_keeps_newest_snapshot_even_when_old(tmp_path):
    s = TuningStore(tmp_path)
    older, newest = _snap(s, "2026-09-01"), _snap(s, "2026-10-01")
    s.prune("2026-11-23", 30)
    assert not older.exists() and newest.exists() and s.latest_snapshot("2026-10-01") == newest


def test_prune_reads_suffixed_snapshot_names_and_skips_unparsed(tmp_path):
    s = TuningStore(tmp_path)
    first, second = _snap(s, "2026-10-01"), _snap(s, "2026-10-01-2")
    later_old = _snap(s, "2026-10-01-10")  # newest: suffix 10 sorts after 2 by number
    odd = _snap(s, "not-a-date")
    removed = s.prune("2026-11-23", 30)
    assert not first.exists() and not second.exists() and later_old.exists() and odd.exists()
    assert sorted(removed) == ["snapshots/2026-10-01", "snapshots/2026-10-01-2"]
    assert s.latest_snapshot("2026-10-01") == later_old


def test_rollback_reaches_the_snapshot_prune_keeps(tmp_path):
    s = TuningStore(tmp_path)
    kept = [_snap(s, n) for n in ("2026-10-01", "2026-10-01-2", "2026-10-01-3")][-1]
    s.prune("2026-11-23", 30)
    assert [p.name for p in (s.root / "snapshots").iterdir()] == ["2026-10-01-3"]
    assert s.latest_snapshot("2026-10-01") == kept


def test_snapshot_after_a_gap_gets_a_higher_suffix(tmp_path):
    s = TuningStore(tmp_path)
    _snap(s, "2026-11-23-3")
    new = s.snapshot("2026-11-23", "m")
    assert new.name == "2026-11-23-4" and s.latest_snapshot("2026-11-23") == new


def test_prune_never_deletes_todays_report(tmp_path):
    s = TuningStore(tmp_path)
    today = _report(s, "2026-11-23")
    assert s.prune("2026-11-23", 0) == [] and today.exists()


def test_failed_write_leaves_the_old_file_whole_and_no_temp_file(tmp_path, monkeypatch):
    s = TuningStore(tmp_path)
    s.save_ad_examples([ex("old")])

    def killed(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(store_module.os, "replace", killed)
    with pytest.raises(OSError):
        s.save_ad_examples([ex("new")])
    assert [e["title"] for e in s.ad_examples()] == ["old"]
    assert sorted(p.name for p in s.root.iterdir()) == ["learned_ad_examples.json"]


def test_write_goes_through_a_temp_file_in_the_same_folder(tmp_path, monkeypatch):
    s = TuningStore(tmp_path)
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((os.path.dirname(src), os.path.dirname(dst)))
        real(src, dst)

    monkeypatch.setattr(store_module.os, "replace", spy)
    s.save_dedup_cases([{"titles": ["a", "b"], "expected_groups": []}])
    assert seen == [(str(s.root), str(s.root))]
    assert json.loads((s.root / "learned_dedup_cases.json").read_text())[0]["titles"] == ["a", "b"]


# ---------- safe snapshots ----------

def test_snapshot_writes_applied_json_and_modelfile_through_a_temp_file(tmp_path, monkeypatch):
    s = TuningStore(tmp_path)
    replaced = []
    real = os.replace

    def spy(src, dst):
        replaced.append(os.path.basename(dst))
        real(src, dst)

    monkeypatch.setattr(store_module.os, "replace", spy)
    snap = s.snapshot("2026-11-23", "MODELFILE", ["ads"])
    assert sorted(replaced) == ["Modelfile", "applied.json"]
    assert json.loads((snap / "applied.json").read_text()) == {"parts": ["ads"]}
    assert sorted(p.name for p in snap.iterdir()) == ["Modelfile", "applied.json"]


@pytest.mark.parametrize("text", ["{not json", "[]", '{"other": 1}', '{"parts": "ads"}', ""])
def test_damaged_applied_json_raises_bad_snapshot_error(tmp_path, text):
    s = TuningStore(tmp_path)
    snap = s.snapshot("2026-11-23", "M", ["ads"])
    (snap / "applied.json").write_text(text)
    with pytest.raises(store_module.BadSnapshotError, match="applied.json"):
        s.snapshot_parts(snap)
