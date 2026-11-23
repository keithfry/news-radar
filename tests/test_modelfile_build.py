"""Tests for building the live ad-detector Modelfile."""

from __future__ import annotations

import importlib
from pathlib import Path

from newsradar.ad_detector.update_modelfile import (
    LEARNED_RULES_END,
    LEARNED_RULES_START,
    build_live_modelfile,
    load_examples,
)

um = importlib.import_module("newsradar.ad_detector.update_modelfile")
PKG = Path(um.__file__).parent
BASE = (PKG / "AdDetectorModelfile").read_text(encoding="utf-8")
BASE_EXAMPLES = load_examples(Path(__file__).parent / "test_data" / "labeled_examples.json")


def ex(title, is_ad=True, added="2026-11-01", id=None):
    return {"id": id or title, "title": title, "summary": "sum", "source": "S", "link": "https://x/",
            "is_ad": is_ad, "ad_type": "not_ad", "notes": "n", "added": added}


LEARNED = [ex("la", True, "2026-11-02"), ex("lb", False, "2026-11-03")]


def test_live_build_includes_all_base_and_newest_learned_per_class():
    learned = [ex(f"n{i}", is_ad=False, added=f"2026-11-{i:02d}") for i in range(1, 20)]
    text = build_live_modelfile(BASE, BASE_EXAMPLES, learned, "", per_class=12)
    assert all(repr(e["title"]) in text for e in BASE_EXAMPLES)
    assert "'n19'" in text and "'n8'" in text and "'n7'" not in text


def test_live_build_is_deterministic():
    assert build_live_modelfile(BASE, BASE_EXAMPLES, LEARNED, "r", 12) == build_live_modelfile(BASE, BASE_EXAMPLES, LEARNED, "r", 12)


def test_rules_go_between_markers_and_hand_patterns_are_untouched():
    text = build_live_modelfile(BASE, BASE_EXAMPLES, [], "Judge the whole item.", 12)
    start, end = text.index(LEARNED_RULES_START), text.index(LEARNED_RULES_END)
    assert "Judge the whole item." in text[start:end]
    assert text.split(LEARNED_RULES_START)[0] == BASE.split(LEARNED_RULES_START)[0]


def test_empty_rules_leave_markers_adjacent():
    text = build_live_modelfile(BASE, BASE_EXAMPLES, [], "", 12)
    assert LEARNED_RULES_START + LEARNED_RULES_END in text


def test_packaged_modelfile_has_ctx_8192_and_markers():
    assert "PARAMETER num_ctx 8192" in BASE and LEARNED_RULES_START in BASE and LEARNED_RULES_END in BASE


def test_update_modelfile_still_regenerates_examples(tmp_path):
    import json
    data = tmp_path / "d.json"
    data.write_text(json.dumps({"examples": [ex("Only One Title")]}))
    mf = tmp_path / "Modelfile"
    mf.write_text(BASE)
    um.update_modelfile(data_path=data, modelfile_path=mf)
    out = mf.read_text()
    assert "'Only One Title'" in out and out.count("Input: Title:") == 1
    assert LEARNED_RULES_START in out
