"""Tests for building the live ad-detector Modelfile."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from newsradar.ad_detector.update_modelfile import (
    LEARNED_RULES_END,
    LEARNED_RULES_START,
    TRAINING_SECTION_START,
    build_live_modelfile,
    format_training_block,
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


def test_packaged_modelfile_has_ctx_12288_and_markers():
    assert "PARAMETER num_ctx 12288" in BASE and LEARNED_RULES_START in BASE and LEARNED_RULES_END in BASE


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


def _old_format(examples):
    """format_training_block as it was before notes were escaped."""
    ads = [e for e in examples if e["is_ad"]]
    not_ads = [e for e in examples if not e["is_ad"]]
    interleaved = []
    for i in range(max(len(ads), len(not_ads))):
        interleaved += ads[i:i + 1] + not_ads[i:i + 1]
    lines = [TRAINING_SECTION_START]
    for e in interleaved:
        reason = e.get("notes", "").strip() or e.get("ad_type", "")
        label = f'{{"is_ad": {"true" if e["is_ad"] else "false"}, "reason": "{reason}"}}'
        lines.append(f"Input: Title: {e['title']!r} | Summary: {e['summary'][:200]!r}\nOutput: {label}\n")
    return "\n".join(lines)


def test_base_examples_render_byte_for_byte_as_before():
    assert format_training_block(BASE_EXAMPLES) == _old_format(BASE_EXAMPLES)


def test_learned_note_is_capped_and_escaped_as_json():
    note = 'He said "buy" \\ now\nand ' + "x" * 300
    text = format_training_block([dict(ex("t", is_ad=False), notes=note)])
    output = next(line for line in text.splitlines() if line.startswith("Output: "))
    label = json.loads(output.removeprefix("Output: "))
    assert label == {"is_ad": False, "reason": note[:200]}
    live = build_live_modelfile(BASE, BASE_EXAMPLES, [dict(ex("t", is_ad=False), notes=note)], "", 12)
    assert live.count('"""') == BASE.count('"""')


# ---------- the context guard ----------

def _modelfile(system: str, num_ctx: int | None = 1000) -> str:
    param = f"PARAMETER num_ctx {num_ctx}\n" if num_ctx is not None else ""
    return f'FROM some-model\n{param}SYSTEM """\n{system}\n"""\n'


def test_context_guard_passes_under_seventy_percent_of_num_ctx():
    # 2,796 chars is about 699 tokens, under 700 of num_ctx 1000.
    tokens = um.check_context_budget(_modelfile("x" * 2794))
    assert 690 <= tokens <= 700


def test_context_guard_raises_over_seventy_percent_of_num_ctx():
    try:
        um.check_context_budget(_modelfile("x" * 2900))
    except um.ModelfileTooLargeError as exc:
        msg = str(exc)
        assert "num_ctx 1000" in msg and "70%" in msg and "learned_examples_per_class" in msg
    else:
        raise AssertionError("no error for a SYSTEM block over the budget")


def test_context_guard_counts_the_training_examples_inside_system():
    small = build_live_modelfile(BASE, BASE_EXAMPLES, [], "", 24)
    learned = [ex(f"t{i}" * 40, i % 2 == 0, f"2026-11-{1 + i % 28:02d}", id=f"id{i}") for i in range(48)]
    for e in learned:
        e["summary"] = "s" * 200
        e["notes"] = "n" * 200
    big = build_live_modelfile(BASE, BASE_EXAMPLES, learned, "", 24)
    assert um.check_context_budget(small) > 0
    try:
        um.check_context_budget(big)
    except um.ModelfileTooLargeError:
        pass
    else:
        raise AssertionError("48 long learned examples should not fit in 70% of num_ctx 12288")


def test_context_guard_needs_a_num_ctx_and_a_system_block():
    for text in (_modelfile("x", num_ctx=None), "FROM m\nPARAMETER num_ctx 8192\n"):
        try:
            um.check_context_budget(text)
        except um.ModelfileTooLargeError:
            pass
        else:
            raise AssertionError(f"no error for {text!r}")


def test_the_committed_base_fits_the_budget():
    assert um.check_context_budget(build_live_modelfile(BASE, BASE_EXAMPLES, [], "", 24)) < 0.7 * 12288


def _realistic(i: int, is_ad: bool) -> dict:
    """A learned example about the size the tuner saves: a title, a 200-character
    summary and a short note, about 95 tokens once formatted."""
    title = f"Example {i}: a company ships a new agent model"
    summary = (f"Item {i} describes the release, the benchmarks it reports, the price per million "
               "tokens and when the model reaches the API. It quotes the company and an outside "
               "researcher about the results.")[:200]
    return {"id": f"r{i}", "title": title, "summary": summary, "source": "S", "link": "https://x/",
            "is_ad": is_ad, "ad_type": "sponsor_content" if is_ad else "not_ad",
            "notes": "The judge's short reason for the label.",
            "added": f"2026-11-{1 + i % 28:02d}"}


def test_base_plus_24_realistic_learned_examples_per_class_fits_the_guard():
    learned = [_realistic(i, i % 2 == 0) for i in range(48)]
    text = build_live_modelfile(BASE, BASE_EXAMPLES, learned, "", 24)
    assert text.count("Input: Title: 'Example ") == 48
    block = text[text.index("Input: Title: 'Example "):]
    per_example = len(block) / 48 / um.CHARS_PER_TOKEN
    assert 85 <= per_example <= 105  # about the 95 tokens per example measured with Ollama
    assert um.check_context_budget(text) <= 0.7 * 12288
