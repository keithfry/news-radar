"""Ad detection tests, ported from the original script-style harness.

The original `test_ad_detector.py` was a standalone CLI script (argparse +
`main()`, no pytest-collectible test functions) that printed precision/
recall/F1 tables. This version keeps the same labeled dataset and detectors
but expresses them as real pytest assertions:

- the heuristic detector (`newsradar.cli._is_advertisement`) is pure Python,
  no LLM required, and always runs.
- the LLM-gate detector (`newsradar.llm.classify_ad` with the packaged
  `ad-detector` Ollama model) requires a local Ollama daemon with that model
  installed (`newsradar ad-detector install`); skipped automatically if
  unavailable.

Also verifies the `newsradar.ad_detector.*` import paths referenced by the
porting task — `compare_models` and `update_modelfile` are submodules, not
flat top-level names (`compare_models` additionally isn't re-exported from
`newsradar.ad_detector/__init__.py` since it pulls in the heavier RSS/
article-fetching dependency chain; import it directly from
`newsradar.ad_detector.compare_models`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from newsradar.cli import _is_advertisement
from newsradar.llm import classify_ad

from conftest import ollama_has_model

_DATA_PATH = Path(__file__).parent / "test_data" / "labeled_examples.json"
_AD_DETECTOR_MODEL = "ad-detector"


def _load_examples() -> list[dict]:
    data = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return [e for e in data["examples"] if e.get("is_ad") is not None]


def _metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def test_module_import_paths():
    """newsradar.ad_detector.* submodules import correctly (not flat names)."""
    from newsradar.ad_detector.compare_models import compare_models
    from newsradar.ad_detector.update_modelfile import update_modelfile
    from newsradar.ad_detector import install_ad_detector_model, update_modelfile as reexported_update

    assert callable(compare_models)
    assert callable(update_modelfile)
    assert callable(install_ad_detector_model)
    assert reexported_update is update_modelfile


def test_labeled_dataset_loads():
    examples = _load_examples()
    assert len(examples) > 0
    ad_count = sum(1 for e in examples if e["is_ad"])
    assert 0 < ad_count < len(examples), "dataset should contain both ad and non-ad examples"


def test_heuristic_detector_has_no_false_positives():
    """_is_advertisement is a cheap zero-LLM-cost pre-filter — it should never
    flag a genuinely non-ad labeled example (its whole job is to be a safe,
    conservative fast-path; the LLM gate catches the subtler cases)."""
    examples = _load_examples()
    y_true = [e["is_ad"] for e in examples]
    y_pred = [_is_advertisement(e["title"], e["summary"]) for e in examples]
    metrics = _metrics(y_true, y_pred)
    assert metrics["fp"] == 0, f"heuristic flagged a non-ad example as an ad: fp={metrics['fp']}"


def test_heuristic_detector_catches_obvious_price_spam():
    assert _is_advertisement(
        "Huge Sale",
        "Everything on sale: $5.99, $10.00, $15.50 — shop now while supplies last!",
    )


@pytest.mark.skipif(
    not ollama_has_model(_AD_DETECTOR_MODEL),
    reason=f"Ollama model {_AD_DETECTOR_MODEL!r} not installed locally "
    "(run: newsradar ad-detector install)",
)
def test_llm_ad_gate_detector_metrics():
    examples = _load_examples()
    y_true = [e["is_ad"] for e in examples]
    y_pred = []
    for e in examples:
        is_ad, _reason = classify_ad(e["title"], e["summary"], _AD_DETECTOR_MODEL)
        y_pred.append(is_ad)
    metrics = _metrics(y_true, y_pred)
    # The packaged ad-detector model is fine-tuned on exactly this dataset's
    # examples, so it should score very well here — this is a regression
    # guard, not a generalization benchmark.
    assert metrics["f1"] >= 0.8, f"ad-detector F1 too low: {metrics}"
