"""Regenerate AdDetectorModelfile training examples from labeled_examples.json.

Run after adding new labeled examples, then rebuild the Ollama model:
    uv run -m newsradar.ad_detector.update_modelfile
    uv run newsradar tune --config <config> --rebuild
(A bare `ollama create` builds from the base Modelfile only and drops learned data.)

Up to MAX_EXAMPLES are selected (balanced between ads and not-ads).
"""

import json
import random
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent.parent
DEFAULT_DATA_PATH = _REPO_ROOT / "tests" / "test_data" / "labeled_examples.json"
DEFAULT_MODELFILE_PATH = _PACKAGE_DIR / "AdDetectorModelfile"
MAX_EXAMPLES = 25  # per class
NOTES_CHARS = 200  # the longest reason an example's Output line carries

TRAINING_SECTION_START = "TRAINING EXAMPLES (memorize these):\n"
_SECTION_END = '"""\n'

LEARNED_RULES_START = "LEARNED RULES (added by the tuner):\n"
LEARNED_RULES_END = "END LEARNED RULES\n"


def load_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data["examples"] if e.get("is_ad") is not None]


def format_training_block(examples: list[dict]) -> str:
    """Format examples as the training section, interleaving ads and non-ads."""
    ads = [e for e in examples if e["is_ad"]]
    not_ads = [e for e in examples if not e["is_ad"]]

    max_len = max(len(ads), len(not_ads))
    interleaved: list[dict] = []
    for i in range(max_len):
        if i < len(ads):
            interleaved.append(ads[i])
        if i < len(not_ads):
            interleaved.append(not_ads[i])

    lines = [TRAINING_SECTION_START]
    for e in interleaved:
        is_ad = e["is_ad"]
        reason = (e.get("notes", "").strip() or e.get("ad_type", ""))[:NOTES_CHARS]
        # json.dumps keeps a learned note's quotes and newlines from breaking the line.
        label = f'{{"is_ad": {"true" if is_ad else "false"}, "reason": {json.dumps(reason, ensure_ascii=False)}}}'
        lines.append(
            f"Input: Title: {e['title']!r} | Summary: {e['summary'][:200]!r}\n"
            f"Output: {label}\n"
        )
    return "\n".join(lines)


def generate_training_block(examples: list[dict], max_per_class: int = MAX_EXAMPLES) -> str:
    ads = [e for e in examples if e["is_ad"]]
    not_ads = [e for e in examples if not e["is_ad"]]

    random.seed(42)
    selected_ads = random.sample(ads, min(len(ads), max_per_class))
    selected_ok = random.sample(not_ads, min(len(not_ads), max_per_class))
    return format_training_block(selected_ads + selected_ok)


def replace_training_block(content: str, block: str) -> str:
    """Return content with the TRAINING EXAMPLES section replaced by block."""
    start_idx = content.find(TRAINING_SECTION_START)
    if start_idx == -1:
        print(f"[warn] Could not find '{TRAINING_SECTION_START.strip()}' marker in Modelfile.")
        print("[warn] Appending training block instead.")
        # Find end of SYSTEM block and insert before it
        end_sys = content.rfind(_SECTION_END)
        if end_sys == -1:
            raise ValueError("Could not find end of SYSTEM block in Modelfile")
        return content[:end_sys] + "\n" + block + "\n" + content[end_sys:]
    # Find the end of the section (the closing """)
    end_idx = content.find(_SECTION_END, start_idx)
    if end_idx == -1:
        raise ValueError("Malformed Modelfile: SYSTEM block not closed")
    return content[:start_idx] + block + "\n" + content[end_idx:]


def rewrite_modelfile(modelfile_path: Path, training_block: str) -> None:
    content = modelfile_path.read_text(encoding="utf-8")
    modelfile_path.write_text(replace_training_block(content, training_block), encoding="utf-8")


def _newest_per_class(examples: list[dict], per_class: int) -> list[dict]:
    chosen: list[dict] = []
    for is_ad in (True, False):
        group = [e for e in examples if bool(e["is_ad"]) == is_ad]
        group.sort(key=lambda e: (e.get("added", ""), str(e.get("id", ""))))
        chosen.extend(group[-per_class:] if per_class > 0 else [])
    chosen.sort(key=lambda e: (e.get("added", ""), str(e.get("id", ""))))
    return chosen


def build_live_modelfile(
    base_text: str,
    base_examples: list[dict],
    learned_examples: list[dict],
    learned_rules: str,
    per_class: int,
) -> str:
    """Build the live Modelfile from the committed base plus learned data.

    Text above LEARNED_RULES_START is left byte-identical.
    """
    start = base_text.index(LEARNED_RULES_START) + len(LEARNED_RULES_START)
    end = base_text.index(LEARNED_RULES_END, start)
    rules = learned_rules.strip()
    middle = rules + "\n" if rules else ""
    text = base_text[:start] + middle + base_text[end:]
    examples = list(base_examples) + _newest_per_class(learned_examples, per_class)
    return replace_training_block(text, format_training_block(examples))


def update_modelfile(
    data_path: Path | None = None,
    modelfile_path: Path | None = None,
    max_per_class: int = MAX_EXAMPLES,
) -> None:
    data_path = data_path or DEFAULT_DATA_PATH
    modelfile_path = modelfile_path or DEFAULT_MODELFILE_PATH

    examples = load_examples(data_path)
    ads = sum(1 for e in examples if e["is_ad"])
    ok = len(examples) - ads
    print(f"Loaded {len(examples)} examples: {ads} ads, {ok} not-ads")

    training_block = generate_training_block(examples, max_per_class)
    rewrite_modelfile(modelfile_path, training_block)

    line_count = training_block.count("\n")
    print(f"Wrote {line_count} training lines to {modelfile_path.name}")
    print()
    print("Rebuild the model with:")
    print(f"  ollama rm ad-detector && ollama create ad-detector -f {modelfile_path}")


def main() -> None:
    update_modelfile()


if __name__ == "__main__":
    main()
