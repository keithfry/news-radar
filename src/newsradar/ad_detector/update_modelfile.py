"""Regenerate AdDetectorModelfile training examples from labeled_examples.json.

Run after adding new labeled examples, then rebuild the Ollama model:
    uv run -m newsradar.ad_detector.update_modelfile
    ollama rm ad-detector && ollama create ad-detector -f AdDetectorModelfile

Up to MAX_EXAMPLES are selected (balanced between ads and not-ads).
"""

import json
import random
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_DATA_PATH = _PACKAGE_DIR / "test_data" / "labeled_examples.json"
_DEFAULT_MODELFILE_PATH = _PACKAGE_DIR / "AdDetectorModelfile"
MAX_EXAMPLES = 25  # per class

_SECTION_START = "TRAINING EXAMPLES (memorize these):\n"
_SECTION_END = '"""\n'


def load_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data["examples"] if e.get("is_ad") is not None]


def generate_training_block(examples: list[dict], max_per_class: int = MAX_EXAMPLES) -> str:
    ads = [e for e in examples if e["is_ad"]]
    not_ads = [e for e in examples if not e["is_ad"]]

    random.seed(42)
    selected_ads = random.sample(ads, min(len(ads), max_per_class))
    selected_ok = random.sample(not_ads, min(len(not_ads), max_per_class))

    # Interleave ads and non-ads for better few-shot presentation
    max_len = max(len(selected_ads), len(selected_ok))
    interleaved: list[dict] = []
    for i in range(max_len):
        if i < len(selected_ads):
            interleaved.append(selected_ads[i])
        if i < len(selected_ok):
            interleaved.append(selected_ok[i])

    lines = [_SECTION_START]
    for e in interleaved:
        is_ad = e["is_ad"]
        reason = e.get("notes", "").strip() or e.get("ad_type", "")
        label = f'{{"is_ad": {"true" if is_ad else "false"}, "reason": "{reason}"}}'
        lines.append(
            f"Input: Title: {e['title']!r} | Summary: {e['summary'][:200]!r}\n"
            f"Output: {label}\n"
        )
    return "\n".join(lines)


def rewrite_modelfile(modelfile_path: Path, training_block: str) -> None:
    content = modelfile_path.read_text(encoding="utf-8")

    # Find and replace the TRAINING EXAMPLES section
    start_idx = content.find(_SECTION_START)
    if start_idx == -1:
        print(f"[warn] Could not find '{_SECTION_START.strip()}' marker in Modelfile.")
        print("[warn] Appending training block instead.")
        # Find end of SYSTEM block and insert before it
        end_sys = content.rfind(_SECTION_END)
        if end_sys == -1:
            raise ValueError("Could not find end of SYSTEM block in Modelfile")
        new_content = content[:end_sys] + "\n" + training_block + "\n" + content[end_sys:]
    else:
        # Find the end of the section (the closing """)
        end_idx = content.find(_SECTION_END, start_idx)
        if end_idx == -1:
            raise ValueError("Malformed Modelfile: SYSTEM block not closed")
        new_content = content[:start_idx] + training_block + "\n" + content[end_idx:]

    modelfile_path.write_text(new_content, encoding="utf-8")


def update_modelfile(
    data_path: Path | None = None,
    modelfile_path: Path | None = None,
    max_per_class: int = MAX_EXAMPLES,
) -> None:
    data_path = data_path or _DEFAULT_DATA_PATH
    modelfile_path = modelfile_path or _DEFAULT_MODELFILE_PATH

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
