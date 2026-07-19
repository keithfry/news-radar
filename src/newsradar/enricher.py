"""Enrich pipeline items with ranking, audio scripts, and chapter time estimates.

Runs after deduplication. Produces the shared JSON artifact consumed by
html_generator and podcast_generator.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .topics import Topic

# Words per minute for Kokoro TTS (approximate, used for time estimation)
_TTS_WPM = 130

# 5 Kokoro voices for round-robin assignment (American English)
# Note: am_adam and af_nicole are intentionally excluded
KOKORO_VOICES = [
    "af_heart",    # female, warm
    "am_echo",     # male, neutral
    "af_bella",    # female, expressive
    "am_michael",  # male, deep
    "af_nova",     # female, energetic
]


def _words(text: str) -> int:
    return len(text.split())


def _seconds(word_count: int) -> int:
    return round(word_count / _TTS_WPM * 60)


def _estimate_chapter_times(intro_script: str, audio_scripts: list[str]) -> list[int]:
    """Return list of chapter start times in seconds.

    Index 0 = intro start (always 0).
    Index 1..N = item start times, estimated from word counts.
    """
    times = [0]
    cursor = _seconds(_words(intro_script))
    times.append(cursor)
    # Iterate all but the last script: each iteration appends the START of the
    # *next* item. The last item's start = chapter_times[-1] via caller's fallback.
    for script in audio_scripts[:-1]:
        cursor += _seconds(_words(script))
        times.append(cursor)
    return times


def _build_enriched_dict(date: datetime, intro_script: str, episode_tagline: str, outro_script: str, items: list[dict]) -> dict:
    return {
        "date": date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "intro_script": intro_script,
        "episode_tagline": episode_tagline,
        "outro_script": outro_script,
        "items": items,
    }


def write_enriched_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def enrich(
    all_items: list[dict],
    date: datetime,
    output_path: Path,
    summarize_model: str,
    rank_model: str,
    topic: Topic,
    llm_workers: int,
    log=print,
) -> dict:
    """Rank, script, and time-estimate all items. Write JSON. Return enriched dict.

    Args:
        all_items: deduplicated items from the pipeline (emails + articles + papers)
        date: as_of datetime (ET)
        output_path: where to write the JSON file
        summarize_model: Ollama model for scripting/intro generation
        rank_model: Ollama model for ranking
        topic: Topic being enriched (used for intro/outro script generation)
        llm_workers: number of parallel workers for audio script generation
        log: logging function

    Returns:
        enriched dict (same structure as JSON file)
    """
    from .llm import rank_items, generate_audio_script, generate_intro_script, generate_episode_tagline, generate_outro_script, unload_all_models

    # Separate podcast candidates (emails + articles) from papers
    podcast_candidates = [i for i in all_items if not i.get("_is_arxiv")]
    papers = [i for i in all_items if i.get("_is_arxiv")]

    log(f"  Enriching {len(podcast_candidates)} podcast candidates + {len(papers)} papers (excluded from audio)")

    # Evict summarize-phase models before loading the (larger) rank model
    evicted = unload_all_models()
    if evicted:
        log(f"  Unloaded models before ranking: {', '.join(evicted)}")

    # Rank podcast candidates
    log(f"  Ranking items by relevance (model: {rank_model})...")
    ranked = rank_items(podcast_candidates, model=rank_model)

    # Generate audio script per item in parallel (rank order preserved via futures dict)
    log(f"  Generating {len(ranked)} audio scripts ({llm_workers} workers, model: {summarize_model})...")
    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {
            executor.submit(generate_audio_script, item, summarize_model): item
            for item in ranked
        }
        for future in as_completed(futures):
            item = futures[future]
            item["audio_script"] = future.result()
            item["voice_index"] = (item["rank"] - 1) % len(KOKORO_VOICES)
            item["include_in_podcast"] = True
            log(f"    [{item['rank']}] scripted: {item['title'][:60]}")

    # Generate intro, outro, and tagline in parallel (all independent)
    log("  Generating intro, outro, and episode tagline in parallel...")
    intro_result: list = []
    outro_result: list = []
    tagline_result: list = []

    def _gen_intro():
        intro_result.append(generate_intro_script(ranked, date, model=summarize_model, topic=topic))

    def _gen_outro():
        outro_result.append(generate_outro_script(ranked, date, model=summarize_model, topic=topic))

    def _gen_tagline():
        tagline_result.append(generate_episode_tagline(ranked, model=summarize_model))

    t_intro   = threading.Thread(target=_gen_intro)
    t_outro   = threading.Thread(target=_gen_outro)
    t_tagline = threading.Thread(target=_gen_tagline)
    t_intro.start()
    t_outro.start()
    t_tagline.start()
    t_intro.join()
    t_outro.join()
    t_tagline.join()

    intro_script    = intro_result[0]
    outro_script    = outro_result[0]
    episode_tagline = tagline_result[0]
    log(f"  Episode tagline: {episode_tagline}")
    log(f"  Outro: {outro_script[:80]}{'...' if len(outro_script) > 80 else ''}")

    # Estimate chapter times
    audio_scripts = [item["audio_script"] for item in ranked]
    chapter_times = _estimate_chapter_times(intro_script, audio_scripts)
    # times[0] = intro start (always 0), times[1..N] = item starts
    for i, item in enumerate(ranked):
        item["chapter_start_seconds"] = chapter_times[i + 1] if i + 1 < len(chapter_times) else chapter_times[-1]

    # Add papers back (not in podcast, no audio fields)
    for paper in papers:
        paper["include_in_podcast"] = False
        paper["chapter_start_seconds"] = None

    all_enriched = ranked + papers

    data = _build_enriched_dict(date, intro_script, episode_tagline, outro_script, all_enriched)
    write_enriched_json(data, output_path)
    log(f"  Wrote enriched JSON: {output_path}")

    return data
