"""Generate a podcast MP3 with ID3 chapter markers from enriched data.

Requires ffmpeg on PATH and Kokoro TTS installed.
"""

import json
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

from .enricher import KOKORO_VOICES
from .topics import Topic

_tts_thread_local = threading.local()


def _get_pipeline():
    """Return a thread-local KPipeline, creating it on first use per thread."""
    if not hasattr(_tts_thread_local, "pipeline"):
        from kokoro import KPipeline
        _tts_thread_local.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _tts_thread_local.pipeline


def _voice_for(voice_index: int) -> str:
    return KOKORO_VOICES[voice_index % len(KOKORO_VOICES)]


def _build_chapters_json(chapters: list[dict], episode_title: str | None = None) -> str:
    """Build Podcasting 2.0 chapters JSON string."""
    data: dict = {"version": "1.2.0"}
    if episode_title:
        data["title"] = episode_title
    data["chapters"] = chapters
    return json.dumps(data, indent=2)


import re as _re
_SENT_RE = _re.compile(r'(?<=[.!?]["\'])\s+(?=[A-Z"\'])|(?<=[.!?])\s+(?=[A-Z"\'])')
_FROM_HDR_RE = _re.compile(r'^(From [^"]+?"[^"]*"\.?)\s*')


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    # Peel off "From [source], \"[title]\"." header as its own segment before splitting the body
    m = _FROM_HDR_RE.match(text)
    if m:
        header = m.group(1).strip()
        rest = text[m.end():].strip()
        body_parts = _SENT_RE.split(rest) if rest else []
        parts = [header] + body_parts
    else:
        parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _interpolate_sentences(
    sentences: list[str], start: float, end: float
) -> list[tuple[str, float, float]]:
    """Distribute timestamps across sentences proportional to character count."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0], round(start, 3), round(end, 3))]
    total_chars = sum(len(s) for s in sentences) or 1
    duration = end - start
    result = []
    cursor = start
    for sent in sentences:
        sent_end = cursor + duration * len(sent) / total_chars
        result.append((sent, round(cursor, 3), round(sent_end, 3)))
        cursor = sent_end
    return result


def _build_transcript_json(segments: list[dict]) -> str:
    return json.dumps({"version": "1.0.0", "segments": segments}, indent=2)


def _tts_segment(text: str, voice: str, out_wav: Path) -> float:
    """Synthesize text to WAV using thread-local Kokoro pipeline. Returns duration in seconds."""
    import soundfile as sf
    import numpy as np

    pipeline = _get_pipeline()
    audio_chunks = []
    sample_rate = 24000

    for _, _, audio in pipeline(text, voice=voice):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError(f"Kokoro produced no audio for text: {text[:50]!r}")

    full_audio = np.concatenate(audio_chunks)
    sf.write(str(out_wav), full_audio, sample_rate)
    return len(full_audio) / sample_rate


def _concat_wavs_to_mp3(wav_files: list[Path], out_mp3: Path) -> None:
    """Use ffmpeg to concat WAV segments into a single MP3."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = Path(f.name)
        for wav in wav_files:
            f.write(f"file '{wav.resolve()}'\n")

    try:
        subprocess.run(
            [
                _FFMPEG, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-ar", "22050", "-ac", "1", "-b:a", "64k",
                str(out_mp3),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}") from e
    finally:
        concat_file.unlink(missing_ok=True)


def _write_id3_chapters(
    mp3_path: Path,
    chapters: list[dict],
    episode_title: str | None = None,
    cover_path: Path | None = None,
) -> None:
    """Write ID3v2 CHAP frames (and optional cover art) to an MP3 file."""
    from mutagen.id3 import ID3, CHAP, CTOC, TIT2, APIC
    from mutagen.id3 import ID3NoHeaderError

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()

    if episode_title:
        tags.add(TIT2(encoding=3, text=[episode_title]))

    if cover_path and cover_path.exists():
        tags.add(APIC(
            encoding=3,
            mime="image/jpeg",
            type=3,
            desc="Cover",
            data=cover_path.read_bytes(),
        ))

    # Remove existing chapter tags
    for key in list(tags.keys()):
        if key.startswith("CHAP") or key.startswith("CTOC"):
            del tags[key]

    chap_ids = []
    for i, chap in enumerate(chapters):
        start_ms = int(chap["startTime"] * 1000)
        end_ms = int(chap.get("endTime", chap["startTime"] + 30) * 1000)
        elem_id = f"chap{i}"
        chap_ids.append(elem_id)
        tags.add(CHAP(
            element_id=elem_id,
            start_time=start_ms,
            end_time=end_ms,
            start_offset=0xFFFFFFFF,
            end_offset=0xFFFFFFFF,
            sub_frames=[TIT2(encoding=3, text=[chap["title"]])],
        ))

    tags.add(CTOC(
        element_id="toc",
        flags=0x03,  # top-level, ordered
        child_element_ids=chap_ids,
        sub_frames=[TIT2(encoding=3, text=["Table of Contents"])],
    ))
    tags.save(str(mp3_path), v2_version=3)


def generate_podcast(
    enriched_data: dict,
    date: datetime,
    output_dir: Path,
    topic: Topic,
    tts_workers: int = 2,
    log=print,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Generate MP3 + chapters.json + episode cover + OG card from enriched data.

    Also updates enriched_data['items'] with actual chapter_start_seconds from audio timings.

    Returns:
        (mp3_path, chapters_json_path, episode_cover_path_or_None, og_card_path_or_None)
    """
    file_prefix = topic.file_prefix
    topic_label = topic.display_name
    date_str = date.strftime("%Y-%m-%d")
    mp3_path = output_dir / f"{file_prefix}-{date_str}.mp3"
    chapters_json_path = output_dir / f"{file_prefix}-{date_str}.chapters.json"

    tagline = enriched_data.get("episode_tagline", "")
    title_date = f"{date.strftime('%B')} {date.day}, {date.year}"
    episode_title = f"{title_date} : {tagline}" if tagline else f"{topic_label} Radar — {title_date}"

    podcast_items = [i for i in enriched_data["items"] if i.get("include_in_podcast")]
    intro_script = enriched_data["intro_script"]
    outro_script = enriched_data.get("outro_script", "")

    log(f"  Generating audio for intro + {len(podcast_items)} items + outro ({tts_workers} workers)...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # outro order_index sits after all ranked items
    _outro_idx = max((item["rank"] for item in podcast_items), default=0) + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build sentence-level work list: (sort_key, wav_path, text, voice, label)
        # sort_key = (item_order, sentence_idx); item_order: 0=intro, rank=items, _outro_idx=outro
        work = []
        for si, sent in enumerate(_split_sentences(intro_script)):
            work.append(((0, si), tmp / f"intro_{si:03d}.wav", sent, _voice_for(0), f"intro[{si}]"))
        for item in podcast_items:
            for si, sent in enumerate(_split_sentences(item["audio_script"])):
                work.append(((item["rank"], si), tmp / f"item_{item['rank']:03d}_{si:03d}.wav", sent, _voice_for(item["voice_index"]), f"item {item['rank']}[{si}]"))
        if outro_script:
            for si, sent in enumerate(_split_sentences(outro_script)):
                work.append(((_outro_idx, si), tmp / f"outro_{si:03d}.wav", sent, _voice_for(0), f"outro[{si}]"))

        # Synthesize all sentences in parallel; results keyed by sort_key
        durations: dict[tuple, float] = {}

        def _synth(sort_key: tuple, wav_path: Path, text: str, voice: str, label: str) -> tuple[tuple, float]:
            log(f"  [tts] {label} ({voice})...")
            dur = _tts_segment(text, voice, wav_path)
            return sort_key, dur

        with ThreadPoolExecutor(max_workers=tts_workers) as executor:
            futures = {
                executor.submit(_synth, *w): w[0] for w in work
            }
            for future in as_completed(futures):
                sort_key, dur = future.result()
                durations[sort_key] = dur

        # Reconstruct ordered wav list and compute sentence-level start times
        ordered = sorted(work, key=lambda w: w[0])
        wav_files: list[Path] = []
        sentence_starts: list[float] = []
        cursor = 0.0
        for sort_key, wav_path, _, _, _ in ordered:
            wav_files.append(wav_path)
            sentence_starts.append(cursor)
            cursor += durations[sort_key]

        # Derive chapter-level starts: first sentence of each item_order
        chapter_starts: dict[int, float] = {}
        for i, (sort_key, _, _, _, _) in enumerate(ordered):
            item_order, sentence_idx = sort_key
            if sentence_idx == 0:
                chapter_starts[item_order] = sentence_starts[i]

        # Write chapter_start_seconds back to items
        for item in podcast_items:
            item["chapter_start_seconds"] = int(chapter_starts.get(item["rank"], 0))

        total_duration = int(cursor)
        log(f"  Total audio: {total_duration}s ({total_duration // 60}m {total_duration % 60}s)")

        # Build sentence-level transcript segments
        transcript_segments: list[dict] = []
        for i, (_, _, text, voice, _) in enumerate(ordered):
            seg_end = sentence_starts[i + 1] if i + 1 < len(sentence_starts) else float(total_duration)
            transcript_segments.append({
                "startTime": round(sentence_starts[i], 3),
                "endTime": round(seg_end, 3),
                "text": text,
                "voice": voice,
            })

        # Concat all WAV → MP3
        log("  Merging WAV segments → MP3...")
        _concat_wavs_to_mp3(wav_files, mp3_path)

    # Build chapters list
    chapters = [{"startTime": 0, "title": "Introduction"}]
    for item in podcast_items:
        start = chapter_starts.get(item["rank"], 0)
        chap = {"startTime": int(start), "title": item["title"]}
        if item.get("link"):
            chap["url"] = item["link"]
        chapters.append(chap)

    # Outro chapter
    if outro_script:
        outro_start = int(chapter_starts.get(_outro_idx, total_duration))
        chapters.append({"startTime": outro_start, "title": "Sign Off"})

    # Add endTime to each chapter
    for i, chap in enumerate(chapters):
        if i + 1 < len(chapters):
            chap["endTime"] = chapters[i + 1]["startTime"]
        else:
            chap["endTime"] = total_duration

    # Write Podcasting 2.0 chapters JSON
    chapters_json_path.write_text(_build_chapters_json(chapters, episode_title), encoding="utf-8")
    log(f"  Wrote chapters JSON: {chapters_json_path}")

    # Write transcript JSON (segment per chapter, exact text + float timestamps)
    transcript_json_path = chapters_json_path.parent / chapters_json_path.name.replace(
        ".chapters.json", ".transcript.json"
    )
    transcript_json_path.write_text(
        _build_transcript_json(transcript_segments),
        encoding="utf-8",
    )
    log(f"  Wrote transcript JSON: {transcript_json_path}")

    # Generate episode cover + OG social card
    date_str = date.strftime("%Y-%m-%d")
    cover_path = output_dir / f"{file_prefix}-{date_str}.jpg"
    og_path = output_dir / f"{file_prefix}-{date_str}.og.jpg"
    try:
        from .cover_generator import generate_episode_cover, generate_og_card
        display_tagline = tagline or topic_label
        generate_episode_cover(topic, display_tagline, date, total_duration, cover_path)
        generate_og_card(topic, display_tagline, date, total_duration, og_path)
        log(f"  Episode cover: {cover_path.name}, OG card: {og_path.name}")
    except Exception as e:
        log(f"  WARNING: cover generation failed: {e}")
        cover_path = None
        og_path = None

    # Embed ID3 chapter tags and cover art in MP3
    _write_id3_chapters(mp3_path, chapters, episode_title, cover_path)
    log(f"  Wrote ID3 tags to: {mp3_path}")

    return mp3_path, chapters_json_path, transcript_json_path, cover_path, og_path
