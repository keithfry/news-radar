"""Retention: purge old radar episodes from a topic's output directory.

Two tiers, both keyed off the date encoded in an episode's filename stem
(not file mtime — mtimes get touched by backfills and git operations, the
filename date is the authoritative "episode date"):

  - purge_audio: drops audio-only artifacts (mp3, chapters, transcript,
    cover images) once an episode crosses its max_age_days. If the
    episode's HTML page survives, strips the inline podcast player and
    og:image/twitter:image meta tags so it doesn't point at now-deleted
    files.
  - purge_episodes: drops everything left for an episode once it crosses
    its own (larger) max_age_days, and removes the YYYY-MM directory if
    that leaves it empty.

Both operate on a single topic's output_dir, laid out as
output_dir/YYYY-MM/<file_prefix>-YYYY-MM-DD.<ext> — the same layout
podcast_rss.py and the main pipeline already use.
"""

import re
from datetime import date, datetime
from pathlib import Path

from .filename_dates import parse_date_from_stem

_YM_DIR_RE = re.compile(r"^\d{4}-\d{2}$")

# Default retention thresholds. Consumers import these rather than hardcoding
# their own, so every site built on newsradar ages content out the same way
# unless it deliberately overrides max_age_days.
DEFAULT_AUDIO_MAX_AGE_DAYS = 90
DEFAULT_EPISODE_MAX_AGE_DAYS = 150

_AUDIO_EXTS = (".mp3", ".chapters.json", ".transcript.json", ".jpg", ".og.jpg")

_PLAYER_RE = re.compile(r'\n?\s*<div class="podcast-player">.*?</div>\n?', re.DOTALL)
# og:image plus its og:image:width/height companions, and twitter:image.
# og:title/og:description stay — they're still valid without an image.
_IMAGE_META_RE = re.compile(
    r'<meta (?:property="og:image(?::\w+)?"|name="twitter:image")[^>]*/>\n?'
)
# summary_large_image requires an image; fall back to the plain card.
_LARGE_CARD = '<meta name="twitter:card" content="summary_large_image"/>'
_PLAIN_CARD = '<meta name="twitter:card" content="summary"/>'


def _episode_stems(output_dir: Path, file_prefix: str) -> list[tuple[Path, date, str]]:
    """Return (month_dir, episode_date, stem) for every dated episode found under output_dir."""
    results: list[tuple[Path, date, str]] = []
    if not output_dir.is_dir():
        return results

    stem_re = re.compile(rf"^{re.escape(file_prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})")
    for month_dir in sorted(output_dir.iterdir()):
        if not month_dir.is_dir() or not _YM_DIR_RE.match(month_dir.name):
            continue
        seen: set[str] = set()
        for f in sorted(month_dir.iterdir()):
            match = stem_re.match(f.name)
            if not match:
                continue
            date_str = match.group(1)
            if date_str in seen:
                continue
            seen.add(date_str)
            ep_date = parse_date_from_stem(f"{file_prefix}-{date_str}", file_prefix)
            if ep_date is not None:
                results.append((month_dir, ep_date, f"{file_prefix}-{date_str}"))
    return results


def _strip_audio_references(html_path: Path, log=print) -> bool:
    """Remove the inline podcast player and og:image/twitter:image meta tags.

    Returns True if the file was modified (False if there was nothing to strip,
    e.g. a second run against an already-stripped file).
    """
    text = html_path.read_text(encoding="utf-8")
    new_text = _PLAYER_RE.sub("\n", text)
    new_text = _IMAGE_META_RE.sub("", new_text)
    new_text = new_text.replace(_LARGE_CARD, _PLAIN_CARD)
    if new_text == text:
        return False
    html_path.write_text(new_text, encoding="utf-8")
    log(f"    stripped audio references: {html_path}")
    return True


def purge_audio(
    output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print
) -> list[Path]:
    """Delete audio artifacts for episodes older than max_age_days. HTML/JSON survive. Returns changed paths."""
    changed: list[Path] = []
    for month_dir, ep_date, stem in _episode_stems(output_dir, file_prefix):
        if (now.date() - ep_date).days <= max_age_days:
            continue

        for ext in _AUDIO_EXTS:
            f = month_dir / f"{stem}{ext}"
            if f.exists():
                f.unlink(missing_ok=True)
                log(f"    deleted: {f}")
                changed.append(f)

        html_path = month_dir / f"{stem}.html"
        if html_path.exists() and _strip_audio_references(html_path, log=log):
            changed.append(html_path)

    return changed


def purge_episodes(
    output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print
) -> list[Path]:
    """Delete every remaining file for episodes older than max_age_days.

    Removes the YYYY-MM directory if it ends up empty. Returns changed paths.
    """
    changed: list[Path] = []
    touched_dirs: set[Path] = set()

    for month_dir, ep_date, stem in _episode_stems(output_dir, file_prefix):
        if (now.date() - ep_date).days <= max_age_days:
            continue

        for f in sorted(month_dir.glob(f"{stem}.*")):
            f.unlink(missing_ok=True)
            log(f"    deleted: {f}")
            changed.append(f)
        touched_dirs.add(month_dir)

    for month_dir in touched_dirs:
        if month_dir.is_dir() and not any(month_dir.iterdir()):
            month_dir.rmdir()
            log(f"    removed empty dir: {month_dir}")

    return changed
