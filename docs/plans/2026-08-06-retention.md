# Retention Purge Logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable retention logic to the `newsradar` package that purges audio artifacts from radar episodes older than 90 days (default) and whole episodes older than 150 days (default). Consumers (e.g. the `techradar` repo's `retention.py` orchestrator) call it per topic and own everything site-specific: RSS/index regeneration, git commit/push, scheduling.

**Architecture:** Pure purge logic as two functions operating on `Path`s with an injected `now`, fully unit-testable — no git, no CLI, just filesystem + HTML edits. Default thresholds are exported as module constants so every consumer shares one source of truth. See `docs/plans/2026-08-06-retention-design.md` for the requirements.

**Tech Stack:** Python 3.13, `unittest` (matching existing `news-radar` test style), existing `newsradar` `podcast_rss` module.

## Global Constraints

- Age is computed from the date encoded in the filename stem (e.g. `ai-radar-2026-05-01`), never from file mtime.
- Default audio-artifact purge threshold: 90 days (`DEFAULT_AUDIO_MAX_AGE_DAYS`). Default full-episode purge threshold: 150 days (`DEFAULT_EPISODE_MAX_AGE_DAYS`). Both are exported from `newsradar.retention`; callers pass them (or an override) as `max_age_days`.
- Audio artifacts = `.mp3`, `.chapters.json`, `.transcript.json`, `.jpg`, `.og.jpg`.
- When audio artifacts are purged but the episode's `.html` survives, strip the inline `<div class="podcast-player">...</div>` block and the `og:image`/`twitter:image` meta lines from that HTML.
- Missing files during a delete are not errors (`missing_ok=True` / idempotent).
- Follow existing code style: type-hinted Python, `unittest.TestCase` + `tempfile.TemporaryDirectory` (see `tests/test_podcast_rss.py`).

---

### Task 1: Shared filename-date helper

**Files:**
- Create: `news-radar/src/newsradar/filename_dates.py`
- Modify: `news-radar/src/newsradar/podcast_rss.py:35-45` (replace `_date_from_stem`'s body to delegate to the new helper)
- Test: `news-radar/tests/test_filename_dates.py`

**Interfaces:**
- Produces: `parse_date_from_stem(stem: str, prefix: str) -> date | None` — parses `YYYY-MM-DD` out of a stem like `"ai-radar-2026-05-21"` given `prefix="ai-radar"`. Returns `None` if unparseable. Used by Task 2's `retention.py` and by `podcast_rss.py`.

- [ ] **Step 1: Write the failing test**

Create `news-radar/tests/test_filename_dates.py`:

```python
"""Tests for newsradar.filename_dates."""

import unittest
from datetime import date

from newsradar.filename_dates import parse_date_from_stem


class TestParseDateFromStem(unittest.TestCase):
    def test_parses_valid_stem(self):
        result = parse_date_from_stem("ai-radar-2026-05-21", "ai-radar")
        self.assertEqual(result, date(2026, 5, 21))

    def test_parses_robotics_prefix(self):
        result = parse_date_from_stem("robotics-radar-2026-01-03", "robotics-radar")
        self.assertEqual(result, date(2026, 1, 3))

    def test_returns_none_for_unparseable_suffix(self):
        result = parse_date_from_stem("ai-radar-latest", "ai-radar")
        self.assertIsNone(result)

    def test_returns_none_for_too_few_parts(self):
        result = parse_date_from_stem("ai-radar-2026-05", "ai-radar")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/keithfry/projects/news-radar && uv run pytest tests/test_filename_dates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.filename_dates'`

- [ ] **Step 3: Write minimal implementation**

Create `news-radar/src/newsradar/filename_dates.py`:

```python
"""Shared filename-date parsing for radar episode files.

Episode files are named "<prefix>-YYYY-MM-DD" (e.g. "ai-radar-2026-05-21").
This is the single place that regex lives — podcast_rss.py and retention.py
both parse dates the same way.
"""

from datetime import date


def parse_date_from_stem(stem: str, prefix: str) -> date | None:
    """Parse YYYY-MM-DD out of a stem like 'ai-radar-2026-05-21'. None if unparseable."""
    suffix = stem[len(prefix):]
    parts = suffix.lstrip("-").split("-")
    if len(parts) >= 3:
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return date(year, month, day)
        except (ValueError, IndexError):
            pass
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/keithfry/projects/news-radar && uv run pytest tests/test_filename_dates.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Refactor podcast_rss.py to use the shared helper**

In `news-radar/src/newsradar/podcast_rss.py`, add the import near the top (after the existing `from .topics import Topic` line):

```python
from .filename_dates import parse_date_from_stem
```

Replace the body of `_date_from_stem` (lines 35-45) with:

```python
def _date_from_stem(stem: str, prefix: str) -> datetime | None:
    """Parse YYYY-MM-DD from stem like 'ai-radar-2026-05-21' or 'robotics-radar-2026-05-21'."""
    parsed = parse_date_from_stem(stem, prefix)
    if parsed is None:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, 8, 0, 0, tzinfo=timezone.utc)
```

- [ ] **Step 6: Run the existing podcast_rss tests to verify the refactor didn't break anything**

Run: `cd /Users/keithfry/projects/news-radar && uv run pytest tests/test_podcast_rss.py -v`
Expected: PASS (6 tests, same as before the refactor)

- [ ] **Step 7: Commit**

```bash
cd /Users/keithfry/projects/news-radar
git add src/newsradar/filename_dates.py src/newsradar/podcast_rss.py tests/test_filename_dates.py
git commit -m "Extract shared filename-date parsing helper"
```

---

### Task 2: Retention purge logic

**Files:**
- Create: `news-radar/src/newsradar/retention.py`
- Test: `news-radar/tests/test_retention.py`

**Interfaces:**
- Consumes: `parse_date_from_stem(stem: str, prefix: str) -> date | None` (Task 1)
- Produces:
  - `DEFAULT_AUDIO_MAX_AGE_DAYS = 90`, `DEFAULT_EPISODE_MAX_AGE_DAYS = 150` — module constants, the default thresholds consumers pass as `max_age_days`
  - `purge_audio(output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print) -> list[Path]`
  - `purge_episodes(output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print) -> list[Path]`

  Both take a topic's output directory (e.g. `techradar/AI`) laid out as `output_dir/YYYY-MM/<prefix>-YYYY-MM-DD.<ext>`, same layout `podcast_rss.py` and the main pipeline already use. Both return every path they deleted or modified — consumers (e.g. `techradar/retention.py`) use this list for their git-add and summary printout.

- [ ] **Step 1: Write the failing tests**

Create `news-radar/tests/test_retention.py`:

```python
"""Tests for newsradar.retention purge logic."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from newsradar.retention import purge_audio, purge_episodes

_AUDIO_EXTS = (".mp3", ".chapters.json", ".transcript.json", ".jpg", ".og.jpg")

_PLAYER_HTML = (
    '<div class="page-header">\n'
    '  <div class="header-stats">\n'
    '  </div>\n'
    '  <div class="podcast-player">\n'
    '    <audio controls preload="none" src="https://example.com/ai-radar-2026-01-01.mp3">'
    'Your browser does not support audio.</audio>\n'
    '  <a class="podcast-subscribe" href="https://example.com/podcast.rss">🎙 Subscribe</a>\n'
    '  </div>\n'
    '</div>\n'
)

_OG_META = (
    '<meta property="og:title" content="AI Daily Digest — January 1, 2026"/>\n'
    '<meta property="og:description" content="AI news digest — January 1, 2026"/>\n'
    '<meta property="og:image" content="https://example.com/ai-radar-2026-01-01.og.jpg"/>\n'
    '<meta property="og:image:width" content="1200"/>\n'
    '<meta property="og:image:height" content="630"/>\n'
    '<meta property="og:type" content="website"/>\n'
    '<meta name="twitter:card" content="summary_large_image"/>\n'
    '<meta name="twitter:image" content="https://example.com/ai-radar-2026-01-01.og.jpg"/>\n'
)


def _write_episode(month_dir: Path, date_str: str, prefix: str = "ai-radar", with_html: bool = True) -> None:
    month_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}-{date_str}"
    for ext in _AUDIO_EXTS:
        (month_dir / f"{stem}{ext}").write_bytes(b"x")
    (month_dir / f"{stem}.json").write_text("{}")
    if with_html:
        html = f"<html>\n<head>\n{_OG_META}</head>\n<body>\n{_PLAYER_HTML}</body>\n</html>\n"
        (month_dir / f"{stem}.html").write_text(html)


class TestPurgeAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.output_dir = Path(self.tmp.name)
        self.now = datetime(2026, 6, 1, tzinfo=timezone.utc)  # "today" for all tests

    def tearDown(self):
        self.tmp.cleanup()

    def test_purges_audio_files_older_than_max_age(self):
        # 2026-01-01 is 151 days before 2026-06-01 — well past the 90-day threshold.
        _write_episode(self.output_dir / "2026-01", "2026-01-01")

        changed = purge_audio(self.output_dir, "ai-radar", 90, self.now)

        stem_dir = self.output_dir / "2026-01"
        for ext in _AUDIO_EXTS:
            self.assertFalse((stem_dir / f"ai-radar-2026-01-01{ext}").exists(), f"{ext} should be deleted")
        self.assertTrue((stem_dir / "ai-radar-2026-01-01.json").exists(), "json must survive")
        self.assertTrue((stem_dir / "ai-radar-2026-01-01.html").exists(), "html must survive")
        self.assertEqual(len(changed), len(_AUDIO_EXTS) + 1)  # 5 deleted files + 1 modified html

    def test_keeps_audio_files_within_max_age(self):
        # 2026-05-15 is 17 days before 2026-06-01 — inside the 90-day window.
        _write_episode(self.output_dir / "2026-05", "2026-05-15")

        changed = purge_audio(self.output_dir, "ai-radar", 90, self.now)

        stem_dir = self.output_dir / "2026-05"
        for ext in _AUDIO_EXTS:
            self.assertTrue((stem_dir / f"ai-radar-2026-05-15{ext}").exists())
        self.assertEqual(changed, [])

    def test_strips_player_and_og_meta_from_html(self):
        _write_episode(self.output_dir / "2026-01", "2026-01-01")

        purge_audio(self.output_dir, "ai-radar", 90, self.now)

        html = (self.output_dir / "2026-01" / "ai-radar-2026-01-01.html").read_text()
        self.assertNotIn("podcast-player", html)
        self.assertNotIn("og:image", html)
        self.assertNotIn("twitter:image", html)
        self.assertIn("<html>", html)  # rest of the page survives

    def test_idempotent_on_second_run(self):
        _write_episode(self.output_dir / "2026-01", "2026-01-01")
        purge_audio(self.output_dir, "ai-radar", 90, self.now)

        changed = purge_audio(self.output_dir, "ai-radar", 90, self.now)

        self.assertEqual(changed, [])  # nothing left to delete or strip

    def test_boundary_exactly_at_max_age_is_not_purged(self):
        # 2026-03-03 is exactly 90 days before 2026-06-01.
        _write_episode(self.output_dir / "2026-03", "2026-03-03")

        changed = purge_audio(self.output_dir, "ai-radar", 90, self.now)

        self.assertEqual(changed, [])


class TestPurgeEpisodes(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.output_dir = Path(self.tmp.name)
        self.now = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_purges_all_files_older_than_max_age_and_removes_empty_dir(self):
        # 2026-01-01 is 151 days before 2026-06-01 — past the 150-day threshold.
        month_dir = self.output_dir / "2026-01"
        _write_episode(month_dir, "2026-01-01")

        changed = purge_episodes(self.output_dir, "ai-radar", 150, self.now)

        self.assertFalse(month_dir.exists(), "empty month dir should be removed")
        self.assertEqual(len(changed), len(_AUDIO_EXTS) + 2)  # 5 audio + .json + .html

    def test_keeps_dir_when_another_episode_still_present(self):
        month_dir = self.output_dir / "2026-01"
        _write_episode(month_dir, "2026-01-01")  # 151 days old — purged
        _write_episode(month_dir, "2026-01-31")  # 121 days old — kept

        purge_episodes(self.output_dir, "ai-radar", 150, self.now)

        self.assertTrue(month_dir.exists())
        self.assertFalse((month_dir / "ai-radar-2026-01-01.json").exists())
        self.assertTrue((month_dir / "ai-radar-2026-01-31.json").exists())

    def test_keeps_files_within_max_age(self):
        month_dir = self.output_dir / "2026-05"
        _write_episode(month_dir, "2026-05-15")  # 17 days old

        changed = purge_episodes(self.output_dir, "ai-radar", 150, self.now)

        self.assertEqual(changed, [])
        self.assertTrue(month_dir.exists())

    def test_purges_episode_with_audio_already_purged(self):
        # Simulates an episode that already went through purge_audio: only
        # html + json remain, both must still be caught by purge_episodes.
        month_dir = self.output_dir / "2026-01"
        _write_episode(month_dir, "2026-01-01")
        for ext in _AUDIO_EXTS:
            (month_dir / f"ai-radar-2026-01-01{ext}").unlink()

        changed = purge_episodes(self.output_dir, "ai-radar", 150, self.now)

        self.assertFalse(month_dir.exists())
        self.assertEqual(len(changed), 2)  # .json + .html


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/keithfry/projects/news-radar && uv run pytest tests/test_retention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.retention'`

- [ ] **Step 3: Write the implementation**

Create `news-radar/src/newsradar/retention.py`:

```python
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
_OG_META_RE = re.compile(
    r'<meta property="og:title".*?<meta name="twitter:image"[^>]*/>\n',
    re.DOTALL,
)


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
    new_text = _OG_META_RE.sub("", new_text)
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
                f.unlink()
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

        for f in sorted(month_dir.glob(f"{stem}*")):
            f.unlink()
            log(f"    deleted: {f}")
            changed.append(f)
        touched_dirs.add(month_dir)

    for month_dir in touched_dirs:
        if month_dir.is_dir() and not any(month_dir.iterdir()):
            month_dir.rmdir()
            log(f"    removed empty dir: {month_dir}")

    return changed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/keithfry/projects/news-radar && uv run pytest tests/test_retention.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/keithfry/projects/news-radar
git add src/newsradar/retention.py tests/test_retention.py
git commit -m "Add retention purge logic for old radar episodes"
```

