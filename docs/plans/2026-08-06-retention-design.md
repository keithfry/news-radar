# Retention design

Date: 2026-08-06 (moved here from the `techradar` repo 2026-09-20 — retention
rules are core `newsradar` behavior, reusable by any site built on it)

## Problem

A topic's output directory accumulates one episode's worth of files every
run (`.mp3`, `.chapters.json`, `.transcript.json`, `.jpg`, `.og.jpg`,
`.html`, `.json`) and nothing ever deletes them. This grows the consuming
repo (and its Pages build) unboundedly. `newsradar` provides the purge logic;
each consuming site decides when to run it and what to do afterwards
(regenerate indexes, commit, push).

## Requirements

Old content is pruned on two timers. Both thresholds are defaults, exported
as constants from `newsradar.retention`; callers may override per call.

- **3 months (90 days, `DEFAULT_AUDIO_MAX_AGE_DAYS`)**: purge audio-related
  artifacts — mp3, chapters, transcript, cover images. Keep the episode's
  HTML article page and JSON data alive.
- **5 months (150 days, `DEFAULT_EPISODE_MAX_AGE_DAYS`)**: purge everything
  left for that episode — HTML, JSON, and any leftover audio files (covers
  backfilled/edge-case episodes where audio purge didn't already run).
  Remove the `YYYY-MM` directory if it ends up empty.

Age is computed from the date embedded in the filename (e.g.
`ai-radar-2026-05-01`), not file mtime — mtimes get touched by backfills and
git operations; the filename date is the authoritative "episode date."

## Component: `src/newsradar/retention.py` (new module)

Pure, reusable purge logic — no git, no CLI, just filesystem + HTML edits.
Reuses the filename-date regex already in `podcast_rss._date_from_stem`
(promoted to a shared helper, `filename_dates.parse_date_from_stem`, that
both modules import, to avoid duplicating it).

- `DEFAULT_AUDIO_MAX_AGE_DAYS = 90`, `DEFAULT_EPISODE_MAX_AGE_DAYS = 150`
  Module constants. Consumers import these instead of hardcoding their own
  thresholds.

- `purge_audio(output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print) -> list[Path]`
  For each dated episode older than `max_age_days`:
  - Delete `.mp3`, `.chapters.json`, `.transcript.json`, `.jpg`, `.og.jpg`
    if present.
  - If the episode's `.html` exists, strip the inline podcast player block
    (`_podcast_player_html`'s output — the `<div class="podcast-player">…
    </div>`, matched structurally) and the `og:image` (including its
    `og:image:width`/`og:image:height` companions) and `twitter:image` meta
    lines. `og:title`/`og:description` stay, and `twitter:card` is downgraded
    from `summary_large_image` to `summary` since there's no image left.
    Idempotent — if already stripped (e.g. rerunning the job), no-op.
  - Returns every path deleted or modified, for the caller's git-add list
    and summary printout.

- `purge_episodes(output_dir: Path, file_prefix: str, max_age_days: int, now: datetime, log=print) -> list[Path]`
  For each dated episode older than `max_age_days`: delete every remaining
  file for that date stem (`.html`, `.json`, and any audio files that
  survived because `purge_audio` hadn't run on them yet). After processing
  a `YYYY-MM` directory, remove it if empty. Returns deleted paths.

Both functions are per-topic-output-dir (caller loops topics), and both
take `now` as a parameter rather than calling `datetime.now()` internally,
so they're deterministic and testable.

## Consumer contract

After purging, a consumer is expected to:

- Regenerate `podcast.rss` via `newsradar.podcast_rss.generate_podcast_rss`
  — it globs mp3s fresh off disk, so purged episodes fall out with no manual
  list-editing.
- Regenerate whatever site indexes it maintains, and commit/push if it
  publishes from git. These are site-specific and live in the consumer (see
  the `techradar` repo's `docs/plans/2026-08-06-retention-job-design.md`).

## Error handling

- Missing files (already deleted, or a file that never existed for an
  episode) are treated as no-ops, not errors — `Path.unlink(missing_ok=True)`.

## Testing

- Unit tests for `purge_audio`/`purge_episodes` against a temp directory
  seeded with fake dated episode files at various ages, asserting which
  files survive/are deleted and that HTML stripping is idempotent.

## Out of scope

- No change to `MAX_EPISODES = 20` cap in `podcast_rss.py` — that's a
  separate, already-existing limit on RSS feed size, unrelated to on-disk
  retention.
- No CLI, scheduling, or git handling — consumers own those.
