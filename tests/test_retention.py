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

    def test_keeps_og_title_and_downgrades_twitter_card(self):
        _write_episode(self.output_dir / "2026-01", "2026-01-01")

        purge_audio(self.output_dir, "ai-radar", 90, self.now)

        html = (self.output_dir / "2026-01" / "ai-radar-2026-01-01.html").read_text()
        self.assertIn("og:title", html)
        self.assertIn("og:description", html)
        self.assertNotIn("summary_large_image", html)
        self.assertIn('<meta name="twitter:card" content="summary"/>', html)

    def test_ignores_other_prefixes_and_undated_files(self):
        _write_episode(self.output_dir / "2026-01", "2026-01-01", prefix="robotics-radar")
        cover = self.output_dir / "podcast-cover.png"
        cover.write_bytes(b"x")

        changed = purge_audio(self.output_dir, "ai-radar", 90, self.now)

        self.assertEqual(changed, [])
        self.assertTrue(cover.exists())
        self.assertTrue((self.output_dir / "2026-01" / "robotics-radar-2026-01-01.mp3").exists())


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

    def test_ignores_other_prefixes_and_undated_files(self):
        month_dir = self.output_dir / "2026-01"
        _write_episode(month_dir, "2026-01-01")
        _write_episode(month_dir, "2026-01-01", prefix="robotics-radar")
        cover = self.output_dir / "podcast-cover.png"
        cover.write_bytes(b"x")

        purge_episodes(self.output_dir, "ai-radar", 150, self.now)

        self.assertTrue(cover.exists())
        self.assertTrue(month_dir.exists())
        self.assertFalse((month_dir / "ai-radar-2026-01-01.html").exists())
        self.assertTrue((month_dir / "robotics-radar-2026-01-01.html").exists())


if __name__ == "__main__":
    unittest.main()
