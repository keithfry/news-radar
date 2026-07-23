"""Tests for newsradar.podcast_rss RSS generation logic.

Updated for the config-driven signature: `generate_podcast_rss(output_dir, topic,
base_url, author_name, output_dir_rel, log)` — there's no more module-level
BASE_URL_ROOT constant, base_url/output_dir_rel are passed explicitly, and
build_rss_feed expects episode files laid out under a YYYY-MM subdirectory
(matching how the real pipeline writes output_root/<topic>/<YYYY-MM>/<file>).
"""

import json
import tempfile
import unittest
from pathlib import Path

from newsradar.podcast_rss import _duration_from_chapters, build_rss_feed, generate_podcast_rss
from newsradar.topics import Topic

_TOPIC = Topic(
    name="ai",
    display_name="AI",
    output_dir="ai",
    file_prefix="ai-radar",
    classifier_prompt="unused in these tests",
)


class TestRSSGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

        for date_str, ym in [("2026-05-21", "2026-05"), ("2026-05-20", "2026-05")]:
            month_dir = self.tmpdir / ym
            month_dir.mkdir(parents=True, exist_ok=True)

            mp3 = month_dir / f"ai-radar-{date_str}.mp3"
            mp3.write_bytes(b"\xff\xfb" * 100)  # fake MP3 bytes

            chap_json = month_dir / f"ai-radar-{date_str}.chapters.json"
            chap_json.write_text(json.dumps({
                "version": "1.2.0",
                "chapters": [
                    {"startTime": 0, "endTime": 22, "title": "Introduction"},
                    {"startTime": 22, "endTime": 600, "title": "Top story"},
                ]
            }))

    def tearDown(self):
        self.tmp.cleanup()

    def test_rss_contains_both_items(self):
        xml = build_rss_feed(self.tmpdir, "https://example.com/techradar/AI", "ai-radar", "AI", "Jane Example")
        self.assertIn("<rss", xml)
        self.assertIn("ai-radar-2026-05-21.mp3", xml)
        self.assertIn("ai-radar-2026-05-20.mp3", xml)

    def test_rss_limits_to_max_episodes(self):
        xml = build_rss_feed(
            self.tmpdir, "https://example.com/techradar/AI", "ai-radar", "AI", "Jane Example",
            max_episodes=1,
        )
        self.assertIn("2026-05-21", xml)
        self.assertNotIn("2026-05-20", xml)

    def test_rss_has_required_namespaces(self):
        xml = build_rss_feed(self.tmpdir, "https://example.com/techradar/AI", "ai-radar", "AI", "Jane Example")
        self.assertIn("xmlns:itunes", xml)
        self.assertIn("xmlns:podcast", xml)

    def test_rss_duration_from_chapters(self):
        chap_path = self.tmpdir / "2026-05" / "ai-radar-2026-05-21.chapters.json"
        duration = _duration_from_chapters(chap_path)
        self.assertEqual(duration, 600)  # endTime of last chapter

    def test_rss_chapters_tag_present(self):
        xml = build_rss_feed(self.tmpdir, "https://example.com/techradar/AI", "ai-radar", "AI", "Jane Example")
        self.assertIn("podcast:chapters", xml)
        self.assertIn(".chapters.json", xml)

    def test_generate_podcast_rss_writes_file(self):
        logs = []
        out = generate_podcast_rss(
            self.tmpdir, _TOPIC,
            base_url="https://example.com/techradar",
            author_name="Jane Example",
            output_dir_rel="ai",
            log=logs.append,
        )
        self.assertEqual(out, self.tmpdir / "podcast.rss")
        self.assertTrue(out.exists())
        self.assertIn("<rss", out.read_text())
        self.assertTrue(any("podcast.rss" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
