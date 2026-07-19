"""Tests for newsradar.podcast_generator — pure logic, no TTS or ffmpeg needed."""

import json
import unittest

from newsradar.podcast_generator import _build_chapters_json, _voice_for
from newsradar.enricher import KOKORO_VOICES


class TestChaptersJson(unittest.TestCase):
    def test_chapters_json_structure(self):
        chapters = [
            {"startTime": 0, "title": "Introduction"},
            {"startTime": 22, "title": "Google releases Gemini 2.5 Flash"},
        ]
        result = _build_chapters_json(chapters)
        data = json.loads(result)
        self.assertEqual(data["version"], "1.2.0")
        self.assertEqual(len(data["chapters"]), 2)
        self.assertEqual(data["chapters"][0]["startTime"], 0)
        self.assertEqual(data["chapters"][1]["title"], "Google releases Gemini 2.5 Flash")

    def test_chapters_json_empty(self):
        data = json.loads(_build_chapters_json([]))
        self.assertEqual(data["version"], "1.2.0")
        self.assertEqual(data["chapters"], [])


class TestVoiceAssignment(unittest.TestCase):
    def test_voice_index_wraps_around(self):
        for i in range(10):
            voice = _voice_for(i)
            self.assertEqual(voice, KOKORO_VOICES[i % len(KOKORO_VOICES)])

    def test_voice_for_excludes_banned_voices(self):
        all_assigned = {_voice_for(i) for i in range(20)}
        self.assertNotIn("am_adam", all_assigned)
        self.assertNotIn("af_nicole", all_assigned)


if __name__ == "__main__":
    unittest.main()
