"""Tests for a topic's `exclude_topics` list (e.g. AI handing robotics off)."""

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from newsradar import cli
from newsradar.config import load_config


def _write_config(tmp: Path, ai_extra: str) -> Path:
    topics = tmp / "topics"
    topics.mkdir()
    (topics / "ai.toml").write_text('[topic]\nname = "ai"\nclassifier_prompt = "p"\n' + ai_extra)
    (topics / "robotics.toml").write_text('[topic]\nname = "robotics"\nclassifier_prompt = "strict"\n')
    cfg = tmp / "config.toml"
    cfg.write_text('[topics]\nfiles = ["topics/ai.toml", "topics/robotics.toml"]\n')
    return cfg


class TestExcludeTopicsConfig(unittest.TestCase):
    def test_default_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_config(_write_config(Path(d), "")).topic("ai").exclude_topics, [])

    def test_reads_list(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(_write_config(Path(d), 'exclude_topics = ["robotics"]\n'))
        self.assertEqual(cfg.topic("ai").exclude_topics, ["robotics"])

    def test_rejects_unknown_topic(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_config(_write_config(Path(d), 'exclude_topics = ["space"]\n'))


class TestApplyExclusions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = load_config(_write_config(Path(self.tmp.name), 'exclude_topics = ["robotics"]\n'))
        self.ai = self.config.topic("ai")

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, tags, robotics_verdict):
        calls = []

        def fake_classify(title, summary, model, topic):
            calls.append(topic.name)
            return robotics_verdict

        with mock.patch.object(cli, "classify_topic", fake_classify):
            result = cli._apply_exclusions("t", "s", tags, self.ai, self.config)
        return result, calls

    def test_untagged_item_skips_the_classifier(self):
        (tags, excluded), calls = self.run_with(["model"], True)
        self.assertEqual((tags, excluded, calls), (["model"], None, []))

    def test_robotics_item_is_excluded(self):
        (tags, excluded), calls = self.run_with(["robotics", "model"], True)
        self.assertEqual(excluded, "robotics")
        self.assertEqual(calls, ["robotics"])

    def test_kept_item_loses_the_robotics_tag(self):
        (tags, excluded), _ = self.run_with(["model", "robotics"], False)
        self.assertIsNone(excluded)
        self.assertEqual(tags, ["model"])


if __name__ == "__main__":
    unittest.main()
