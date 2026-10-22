"""Tests for the per-topic `papers` mode and `podcast_max_items` cap."""

import tempfile
import textwrap
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from newsradar import llm
from newsradar.cli import _filter_for_topic
from newsradar.config import load_config
from newsradar.enricher import enrich
from newsradar.topics import Topic

ET = ZoneInfo("America/New_York")


def _topic(**kw) -> Topic:
    base = dict(name="t", display_name="T", output_dir="T", file_prefix="t-radar", classifier_prompt="p")
    base.update(kw)
    return Topic(**base)


def _write_config(tmp: Path, topic_extra: str) -> Path:
    (tmp / "topics").mkdir()
    (tmp / "topics" / "t.toml").write_text(
        textwrap.dedent(
            """\
            [topic]
            name = "t"
            classifier_prompt = "p"
            """
        )
        + topic_extra
    )
    cfg = tmp / "config.toml"
    cfg.write_text('[topics]\nfiles = ["topics/t.toml"]\n')
    return cfg


class TestTopicPapersConfig(unittest.TestCase):
    def test_defaults_keep_current_behavior(self):
        with tempfile.TemporaryDirectory() as d:
            topic = load_config(_write_config(Path(d), "")).topic("t")
        self.assertEqual(topic.papers, "include")
        self.assertIsNone(topic.podcast_max_items)

    def test_reads_papers_and_podcast_cap(self):
        with tempfile.TemporaryDirectory() as d:
            topic = load_config(_write_config(Path(d), 'papers = "only"\npodcast_max_items = 10\n')).topic("t")
        self.assertEqual(topic.papers, "only")
        self.assertEqual(topic.podcast_max_items, 10)

    def test_rejects_unknown_papers_mode(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_config(_write_config(Path(d), 'papers = "sometimes"\n'))

    def test_rejects_non_positive_podcast_cap(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_config(_write_config(Path(d), "podcast_max_items = 0\n"))


class TestFilterForTopic(unittest.TestCase):
    ITEMS = [
        {"title": "news", "source": "The Decoder"},
        {"title": "paper", "source": "arXiv - cs.AI"},
        {"title": "linked paper", "source": "arxiv.org"},
    ]

    def titles(self, topic):
        return [i["title"] for i in _filter_for_topic(self.ITEMS, topic)]

    def test_include_keeps_everything(self):
        self.assertEqual(self.titles(_topic()), ["news", "paper", "linked paper"])

    def test_exclude_drops_papers(self):
        self.assertEqual(self.titles(_topic(papers="exclude")), ["news"])

    def test_only_keeps_papers(self):
        self.assertEqual(self.titles(_topic(papers="only")), ["paper", "linked paper"])


def _item(title: str, arxiv: bool) -> dict:
    return {
        "title": title, "link": "", "source": "arXiv - cs.AI" if arxiv else "Blog",
        "summary": "s", "tags": [], "_source_type": "rss", "_is_arxiv": arxiv,
    }


class TestEnrichPapers(unittest.TestCase):
    """enrich() with the LLM calls stubbed out: ranking keeps input order."""

    def run_enrich(self, items, topic):
        stubs = {
            "rank_items": lambda its, model: [dict(it, rank=n) for n, it in enumerate(its, 1)],
            "generate_audio_script": lambda item, model: f"script {item['title']}",
            "generate_intro_script": lambda its, date, model, topic: f"intro {len(its)}",
            "generate_outro_script": lambda its, date, model, topic: "outro",
            "generate_episode_tagline": lambda its, model: "tagline",
            "unload_all_models": lambda: [],
        }
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(llm, **stubs):
            return enrich(
                items, datetime(2026, 10, 1, 8, 0, tzinfo=ET), Path(d) / "out.json",
                summarize_model="m", rank_model="m", topic=topic, llm_workers=1, log=lambda *_: None,
            )

    def test_include_mode_leaves_papers_out_of_podcast(self):
        data = self.run_enrich([_item("a", False), _item("p", True)], _topic())
        podcast = [i["title"] for i in data["items"] if i["include_in_podcast"]]
        self.assertEqual(podcast, ["a"])

    def test_only_mode_narrates_papers_up_to_cap(self):
        items = [_item(f"p{n}", True) for n in range(5)]
        data = self.run_enrich(items, _topic(papers="only", podcast_max_items=2))
        podcast = [i for i in data["items"] if i["include_in_podcast"]]
        self.assertEqual([i["title"] for i in podcast], ["p0", "p1"])
        self.assertEqual(data["intro_script"], "intro 2")
        # Every paper stays in the JSON; the ones past the cap have no audio.
        self.assertEqual(len(data["items"]), 5)
        rest = [i for i in data["items"] if not i["include_in_podcast"]]
        self.assertTrue(all(i["chapter_start_seconds"] is None for i in rest))
        self.assertTrue(all("audio_script" not in i for i in rest))


if __name__ == "__main__":
    unittest.main()
