"""Generate podcast.rss RSS feed from radar MP3 files.

Scans output_dir for MP3 files matching file_prefix, reads paired
.chapters.json for duration, writes podcast.rss. Keeps last 20 episodes.
"""

import json
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

from .topics import Topic

MAX_EPISODES = 20


def _duration_from_chapters(chapters_path: Path) -> int:
    try:
        data = json.loads(chapters_path.read_text())
        chapters = data.get("chapters", [])
        if chapters:
            return int(chapters[-1].get("endTime", 0))
    except Exception:
        pass
    return 0


def _tagline_from_chapters(chapters_path: Path) -> str:
    try:
        return json.loads(chapters_path.read_text()).get("title", "")
    except Exception:
        return ""


def _date_from_stem(stem: str, prefix: str) -> datetime | None:
    """Parse YYYY-MM-DD from stem like 'ai-radar-2026-05-21' or 'robotics-radar-2026-05-21'."""
    suffix = stem[len(prefix):]  # e.g. "2026-05-21"
    parts = suffix.lstrip("-").split("-")
    if len(parts) >= 3:
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(y, m, d, 8, 0, 0, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            pass
    return None


def build_rss_feed(
    output_dir: Path,
    base_url: str,
    file_prefix: str,
    topic_label: str,
    author_name: str,
    max_episodes: int = MAX_EPISODES,
) -> str:
    mp3_files = sorted(output_dir.glob(f"**/{file_prefix}-*.mp3"), reverse=True)[:max_episodes]

    items_xml = []
    for mp3 in mp3_files:
        date = _date_from_stem(mp3.stem, file_prefix)
        if not date:
            continue

        date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
        ym_dir = f"{date.year:04d}-{date.month:02d}"
        chap_json = output_dir / ym_dir / f"{file_prefix}-{date_str}.chapters.json"
        duration = _duration_from_chapters(chap_json) if chap_json.exists() else 0
        file_size = mp3.stat().st_size
        mp3_url = f"{base_url}/{ym_dir}/{file_prefix}-{date_str}.mp3"
        chap_url = f"{base_url}/{ym_dir}/{file_prefix}-{date_str}.chapters.json"
        pub_date = format_datetime(date)
        title_date = f"{date.strftime('%B')} {date.day}, {date.year}"

        tagline = _tagline_from_chapters(chap_json) if chap_json.exists() else ""
        if tagline:
            episode_title = tagline
        else:
            episode_title = f"{topic_label} Radar — {title_date}"

        chap_tag = (
            f'      <podcast:chapters url="{chap_url}" type="application/json+chapters"/>\n'
            if chap_json.exists() else ""
        )

        transcript_path = chap_json.parent / chap_json.name.replace(".chapters.json", ".transcript.json")
        transcript_url = chap_url.replace(".chapters.json", ".transcript.json")
        transcript_tag = (
            f'      <podcast:transcript url="{transcript_url}" type="application/json"/>\n'
            if transcript_path.exists() else ""
        )

        ep_cover_path = output_dir / ym_dir / f"{file_prefix}-{date_str}.jpg"
        ep_cover_tag = (
            f'      <itunes:image href="{base_url}/{ym_dir}/{file_prefix}-{date_str}.jpg"/>\n'
            if ep_cover_path.exists() else ""
        )

        items_xml.append(f"""  <item>
    <title>{episode_title}</title>
    <pubDate>{pub_date}</pubDate>
    <enclosure url="{mp3_url}" type="audio/mpeg" length="{file_size}"/>
    <itunes:duration>{duration // 3600:02d}:{(duration % 3600) // 60:02d}:{duration % 60:02d}</itunes:duration>
    <guid isPermaLink="true">{mp3_url}</guid>
    <description>{topic_label} Radar for {title_date}. {duration // 60} minutes of news.</description>
{ep_cover_tag}{chap_tag}{transcript_tag}  </item>""")

    items_block = "\n".join(items_xml)
    now = format_datetime(datetime.now(timezone.utc))

    feed_url = f"{base_url}/podcast.rss"
    channel_cover_tag = ""
    channel_cover_path = output_dir / "podcast-cover.png"
    if channel_cover_path.exists():
        channel_cover_tag = f'    <itunes:image href="{base_url}/podcast-cover.png"/>\n'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
  xmlns:podcast="https://podcastindex.org/namespace/1.0"
  xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{topic_label} Daily Radar</title>
    <link>{base_url}/</link>
    <atom:link href="{feed_url}" rel="self" type="application/rss+xml"/>
    <description>Daily {topic_label} news digest in audio form. New episode each weekday.</description>
    <language>en-us</language>
    <lastBuildDate>{now}</lastBuildDate>
    <itunes:author>{author_name}</itunes:author>
    <itunes:category text="Technology">
      <itunes:category text="Tech News"/>
    </itunes:category>
    <itunes:explicit>false</itunes:explicit>
{channel_cover_tag}{items_block}
  </channel>
</rss>
"""


def generate_podcast_rss(
    output_dir: Path,
    topic: Topic,
    base_url: str,
    author_name: str,
    output_dir_rel: str,
    log=print,
) -> Path:
    """Generate podcast.rss for the given topic output directory. Returns path written."""
    full_base_url = f"{base_url}/{output_dir_rel}"

    xml = build_rss_feed(
        output_dir,
        full_base_url,
        topic.file_prefix,
        topic.display_name,
        author_name,
    )
    out = output_dir / "podcast.rss"
    out.write_text(xml, encoding="utf-8")
    episode_count = xml.count("<item>")
    log(f"  podcast.rss: {episode_count} episodes → {out}")
    return out
