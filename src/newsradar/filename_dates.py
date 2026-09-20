"""Shared filename-date parsing for radar episode files.

Episode files are named "<prefix>-YYYY-MM-DD" (e.g. "ai-radar-2026-05-21").
This is the single place that parsing lives — podcast_rss.py and retention.py
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
