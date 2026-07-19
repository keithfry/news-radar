"""Topic model — defines a single digest topic (e.g. "ai", "robotics")."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Topic:
    """A single digest topic, fully defined by config — nothing is hardcoded in code.

    classifier_prompt is the full LLM instruction block used to decide whether a
    piece of content belongs in this topic's digest (criteria + examples). keywords
    is an optional fast-path: if non-empty, content matching none of the keywords
    skips straight to the LLM classifier rather than being auto-accepted/rejected.
    """

    name: str
    display_name: str
    feed_category: str
    output_dir: str
    file_prefix: str
    classifier_prompt: str
    keywords: list[str] = field(default_factory=list)
    fail_open: bool = True

    # Cover-image theme (RGB tuples). Defaults match the original "AI" blue theme.
    accent_color: tuple[int, int, int] = (96, 165, 250)       # #60a5fa
    accent_color_light: tuple[int, int, int] = (147, 197, 253)  # #93c5fd
    bg_top_color: tuple[int, int, int] = (15, 23, 42)          # #0f172a
    bg_mid_color: tuple[int, int, int] = (30, 58, 95)           # #1e3a5f
