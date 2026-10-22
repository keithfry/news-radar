"""Topic model — defines a single digest topic (e.g. "ai", "robotics")."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelsConfig

PAPERS_MODES = ("include", "exclude", "only")


@dataclass
class Topic:
    """A single digest topic, fully defined by config — nothing is hardcoded in code.

    classifier_prompt is the full LLM instruction block used to decide whether a
    piece of content belongs in this topic's digest (criteria + examples). keywords
    is an optional fast-path: if non-empty, content matching none of the keywords
    skips straight to the LLM classifier rather than being auto-accepted/rejected.

    models and feeds_csv are optional per-topic overrides of the base config's
    values — None means "use the base config's value" (see config.effective_models).

    papers controls arXiv items:
      "include" keeps papers in their own page section, out of the podcast (default).
      "exclude" drops papers before any LLM work.
      "only" keeps only papers and narrates them in the podcast.

    podcast_max_items caps how many ranked items get narrated. The rest stay on
    the page and in the JSON without audio. None narrates every candidate.

    exclude_topics names other topics whose content this topic drops. An item
    tagged with one of those names goes through that topic's classifier, and is
    dropped if the classifier accepts it. Otherwise the item stays without the tag.
    """

    name: str
    display_name: str
    output_dir: str
    file_prefix: str
    classifier_prompt: str
    keywords: list[str] = field(default_factory=list)
    fail_open: bool = True
    models: "ModelsConfig | None" = None
    feeds_csv: Path | None = None
    papers: str = "include"
    podcast_max_items: int | None = None
    exclude_topics: list[str] = field(default_factory=list)

    # Cover-image theme (RGB tuples). Defaults match the original "AI" blue theme.
    accent_color: tuple[int, int, int] = (96, 165, 250)       # #60a5fa
    accent_color_light: tuple[int, int, int] = (147, 197, 253)  # #93c5fd
    bg_top_color: tuple[int, int, int] = (15, 23, 42)          # #0f172a
    bg_mid_color: tuple[int, int, int] = (30, 58, 95)           # #1e3a5f
