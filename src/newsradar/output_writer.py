"""Write digest output files. No git/publish awareness — see hooks/ examples for that."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .config import Config

PublishHook = Callable[[list[Path], Config], None]


def save_html(html: str, date: datetime, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{date.strftime('%Y-%m-%d')}.html"
    out_path = output_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path


def save_json(data: dict, date: datetime, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{date.strftime('%Y-%m-%d')}.json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out_path


def write_outputs(
    paths: list[Path], config: Config, publish_hook: PublishHook | None = None
) -> None:
    """Invoke the publish hook (if any) with the written output paths. No-op otherwise."""
    if publish_hook is not None:
        publish_hook(paths, config)
