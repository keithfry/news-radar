"""Private files the tuner learns into: examples, rules, dedup cases, snapshots."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import date as _date, timedelta
from pathlib import Path
from urllib.parse import urlparse

DEDUP_CASES_IN_PROMPT = 12
_MODELFILE_NAME = "Modelfile"
_DEDUP_HEADING = "LEARNED EXAMPLES (past mistakes to avoid):"
AD_EXAMPLES_FILE = "learned_ad_examples.json"
AD_RULES_FILE = "learned_ad_rules.txt"
DEDUP_CASES_FILE = "learned_dedup_cases.json"
_DATED_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})(?:-(\d+))?")


def _dated(name: str) -> tuple[_date, int] | None:
    """Read "2026-11-23" or a same-day "2026-11-23-2" as (date, n). None if it doesn't parse."""
    m = _DATED_NAME.fullmatch(name)
    if not m:
        return None
    try:
        return _date.fromisoformat(m.group(1)), int(m.group(2) or 1)
    except ValueError:
        return None


def domain_only(url: str) -> str:
    """Cut a link to scheme://host/ with no port. Returns "" for anything else; never raises."""
    try:
        parts = urlparse((url or "").strip())
        host = parts.hostname
    except (ValueError, TypeError, AttributeError):
        return ""
    if not parts.scheme or not host:
        return ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    return f"{parts.scheme}://{host}/"


class TuningStore:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "tuning"
        self._ad_examples = self.root / AD_EXAMPLES_FILE
        self._ad_rules = self.root / AD_RULES_FILE
        self._dedup_cases = self.root / DEDUP_CASES_FILE

    def _read_json(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, text: str) -> None:
        """Write to a temp file in the same folder, then rename it over path, so
        a process killed mid-write never leaves a truncated file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _write_json(self, path: Path, data: list[dict]) -> None:
        self._write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def ad_examples(self) -> list[dict]:
        return self._read_json(self._ad_examples)

    def save_ad_examples(self, examples: list[dict]) -> None:
        self._write_json(self._ad_examples, examples)

    def ad_rules(self) -> str:
        return self._ad_rules.read_text(encoding="utf-8") if self._ad_rules.exists() else ""

    def save_ad_rules(self, text: str) -> None:
        self._write(self._ad_rules, text)

    def dedup_cases(self) -> list[dict]:
        return self._read_json(self._dedup_cases)

    def save_dedup_cases(self, cases: list[dict]) -> None:
        self._write_json(self._dedup_cases, cases)

    def dedup_examples_block(self) -> str:
        return self.dedup_block_for(self.dedup_cases())

    @staticmethod
    def dedup_block_for(cases: list[dict]) -> str:
        """Format cases as the dedup prompt's learned block, without saving them."""
        if not cases:
            return ""
        # sorted() is stable, so cases without "added" keep their stored order.
        newest = sorted(cases, key=lambda c: c.get("added", ""))[-DEDUP_CASES_IN_PROMPT:]
        parts = [_DEDUP_HEADING]
        for case in newest:
            titles = "\n".join(f"{i}. {t}" for i, t in enumerate(case["titles"]))
            groups = json.dumps(case["expected_groups"])
            parts.append(f"Example articles:\n{titles}\nExpected duplicate groups: {groups}")
        return "\n\n".join(parts) + "\n"

    def _snapshots(self) -> list[tuple[tuple[_date, int], Path]]:
        """Every snapshot folder whose name parses, with its (date, suffix number)."""
        base = self.root / "snapshots"
        if not base.is_dir():
            return []
        found = []
        for d in base.iterdir():
            dated = _dated(d.name) if d.is_dir() else None
            if dated is not None:
                found.append((dated, d))
        return sorted(found, key=lambda s: s[0])

    def _snapshots_on(self, date: str) -> list[tuple[int, Path]]:
        day = _date.fromisoformat(date)
        return [(n, d) for (d_day, n), d in self._snapshots() if d_day == day]

    def snapshot(self, date: str, built_modelfile: str) -> Path:
        """Copy the learned files and built Modelfile into a new snapshot folder.
        A later snapshot on the same day gets a suffix above every existing one."""
        same_day = self._snapshots_on(date)
        n = max(n for n, _ in same_day) + 1 if same_day else 1
        path = self.root / "snapshots" / (date if n == 1 else f"{date}-{n}")
        path.mkdir(parents=True)
        for src in self._learned_files():
            if src.exists():
                shutil.copy2(src, path / src.name)
        (path / _MODELFILE_NAME).write_text(built_modelfile, encoding="utf-8")
        return path

    def restore(self, snapshot_dir: Path) -> str:
        snapshot_dir = Path(snapshot_dir)
        for dest in self._learned_files():
            saved = snapshot_dir / dest.name
            if saved.exists():
                self._write(dest, saved.read_text(encoding="utf-8"))
            elif dest.exists():
                dest.unlink()
        return (snapshot_dir / _MODELFILE_NAME).read_text(encoding="utf-8")

    def latest_snapshot(self, date: str) -> Path | None:
        """The snapshot on date with the highest suffix, gaps allowed."""
        same_day = self._snapshots_on(date)
        return max(same_day)[1] if same_day else None

    def prune(self, today: str, keep_days: int) -> list[str]:
        """Delete snapshots and reports dated more than keep_days before today.

        The newest snapshot stays however old it is, so rollback still works,
        and today's report always stays. Returns the removed paths relative to
        the tuning folder.
        """
        cutoff = _date.fromisoformat(today) - timedelta(days=keep_days)
        removed: list[str] = []
        snaps = self._snapshots()
        if snaps:
            newest = snaps[-1][1]
            for (day, _), d in snaps:
                if day < cutoff and d != newest:
                    shutil.rmtree(d, ignore_errors=True)
                    removed.append(f"snapshots/{d.name}")
        reports = self.root / "reports"
        if reports.is_dir():
            for f in sorted(reports.glob("*.md")):
                dated = _dated(f.stem)
                if dated is not None and dated[0] < cutoff and f.stem != today:
                    f.unlink(missing_ok=True)
                    removed.append(f"reports/{f.name}")
        return removed

    def report_path(self, date: str) -> Path:
        return self.root / "reports" / f"{date}.md"

    def _learned_files(self) -> list[Path]:
        return [self._ad_examples, self._ad_rules, self._dedup_cases]
