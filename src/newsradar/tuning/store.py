"""Private files the tuner learns into: examples, rules, dedup cases, snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

DEDUP_CASES_IN_PROMPT = 12
_MODELFILE_NAME = "Modelfile"
_DEDUP_HEADING = "LEARNED EXAMPLES (past mistakes to avoid):"


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
        self._ad_examples = self.root / "learned_ad_examples.json"
        self._ad_rules = self.root / "learned_ad_rules.txt"
        self._dedup_cases = self.root / "learned_dedup_cases.json"

    def _read_json(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

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
        cases = self.dedup_cases()
        if not cases:
            return ""
        # sorted() is stable, so cases without "added" keep their stored order.
        newest = sorted(cases, key=lambda c: c.get("added", ""))[-DEDUP_CASES_IN_PROMPT:]
        parts = [_DEDUP_HEADING]
        for case in newest:
            titles = "\n".join(f"{i}. {t}" for i, t in enumerate(case["titles"]))
            groups = json.dumps(case["expected_groups"])
            parts.append(f"Articles:\n{titles}\nExpected duplicate groups: {groups}")
        return "\n\n".join(parts) + "\n"

    def snapshot(self, date: str, built_modelfile: str) -> Path:
        base = self.root / "snapshots"
        path = base / date
        n = 2
        while path.exists():
            path = base / f"{date}-{n}"
            n += 1
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
        base = self.root / "snapshots"
        found = [base / date] if (base / date).is_dir() else []
        n = 2
        while (base / f"{date}-{n}").is_dir():
            found.append(base / f"{date}-{n}")
            n += 1
        return found[-1] if found else None

    def report_path(self, date: str) -> Path:
        return self.root / "reports" / f"{date}.md"

    def _learned_files(self) -> list[Path]:
        return [self._ad_examples, self._ad_rules, self._dedup_cases]
