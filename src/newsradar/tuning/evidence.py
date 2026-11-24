"""Per-day evidence of what the ad gate and the dedup check saw and decided."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass, fields
from datetime import date as _date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

TEXT_SENT_CHARS = 1000
TEXT_FULL_CHARS = 4000


@dataclass
class AdRecord:
    key: str
    date: str
    topic: str
    title: str
    link: str
    source: str
    source_type: str
    text_sent: str
    text_full: str
    verdict: str  # "ad" or "not_ad"
    reason: str
    model: str


@dataclass
class DedupRecord:
    key: str
    date: str
    topic: str
    titles: list[str]
    dropped: list[int]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EvidenceRecorder:
    def __init__(self, data_dir: Path, date: str):
        self.data_dir = Path(data_dir)
        self.date = date
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def _append(self, topic: str, obj: dict) -> None:
        path = self.data_dir / "evidence" / self.date / f"{topic}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("could not write evidence to %s: %s", path, exc)

    def record_ad(
        self, *, topic: str, item: dict, title: str, text: str, source_type: str,
        verdict: str, reason: str, model: str,
    ) -> None:
        """Never raises: a recorder bug must not stop the scan."""
        try:
            text_sent = text[:TEXT_SENT_CHARS]
            key = _sha1(title + "\n" + text_sent)
            with self._lock:
                if key in self._seen:
                    return
                self._seen.add(key)
                rec = AdRecord(
                    key=key, date=self.date, topic=topic, title=title,
                    link=item.get("link", ""), source=item.get("source", ""),
                    source_type=source_type, text_sent=text_sent,
                    text_full=text[:TEXT_FULL_CHARS], verdict=verdict,
                    reason=reason, model=model,
                )
                self._append(topic, {"kind": "ad", **asdict(rec)})
        except Exception as exc:
            log.warning("could not record ad evidence for %r: %s: %s", title[:60], type(exc).__name__, exc)

    def record_dedup(self, *, topic: str, titles: list[str], dropped: list[int]) -> None:
        """Never raises: a recorder bug must not stop the scan."""
        try:
            key = _sha1(topic + "\n" + "\n".join(titles))
            with self._lock:
                if key in self._seen:
                    return
                self._seen.add(key)
                rec = DedupRecord(key=key, date=self.date, topic=topic, titles=list(titles),
                                  dropped=list(dropped))
                self._append(topic, {"kind": "dedup", **asdict(rec)})
        except Exception as exc:
            log.warning("could not record dedup evidence for topic %s: %s: %s", topic, type(exc).__name__, exc)


def load_day(data_dir: Path, date: str) -> tuple[list[AdRecord], list[DedupRecord], int]:
    """Load a day's records. The last record per key wins. Returns bad lines skipped.

    Evidence written before the title-phrase check was removed can hold ad
    records with verdict "heuristic". They are skipped and not counted.
    """
    ads: dict[str, AdRecord] = {}
    dedups: dict[str, DedupRecord] = {}
    bad = 0
    day_dir = Path(data_dir) / "evidence" / date
    if not day_dir.is_dir():
        return [], [], 0
    ad_fields = {f.name for f in fields(AdRecord)}
    dedup_fields = {f.name for f in fields(DedupRecord)}
    for path in sorted(day_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                kind = obj.get("kind")
                if kind == "ad" and obj.get("verdict") == "heuristic":
                    continue
                if kind == "ad":
                    rec = AdRecord(**{k: v for k, v in obj.items() if k in ad_fields})
                    ads[rec.key] = rec
                elif kind == "dedup":
                    drec = DedupRecord(**{k: v for k, v in obj.items() if k in dedup_fields})
                    dedups[drec.key] = drec
                else:
                    bad += 1
            except (ValueError, TypeError, AttributeError):
                bad += 1
    return list(ads.values()), list(dedups.values()), bad


def prune_evidence(data_dir: Path, today: str, keep_days: int) -> list[str]:
    """Delete evidence day folders older than keep_days. Returns the dates removed."""
    root = Path(data_dir) / "evidence"
    if not root.is_dir():
        return []
    cutoff = _date.fromisoformat(today) - timedelta(days=keep_days)
    removed: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            day = _date.fromisoformat(d.name)
        except ValueError:
            continue
        if day < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed
