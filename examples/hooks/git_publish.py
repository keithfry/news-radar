"""Reference publish hook: git add/commit/push generated output files.

This lives in examples/, not src/newsradar/, because publishing is explicitly
NOT core package behavior — the pipeline just writes files and hands the list
of written paths to whatever publish hook you configure (or none at all).

Usage (in your config.toml):

    [publish]
    hook = "hooks.git_publish:publish_hook"

...with `hooks/` (this directory, or a copy of it) importable on PYTHONPATH —
e.g. run newsradar from a working directory that has `hooks/git_publish.py`
next to it, or install it as part of your own project package.

Git identity is read from environment variables so nothing personal is
hardcoded here:

    NEWSRADAR_GIT_USER_NAME   (falls back to GIT_AUTHOR_NAME, then "news-radar bot")
    NEWSRADAR_GIT_USER_EMAIL  (falls back to GIT_AUTHOR_EMAIL, then "news-radar@example.com")
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from newsradar.config import Config

_DEFAULT_NAME = "news-radar bot"
_DEFAULT_EMAIL = "news-radar@example.com"


def _git_identity() -> tuple[str, str]:
    name = os.environ.get("NEWSRADAR_GIT_USER_NAME") or os.environ.get("GIT_AUTHOR_NAME") or _DEFAULT_NAME
    email = os.environ.get("NEWSRADAR_GIT_USER_EMAIL") or os.environ.get("GIT_AUTHOR_EMAIL") or _DEFAULT_EMAIL
    return name, email


def _run(repo_root: Path, args: list[str], check: bool = True, log=print) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, cwd=repo_root)
    if result.stdout.strip():
        log(f"  [git] {result.stdout.strip()}")
    if result.stderr.strip():
        log(f"  [git] {result.stderr.strip()}")
    if check and result.returncode != 0:
        raise RuntimeError(f"git command failed: {' '.join(args)}\n{result.stderr}")
    return result


def _commit_message(paths: list[Path]) -> str:
    """Derive a commit message from the topic output directories present in paths."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    topic_dirs: list[str] = []
    for p in paths:
        # output paths look like <output_root>/<topic.output_dir>/<YYYY-MM>/<file>
        parts = p.parts
        if len(parts) >= 3:
            topic_dir = parts[-3]
            if topic_dir not in topic_dirs:
                topic_dirs.append(topic_dir)
    if topic_dirs:
        label = " + ".join(t.capitalize() for t in topic_dirs)
        return f"Add {label} digest update {date_str}"
    return f"Add digest update {date_str}"


def publish_hook(paths: list[Path], config: Config, log=print) -> None:
    """git pull --rebase --autostash, add all paths, commit, push.

    Matches the `PublishHook = Callable[[list[Path], Config], None]` type
    alias in newsradar.output_writer.
    """
    if not paths:
        log("  no paths to publish, skipping")
        return

    repo_root = config.repo_root
    name, email = _git_identity()
    commit_msg = _commit_message(paths)

    lock = repo_root / ".git" / "index.lock"
    if lock.exists():
        lock.unlink()
        log("  removed stale .git/index.lock")

    _run(repo_root, ["git", "-C", str(repo_root), "pull", "--rebase", "--autostash"], log=log)

    for path in paths:
        rel = path.relative_to(repo_root)
        _run(repo_root, ["git", "-C", str(repo_root), "add", str(rel)], log=log)

    result = _run(
        repo_root,
        [
            "git", "-C", str(repo_root),
            "-c", f"user.name={name}",
            "-c", f"user.email={email}",
            "commit", "-m", commit_msg,
        ],
        check=False,
        log=log,
    )
    if result.returncode != 0:
        if "nothing to commit" in result.stdout + result.stderr:
            log("  nothing to commit, skipping push")
            return
        raise RuntimeError(f"git commit failed:\n{result.stderr}")

    _run(repo_root, ["git", "-C", str(repo_root), "push"], log=log)
    log(f"  pushed: {commit_msg}")
