"""Find the ollama command without relying on PATH.

launchd and cron start jobs with a short PATH that often lacks the Homebrew
folders, so a bare "ollama" fails there even when ollama is installed.
"""

from __future__ import annotations

import os
import shutil

# Homebrew's install paths on Intel and on Apple silicon, tried in this order after PATH.
FALLBACK_PATHS = ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama")


class OllamaNotFoundError(FileNotFoundError):
    """No ollama command on PATH or at any of FALLBACK_PATHS."""


def ollama_bin() -> str:
    """The path of the ollama command: PATH first, then each of FALLBACK_PATHS.
    Raises OllamaNotFoundError when none of them is an executable file."""
    found = shutil.which("ollama")
    if found:
        return found
    for path in FALLBACK_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise OllamaNotFoundError(
        f"can't find the ollama command: it is not on PATH and not at {' or '.join(FALLBACK_PATHS)}")
