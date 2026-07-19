"""Shared test fixtures/helpers."""

from __future__ import annotations

import socket
from pathlib import Path

EXAMPLE_CONFIG = Path(__file__).parents[1] / "examples" / "config.example.toml"


def ollama_reachable(host: str = "localhost", port: int = 11434) -> bool:
    """Best-effort check that a local Ollama daemon is listening."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def ollama_has_model(model: str) -> bool:
    if not ollama_reachable():
        return False
    try:
        import ollama

        names = {m.model for m in ollama.list().models}
        # Exact match only — "llama3.2" (implicit :latest) and "llama3.2:3b"
        # are different pulls as far as `ollama.chat()` is concerned.
        return model in names or f"{model}:latest" in names
    except Exception:
        return False
