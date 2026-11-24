"""Tests for finding the ollama command without relying on PATH. Nothing here runs ollama."""

from __future__ import annotations

import subprocess

import pytest

from newsradar import ollama_cli
from newsradar.ad_detector import install


def _exe(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_fallbacks_are_the_intel_then_apple_silicon_homebrew_paths():
    assert ollama_cli.FALLBACK_PATHS == ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama")


def test_path_lookup_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_cli.shutil, "which", lambda name: "/somewhere/ollama")
    monkeypatch.setattr(ollama_cli, "FALLBACK_PATHS", (_exe(tmp_path / "a"),))
    assert ollama_cli.ollama_bin() == "/somewhere/ollama"


def test_first_executable_fallback_when_path_misses(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_cli.shutil, "which", lambda name: None)
    plain = tmp_path / "plain"
    plain.write_text("")  # not executable, so skipped
    second = _exe(tmp_path / "second")
    monkeypatch.setattr(ollama_cli, "FALLBACK_PATHS", (str(tmp_path / "missing"), str(plain), second))
    assert ollama_cli.ollama_bin() == second


def test_no_ollama_anywhere_raises_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(ollama_cli, "FALLBACK_PATHS", (str(tmp_path / "nope"),))
    with pytest.raises(ollama_cli.OllamaNotFoundError) as info:
        ollama_cli.ollama_bin()
    assert "PATH" in str(info.value) and str(tmp_path / "nope") in str(info.value)
    assert isinstance(info.value, FileNotFoundError)


def test_install_runs_the_resolved_binary(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(install, "ollama_bin", lambda: "/opt/homebrew/bin/ollama")
    monkeypatch.setattr(install.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    install.install_ad_detector_model(tmp_path / "Modelfile", "ad-detector")
    assert calls == [["/opt/homebrew/bin/ollama", "create", "ad-detector", "-f", str(tmp_path / "Modelfile")]]


def test_install_without_ollama_raises_before_running_anything(monkeypatch, tmp_path):
    def missing():
        raise ollama_cli.OllamaNotFoundError("no ollama")

    def no_run(*a, **kw):
        raise AssertionError("subprocess.run called")

    monkeypatch.setattr(install, "ollama_bin", missing)
    monkeypatch.setattr(install.subprocess, "run", no_run)
    with pytest.raises(ollama_cli.OllamaNotFoundError):
        install.install_ad_detector_model(tmp_path / "Modelfile", "ad-detector")
