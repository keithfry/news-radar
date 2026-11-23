"""Tests for the [tuning] config table and the ignored data folder."""

from __future__ import annotations

import subprocess
from pathlib import Path

import newsradar
from newsradar.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, extra: str) -> Path:
    (tmp_path / "t.toml").write_text('[topic]\nname = "t"\nclassifier_prompt = "p"\n')
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[topics]\nfiles = ["t.toml"]\n\n{extra}')
    return cfg


def test_tuning_defaults_off_with_repo_data_dir(tmp_path):
    cfg = load_config(_write_config(tmp_path, ""), env_file=tmp_path / "none.env")
    assert cfg.tuning.enabled is False
    assert cfg.tuning.data_dir == Path(newsradar.__file__).resolve().parents[2] / "data"
    assert cfg.tuning.judge_model == "claude-opus-5-5" and cfg.tuning.judge_effort == "medium"


def test_tuning_table_overrides_and_resolves_relative_data_dir(tmp_path):
    cfg = load_config(
        _write_config(tmp_path, '[tuning]\nenabled = true\ndata_dir = "d"\nmax_items_per_run = 50\n'),
        env_file=tmp_path / "none.env",
    )
    assert cfg.tuning.enabled and cfg.tuning.data_dir == tmp_path / "d" and cfg.tuning.max_items_per_run == 50


def test_data_dir_is_git_ignored():
    out = subprocess.run(["git", "check-ignore", "data/x"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0
