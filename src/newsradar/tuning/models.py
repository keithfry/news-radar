"""The Ollama side of the tuner: model names, building and installing a
Modelfile, and the `ollama show` and `ollama rm` calls."""

from __future__ import annotations

import subprocess
from pathlib import Path

from newsradar.ad_detector.install import install_ad_detector_model
from newsradar.ad_detector.update_modelfile import (
    CONTEXT_SHARE,
    DEFAULT_DATA_PATH,
    DEFAULT_MODELFILE_PATH,
    build_live_modelfile,
    check_context_budget,
    load_examples,
)
from newsradar.config import Config
from newsradar.ollama_cli import ollama_bin
from newsradar.tuning.store import TuningStore

BASE_MODELFILE = DEFAULT_MODELFILE_PATH
BASE_EXAMPLES = DEFAULT_DATA_PATH


class SnapshotError(RuntimeError):
    """`ollama show --modelfile` failed, so the live model can't be snapshotted."""



def live_model_name(config: Config) -> str:
    return config.models.ad_detector_model.removeprefix("ollama/")


def candidate_model_name(config: Config) -> str:
    """`<live>-candidate`, with any `:tag` on the live name dropped first."""
    name = live_model_name(config)
    base, sep, tag = name.rpartition(":")
    if sep and "/" not in tag:
        name = base
    return f"{name}-candidate"


def installed_modelfile(name: str) -> str:
    """The Modelfile of the installed model, from `ollama show --modelfile`.
    Raises SnapshotError when the call fails or prints nothing."""
    try:
        proc = subprocess.run([ollama_bin(), "show", "--modelfile", name], capture_output=True, text=True,
                              check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise SnapshotError(f"ollama show --modelfile {name} exited {exc.returncode}: "
                            f"{(exc.stderr or '').strip()[:300]}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise SnapshotError(f"ollama show --modelfile {name} failed: {exc}") from exc
    if not proc.stdout.strip():
        raise SnapshotError(f"ollama show --modelfile {name} printed nothing")
    return proc.stdout


def remove_ollama_model(name: str) -> None:
    """Run `ollama rm`, ignoring every failure (the model may not exist)."""
    try:
        subprocess.run([ollama_bin(), "rm", name], capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def _build_text(config: Config, examples: list[dict], rules: str) -> str:
    """Build a Modelfile from base plus the given learned data. Raises
    ModelfileTooLargeError, before anything is installed, when its SYSTEM
    block would fill more than CONTEXT_SHARE of num_ctx."""
    text = build_live_modelfile(
        BASE_MODELFILE.read_text(encoding="utf-8"),
        load_examples(BASE_EXAMPLES),
        examples,
        rules,
        config.tuning.learned_examples_per_class,
    )
    check_context_budget(text, CONTEXT_SHARE)
    return text


def _build_dir(config: Config) -> Path:
    return TuningStore(config.tuning.data_dir).root / "build"


def _install(config: Config, text: str, model_name: str) -> None:
    """Write text to a temporary Modelfile under data_dir, run `ollama create`, delete the file."""
    path = _build_dir(config) / f"{model_name}.Modelfile"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        install_ad_detector_model(path, model_name)
    finally:
        path.unlink(missing_ok=True)
