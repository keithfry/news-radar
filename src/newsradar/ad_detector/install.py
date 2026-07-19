"""Install the ad-detector Ollama model from its packaged Modelfile."""

import subprocess
from pathlib import Path

_DEFAULT_MODELFILE_PATH = Path(__file__).parent / "AdDetectorModelfile"


def install_ad_detector_model(modelfile_path: Path | None = None, model_name: str = "ad-detector") -> None:
    modelfile_path = modelfile_path or _DEFAULT_MODELFILE_PATH
    subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)
