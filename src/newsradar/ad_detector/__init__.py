"""Ad-detector model tooling: training data compilation, model comparison, and installation.

Note: `compare_models` is intentionally not imported here — it depends on
`newsradar.article_fetcher`, a heavier optional dependency chain (RSS + full-text
fetching). Import it directly from `newsradar.ad_detector.compare_models` when needed,
so `install_ad_detector_model` and `update_modelfile` stay importable without it.
"""

from .install import install_ad_detector_model
from .update_modelfile import update_modelfile

__all__ = [
    "install_ad_detector_model",
    "update_modelfile",
]
