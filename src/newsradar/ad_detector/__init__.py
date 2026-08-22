"""Ad-detector model tooling: training data compilation and installation."""

from .install import install_ad_detector_model
from .update_modelfile import update_modelfile

__all__ = [
    "install_ad_detector_model",
    "update_modelfile",
]
