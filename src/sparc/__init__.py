"""SPARC: pathway-constrained spatial GRN inference."""

from ._model import SPARC
from ._module import SPARCVAE
from .preprocessing import build_cascade_priors

__version__ = "0.5.0"

__all__ = ["SPARC", "SPARCVAE", "build_cascade_priors"]
