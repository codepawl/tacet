"""Tacet: typed decisions (choice, score, yes/no) with a probability for every option,
answered by one encoder pass."""

from .model import TacetModel, load
from .questions import choice, noul, score
from .validation import RequestError

__version__ = "0.2.1"

__all__ = ["TacetModel", "RequestError", "load", "choice", "score", "noul", "__version__"]
