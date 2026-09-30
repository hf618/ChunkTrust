"""ChunkTrust: action-expert evidence, episode memory, and learned horizon priors."""
from .selector import AHS, AHSConfig
from .prior import HybridQHARuntimeSelector

__all__ = ["AHS", "AHSConfig", "HybridQHARuntimeSelector"]
__version__ = "0.1.0"
