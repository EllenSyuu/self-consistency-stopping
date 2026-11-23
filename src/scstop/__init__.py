"""Early-stopping rules for LLM self-consistency, implemented once and compared on the same samples."""
from .rules import (
    AdaptiveConsistencyBeta,
    ConSol,
    Decision,
    EarlyStoppingSC,
    FixedN,
    MartingaleMajorityCertificate,
    majority,
)
from .simulate import pool_mode, replay, summarise

__all__ = [
    "AdaptiveConsistencyBeta", "ConSol", "Decision", "EarlyStoppingSC", "FixedN",
    "MartingaleMajorityCertificate", "majority", "pool_mode", "replay", "summarise",
]
__version__ = "0.2.0"
