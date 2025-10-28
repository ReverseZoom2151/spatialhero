"""
SpatialHero - Making instruction-tuned LLMs spatially aware.

An improved implementation with multi-modal reward signals for
fine-tuning code generation models on CAD tasks.
"""

__version__ = "0.1.0"
__author__ = "SpatialHero Team"

from core import (
    CodeGenerator,
    CADRenderer,
    CodeVerifier,
    RewardModel
)

from utils import (
    load_config,
    compute_metrics,
    print_metrics
)

__all__ = [
    'CodeGenerator',
    'CADRenderer',
    'CodeVerifier',
    'RewardModel',
    'load_config',
    'compute_metrics',
    'print_metrics',
]
