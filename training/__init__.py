"""Training modules for SpatialHero."""

# Dataset imports (no heavy dependencies)
from training.dataset import CADDataset, CADSample, split_dataset

# PPO trainer imports are truly optional
# These require TRL which has heavy dependencies (Keras, TensorFlow, etc.)
_TRAINING_AVAILABLE = False
_TRAINING_IMPORT_ERROR = ""
SpatialHeroPPOTrainer = None
TrainingConfig = None

try:
    from training.ppo_trainer import SpatialHeroPPOTrainer as _PPOTrainer
    from training.ppo_trainer import TrainingConfig as _TrainingConfig
    SpatialHeroPPOTrainer = _PPOTrainer
    TrainingConfig = _TrainingConfig
    _TRAINING_AVAILABLE = True
except (ImportError, RuntimeError, ValueError) as e:
    # Training not available - this is OK, most users won't need it
    _TRAINING_IMPORT_ERROR = str(e).split('\n')[0]  # First line only

    # Create placeholder classes that give helpful error messages
    class SpatialHeroPPOTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Training module not available. "
                "This requires TRL (Transformer Reinforcement Learning) library.\n\n"
                "If you need training, install dependencies:\n"
                "  pip install tf-keras\n\n"
                "Note: Training is optional - code generation and evaluation work without it!"
            )

    class TrainingConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Training module not available. See SpatialHeroPPOTrainer for details."
            )

__all__ = [
    'SpatialHeroPPOTrainer',
    'TrainingConfig',
    'CADDataset',
    'CADSample',
    'split_dataset',
    '_TRAINING_AVAILABLE',
]
