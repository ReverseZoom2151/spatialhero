"""
Unit tests for SpatialHero components.

These tests do NOT require an API key and should run quickly.
"""

import pytest
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from core.verifier import CodeVerifier
from utils.cad_utils import CADQueryExecutor
from utils.config_loader import ConfigLoader

# Import training components (gracefully handle if training module unavailable)
try:
    from training.dataset import CADDataset, CADSample, split_dataset
    from training import _TRAINING_AVAILABLE
except (ImportError, RuntimeError, ValueError):
    # Training not available - define minimal versions for testing
    _TRAINING_AVAILABLE = False
    CADDataset = None
    CADSample = None
    split_dataset = None

from tests.fixtures.sample_codes import (
    VALID_SIMPLE_BOX,
    VALID_CHAIR,
    INVALID_SYNTAX,
    MISSING_RESULT,
    RUNTIME_ERROR,
    DANGEROUS_CODE
)


class TestCodeVerifier:
    """Test code verification without execution."""

    def test_valid_syntax(self):
        """Test that valid code passes syntax check."""
        verifier = CodeVerifier()
        result = verifier._validate_syntax(VALID_SIMPLE_BOX)
        assert result[0] == True
        assert len(result[1]) == 0

    def test_invalid_syntax(self):
        """Test that invalid syntax is caught."""
        verifier = CodeVerifier()
        result = verifier._validate_syntax(INVALID_SYNTAX)
        assert result[0] == False
        assert len(result[1]) > 0
        assert "Syntax error" in result[1][0] or "Parse error" in result[1][0]

    def test_static_analysis_missing_result(self):
        """Test that missing result variable triggers warning."""
        verifier = CodeVerifier()
        warnings = verifier._static_analysis(MISSING_RESULT)
        assert any("result" in w.lower() for w in warnings)

    def test_static_analysis_dangerous_ops(self):
        """Test that dangerous operations are flagged."""
        verifier = CodeVerifier()
        warnings = verifier._static_analysis(DANGEROUS_CODE)
        # Should warn about os.system or similar
        assert len(warnings) > 0


class TestCADQueryExecutor:
    """Test CADQuery code execution."""

    def test_execute_valid_code(self):
        """Test executing valid CADQuery code."""
        executor = CADQueryExecutor()
        result = executor.execute(VALID_SIMPLE_BOX)
        assert result is not None

    def test_execute_invalid_code(self):
        """Test that invalid code returns None."""
        executor = CADQueryExecutor()
        result = executor.execute(INVALID_SYNTAX)
        assert result is None

    def test_execute_runtime_error(self):
        """Test that runtime errors are caught."""
        executor = CADQueryExecutor()
        result = executor.execute(RUNTIME_ERROR)
        assert result is None


class TestConfigLoader:
    """Test configuration loading."""

    def test_load_config(self):
        """Test loading configuration file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'config.yaml'
        )
        loader = ConfigLoader(config_path)
        assert loader.config_dict is not None
        assert 'models' in loader.config_dict
        assert 'reward' in loader.config_dict

    def test_get_model_config(self):
        """Test getting model configuration."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'config.yaml'
        )
        loader = ConfigLoader(config_path)
        model_config = loader.get_model_config('code_generator')
        assert model_config.provider == 'openai'
        assert model_config.model is not None

    def test_get_reward_config(self):
        """Test getting reward configuration."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'config.yaml'
        )
        loader = ConfigLoader(config_path)
        reward_config = loader.get_reward_config()
        assert reward_config.weights is not None
        assert 'code_valid' in reward_config.weights
        # Weights should sum to 1.0
        assert abs(sum(reward_config.weights.values()) - 1.0) < 0.01


@pytest.mark.skipif(
    not _TRAINING_AVAILABLE if '_TRAINING_AVAILABLE' in dir() else True,
    reason="Training module not available (optional dependency)"
)
class TestDataset:
    """Test dataset functionality."""

    def test_create_sample(self):
        """Test creating a CAD sample."""
        if CADSample is None:
            pytest.skip("Training module not available")
        sample = CADSample(
            prompt="Create a box",
            expected_dimensions={'width': 10, 'height': 10, 'depth': 10},
            category="basic"
        )
        assert sample.prompt == "Create a box"
        assert sample.expected_dimensions['width'] == 10

    def test_dataset_creation(self):
        """Test creating a dataset."""
        samples = [
            CADSample(prompt="Create a box", category="basic"),
            CADSample(prompt="Create a cylinder", category="basic"),
        ]
        dataset = CADDataset(samples)
        assert len(dataset) == 2
        assert dataset[0]['prompt'] == "Create a box"

    def test_dataset_split(self):
        """Test splitting dataset."""
        samples = [CADSample(prompt=f"Prompt {i}") for i in range(10)]
        dataset = CADDataset(samples)

        train, val, test = split_dataset(dataset, 0.6, 0.2, 0.2, seed=42)

        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_dataset_augmentation(self):
        """Test dataset augmentation."""
        samples = [CADSample(prompt="Create a box")]
        dataset = CADDataset(samples, augment=True)

        # Get same item multiple times with augmentation
        prompts = [dataset[0]['prompt'] for _ in range(5)]

        # With augmentation, some prompts might be different
        # (though not guaranteed due to randomness)
        assert all(isinstance(p, str) for p in prompts)


class TestIntegration:
    """Test component integration without API."""

    def test_verifier_full_pipeline(self):
        """Test full verification pipeline."""
        verifier = CodeVerifier()

        # Valid code should pass all checks
        result = verifier.verify(VALID_SIMPLE_BOX)
        assert result.code_valid == True
        assert result.execution_valid == True

        # Invalid syntax should fail early
        result = verifier.verify(INVALID_SYNTAX)
        assert result.code_valid == False
        assert result.execution_valid == False

    def test_config_to_components(self):
        """Test using config with components."""
        from utils import load_config

        config = load_config()

        # Check that config can be used
        assert config.reward.weights is not None
        assert config.rendering.width > 0
        assert config.training.ppo is not None


def test_imports():
    """Test that all modules can be imported."""
    try:
        from core import CodeGenerator, CADRenderer, CodeVerifier, RewardModel
        from training import CADDataset, _TRAINING_AVAILABLE
        from utils import load_config, compute_metrics

        # Training module is optional
        if not _TRAINING_AVAILABLE:
            print("\nℹ Training module not available (optional)")

        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_project_structure():
    """Test that required files exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))

    required_files = [
        'README.md',
        'requirements.txt',
        'config/config.yaml',
        '.env.example',
        '.gitignore'
    ]

    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        assert os.path.exists(full_path), f"Missing required file: {file_path}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
