"""
Configuration loading and management utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass
class RenderingConfig:
    """Rendering configuration."""
    width: int
    height: int
    views: list
    background_color: tuple


@dataclass
class RewardConfig:
    """Reward model configuration."""
    weights: Dict[str, float]
    thresholds: Dict[str, float]


@dataclass
class TrainingConfig:
    """Training configuration."""
    ppo: Dict[str, Any]
    checkpoint: Dict[str, Any]
    logging: Dict[str, Any]


@dataclass
class Config:
    """Master configuration."""
    models: Dict[str, ModelConfig]
    rendering: RenderingConfig
    reward: RewardConfig
    training: TrainingConfig
    dataset: Dict[str, Any]
    output: Dict[str, str]


class ConfigLoader:
    """
    Load and manage configuration from YAML files and environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()

        # Set default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'config.yaml'
            )

        self.config_path = config_path
        self.config_dict = self._load_yaml(config_path)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary of configuration
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get model configuration.

        Args:
            model_name: Name of the model (e.g., 'code_generator')

        Returns:
            ModelConfig instance
        """
        model_dict = self.config_dict['models'][model_name]
        return ModelConfig(**model_dict)

    def get_rendering_config(self) -> RenderingConfig:
        """Get rendering configuration."""
        render_dict = self.config_dict['rendering']
        return RenderingConfig(
            width=render_dict['width'],
            height=render_dict['height'],
            views=render_dict['views'],
            background_color=tuple(render_dict['background_color'])
        )

    def get_reward_config(self) -> RewardConfig:
        """Get reward model configuration."""
        reward_dict = self.config_dict['reward']
        return RewardConfig(
            weights=reward_dict['weights'],
            thresholds=reward_dict['thresholds']
        )

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        train_dict = self.config_dict['training']
        return TrainingConfig(
            ppo=train_dict['ppo'],
            checkpoint=train_dict['checkpoint'],
            logging=train_dict['logging']
        )

    def get_config(self) -> Config:
        """
        Get complete configuration.

        Returns:
            Config instance with all settings
        """
        models = {
            name: ModelConfig(**config)
            for name, config in self.config_dict['models'].items()
        }

        rendering = self.get_rendering_config()
        reward = self.get_reward_config()
        training = self.get_training_config()
        dataset = self.config_dict['dataset']
        output = self.config_dict['output']

        return Config(
            models=models,
            rendering=rendering,
            reward=reward,
            training=training,
            dataset=dataset,
            output=output
        )

    def get_openai_api_key(self) -> str:
        """
        Get OpenAI API key from environment.

        Returns:
            API key string

        Raises:
            ValueError: If API key not found
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        return api_key

    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        output = self.config_dict['output']
        for dir_path in output.values():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Config instance
    """
    loader = ConfigLoader(config_path)
    return loader.get_config()
