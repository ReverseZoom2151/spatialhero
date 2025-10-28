"""
Proximal Policy Optimization (PPO) trainer for fine-tuning code generation models.

This implements PPO to train an LLM to generate better CADQuery code
using the multi-modal reward signal.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import numpy as np

from core.reward_model import RewardModel
from training.dataset import CADDataset


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""
    # Model
    model_name: str = "ise-uiuc/Magicoder-S-DS-6.7B"

    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training
    num_train_epochs: int = 3
    max_steps: int = 10000
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Paths
    output_dir: str = "outputs/checkpoints"
    logging_dir: str = "outputs/logs"

    # Hardware
    use_fp16: bool = True
    gradient_accumulation_steps: int = 1


class SpatialHeroPPOTrainer:
    """
    PPO trainer for SpatialHero code generation model.
    """

    def __init__(
        self,
        config: TrainingConfig,
        reward_model: RewardModel,
        train_dataset: CADDataset,
        eval_dataset: Optional[CADDataset] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            config: Training configuration
            reward_model: Multi-modal reward model for evaluation
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.config = config
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        )
        self.model.to(self.device)

        # Reference model (frozen copy for KL divergence)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()

        # PPO config
        ppo_config = PPOConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            ppo_epochs=config.ppo_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            cliprange=config.clip_range,
            vf_coef=config.value_loss_coef,
            ent_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = -float('inf')

        # Metrics tracking
        self.metrics_history = {
            'rewards': [],
            'code_valid': [],
            'dimension_accuracy': [],
            'visual_quality': [],
            'losses': []
        }

    def train(self):
        """
        Main training loop.
        """
        print("Starting training...")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            print(f"{'='*50}")

            self._train_epoch(dataloader)

            # Evaluate
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                print(f"\nEvaluation metrics: {eval_metrics}")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        print("\nTraining completed!")

    def _train_epoch(self, dataloader: DataLoader):
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
        """
        self.model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Generate responses
            prompts = batch['prompt']
            query_tensors = [
                self.tokenizer.encode(p, return_tensors="pt")[0].to(self.device)
                for p in prompts
            ]

            # Generate code
            response_tensors = []
            for query in query_tensors:
                response = self.ppo_trainer.generate(
                    query.unsqueeze(0),
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )
                response_tensors.append(response.squeeze())

            # Decode generated code
            generated_codes = [
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]

            # Compute rewards
            rewards = self._compute_rewards(
                prompts,
                generated_codes,
                batch.get('expected_dimensions')
            )

            # Convert rewards to tensors
            reward_tensors = [torch.tensor(r).to(self.device) for r in rewards]

            # PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Log metrics
            self._log_metrics(stats, rewards)

            # Update progress bar
            avg_reward = np.mean(rewards)
            progress_bar.set_postfix({
                'reward': f'{avg_reward:.3f}',
                'loss': f'{stats.get("ppo/loss/total", 0):.3f}'
            })

            self.global_step += 1

            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")

            # Evaluate
            if self.global_step % self.config.eval_steps == 0 and self.eval_dataset:
                eval_metrics = self.evaluate()
                print(f"\nStep {self.global_step} eval: {eval_metrics}")
                self.model.train()

    def _compute_rewards(
        self,
        prompts: List[str],
        codes: List[str],
        expected_dimensions_list: Optional[List[Dict[str, float]]] = None
    ) -> List[float]:
        """
        Compute rewards for generated code using multi-modal reward model.

        Args:
            prompts: Original prompts
            codes: Generated code
            expected_dimensions_list: Expected dimensions

        Returns:
            List of reward scores
        """
        if expected_dimensions_list is None:
            expected_dimensions_list = [None] * len(codes)

        rewards = []
        for prompt, code, expected_dims in zip(prompts, codes, expected_dimensions_list):
            try:
                # Evaluate with multi-modal reward model
                result = self.reward_model.evaluate(
                    code=code,
                    prompt=prompt,
                    expected_dimensions=expected_dims
                )

                reward = result.total_reward

                # Track component metrics
                self.metrics_history['code_valid'].append(result.components.code_valid)
                self.metrics_history['dimension_accuracy'].append(result.components.dimension_accuracy)
                self.metrics_history['visual_quality'].append(result.components.visual_quality)

            except Exception as e:
                print(f"Error computing reward: {e}")
                reward = 0.0

            rewards.append(reward)

        return rewards

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.eval_dataset:
            return {}

        self.model.eval()

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        all_rewards = []
        all_code_valid = []
        all_dim_accuracy = []
        all_visual_quality = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prompts = batch['prompt']

                # Generate code
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                    )

                    code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Evaluate
                    result = self.reward_model.evaluate(code=code, prompt=prompt)

                    all_rewards.append(result.total_reward)
                    all_code_valid.append(result.components.code_valid)
                    all_dim_accuracy.append(result.components.dimension_accuracy)
                    all_visual_quality.append(result.components.visual_quality)

        metrics = {
            'eval/reward': np.mean(all_rewards),
            'eval/code_valid': np.mean(all_code_valid),
            'eval/dimension_accuracy': np.mean(all_dim_accuracy),
            'eval/visual_quality': np.mean(all_visual_quality),
        }

        # Update best model
        if metrics['eval/reward'] > self.best_reward:
            self.best_reward = metrics['eval/reward']
            self.save_checkpoint("best_model")

        return metrics

    def _log_metrics(self, stats: Dict[str, Any], rewards: List[float]):
        """
        Log training metrics.

        Args:
            stats: Training statistics from PPO
            rewards: Batch rewards
        """
        avg_reward = np.mean(rewards)
        self.metrics_history['rewards'].append(avg_reward)

        if 'ppo/loss/total' in stats:
            self.metrics_history['losses'].append(stats['ppo/loss/total'])

        # Log every N steps
        if self.global_step % self.config.logging_steps == 0:
            print(f"\nStep {self.global_step}:")
            print(f"  Reward: {avg_reward:.3f}")
            print(f"  Code Valid: {np.mean(self.metrics_history['code_valid'][-10:]):.3f}")
            print(f"  Dim Accuracy: {np.mean(self.metrics_history['dimension_accuracy'][-10:]):.3f}")
            print(f"  Visual Quality: {np.mean(self.metrics_history['visual_quality'][-10:]):.3f}")

    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'config': asdict(self.config),
            'metrics_history': self.metrics_history
        }

        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load model
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint_path)
        self.model.to(self.device)

        # Load training state
        state_path = os.path.join(checkpoint_path, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)

            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_reward = state['best_reward']
            self.metrics_history = state['metrics_history']

        print(f"Checkpoint loaded from {checkpoint_path}")
