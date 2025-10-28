"""
Train SpatialHero model with PPO.

This script trains a code generation model to produce better CAD code
using our improved multi-modal reward signal.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import RewardModel
from training import SpatialHeroPPOTrainer, TrainingConfig, CADDataset, split_dataset
from utils import load_config


def main():
    """Train model."""
    parser = argparse.ArgumentParser(description='Train SpatialHero model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/seed_prompts.json',
        help='Path to training dataset JSON'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    args = parser.parse_args()

    print("="*60)
    print("SpatialHero Model Training")
    print("="*60)

    # Load configuration
    print("\nLoading configuration...")
    try:
        config = load_config(args.config)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return

    # Create output directories
    os.makedirs(config.output['checkpoints_dir'], exist_ok=True)
    os.makedirs(config.output['logs_dir'], exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    try:
        full_dataset = CADDataset.from_json(args.dataset, augment=True)
        print(f"✓ Loaded {len(full_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Split dataset
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=config.dataset['train_split'],
        val_ratio=config.dataset['val_split'],
        test_ratio=config.dataset['test_split']
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Initialize reward model
    print("\nInitializing reward model...")
    try:
        reward_model = RewardModel(
            weights=config.reward.weights,
            use_visual_eval=True
        )
        print("✓ Reward model initialized")
    except Exception as e:
        print(f"✗ Failed to initialize reward model: {e}")
        return

    # Initialize training config
    train_config = TrainingConfig(
        learning_rate=config.training.ppo['learning_rate'],
        batch_size=config.training.ppo['batch_size'],
        mini_batch_size=config.training.ppo['mini_batch_size'],
        ppo_epochs=config.training.ppo['epochs'],
        gamma=config.training.ppo['gamma'],
        gae_lambda=config.training.ppo['gae_lambda'],
        clip_range=config.training.ppo['clip_range'],
        value_loss_coef=config.training.ppo['value_loss_coef'],
        entropy_coef=config.training.ppo['entropy_coef'],
        max_grad_norm=config.training.ppo['max_grad_norm'],
        save_steps=config.training.checkpoint['save_every'],
        eval_steps=config.training.logging['eval_every'],
        logging_steps=config.training.logging['log_every'],
        output_dir=config.output['checkpoints_dir'],
        logging_dir=config.output['logs_dir']
    )

    if args.epochs:
        train_config.num_train_epochs = args.epochs

    # Initialize trainer
    print("\nInitializing trainer...")
    try:
        trainer = SpatialHeroPPOTrainer(
            config=train_config,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        print("✓ Trainer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        try:
            trainer.load_checkpoint(args.checkpoint)
            print("✓ Checkpoint loaded")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nCheckpoints saved to: {train_config.output_dir}")
    print(f"Logs saved to: {train_config.logging_dir}")


if __name__ == "__main__":
    main()
