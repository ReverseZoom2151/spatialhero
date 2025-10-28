"""
Evaluate a model on a test dataset.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import CodeGenerator, RewardModel
from training import CADDataset
from utils import compute_metrics, print_metrics


def main():
    """Evaluate model."""
    parser = argparse.ArgumentParser(description='Evaluate SpatialHero model')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/seed_prompts.json',
        help='Path to test dataset JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (if fine-tuned)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )

    args = parser.parse_args()

    print("="*60)
    print("SpatialHero Model Evaluation")
    print("="*60)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    try:
        dataset = CADDataset.from_json(args.dataset)
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Limit samples if specified
    if args.num_samples:
        dataset.samples = dataset.samples[:args.num_samples]
        print(f"Evaluating on {len(dataset)} samples")

    # Initialize components
    print("\nInitializing model...")
    try:
        generator = CodeGenerator()
        reward_model = RewardModel()
        print("✓ Models initialized")
    except Exception as e:
        print(f"✗ Failed to initialize models: {e}")
        return

    # Evaluate
    print("\nEvaluating...")
    results = []

    for i, sample in enumerate(dataset.samples):
        print(f"\nSample {i+1}/{len(dataset.samples)}: {sample.prompt}")

        # Generate code
        gen_result = generator.generate(sample.prompt)

        if not gen_result.success:
            print(f"  ✗ Generation failed: {gen_result.error}")
            results.append({
                'code_valid': 0,
                'execution_valid': 0,
                'geometry_valid': 0,
                'total_reward': 0.0,
                'dimension_accuracy': 0.0,
                'visual_quality': 0.0
            })
            continue

        # Evaluate
        eval_result = reward_model.evaluate(
            code=gen_result.code,
            prompt=sample.prompt,
            expected_dimensions=sample.expected_dimensions
        )

        # Store results
        result = {
            'code_valid': eval_result.components.code_valid,
            'execution_valid': eval_result.components.execution_valid,
            'geometry_valid': eval_result.components.geometry_valid,
            'total_reward': eval_result.total_reward,
            'dimension_accuracy': eval_result.components.dimension_accuracy,
            'visual_quality': eval_result.components.visual_quality
        }
        results.append(result)

        print(f"  Reward: {eval_result.total_reward:.3f}")
        if eval_result.validation.errors:
            print(f"  Errors: {'; '.join(eval_result.validation.errors[:2])}")

    # Compute and print metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
