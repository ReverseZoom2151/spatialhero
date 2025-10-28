"""
Evaluation metrics and utilities.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    code_validity_rate: float
    execution_success_rate: float
    geometry_validity_rate: float
    average_reward: float
    average_dimension_accuracy: float
    average_visual_quality: float


def compute_metrics(results: List[Dict[str, Any]]) -> EvaluationMetrics:
    """
    Compute aggregate metrics from evaluation results.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        EvaluationMetrics with aggregated scores
    """
    if not results:
        return EvaluationMetrics(
            code_validity_rate=0.0,
            execution_success_rate=0.0,
            geometry_validity_rate=0.0,
            average_reward=0.0,
            average_dimension_accuracy=0.0,
            average_visual_quality=0.0
        )

    code_valid = [r.get('code_valid', 0) for r in results]
    execution_valid = [r.get('execution_valid', 0) for r in results]
    geometry_valid = [r.get('geometry_valid', 0) for r in results]
    rewards = [r.get('total_reward', 0) for r in results]
    dim_accuracy = [r.get('dimension_accuracy', 0) for r in results]
    visual_quality = [r.get('visual_quality', 0) for r in results]

    return EvaluationMetrics(
        code_validity_rate=np.mean(code_valid),
        execution_success_rate=np.mean(execution_valid),
        geometry_validity_rate=np.mean(geometry_valid),
        average_reward=np.mean(rewards),
        average_dimension_accuracy=np.mean(dim_accuracy),
        average_visual_quality=np.mean(visual_quality)
    )


def print_metrics(metrics: EvaluationMetrics):
    """
    Print metrics in a readable format.

    Args:
        metrics: EvaluationMetrics to print
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Code Validity Rate:        {metrics.code_validity_rate:.2%}")
    print(f"Execution Success Rate:    {metrics.execution_success_rate:.2%}")
    print(f"Geometry Validity Rate:    {metrics.geometry_validity_rate:.2%}")
    print(f"Average Reward:            {metrics.average_reward:.3f}")
    print(f"Average Dimension Accuracy: {metrics.average_dimension_accuracy:.3f}")
    print(f"Average Visual Quality:    {metrics.average_visual_quality:.3f}")
    print("="*50 + "\n")
