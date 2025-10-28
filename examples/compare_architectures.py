"""
Compare original vs improved architecture.

This script demonstrates the differences between the original SpatialHero
approach and our improved multi-modal architecture.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import CodeGenerator, RewardModel
from utils.banner import print_banner, print_header
import time


def simulate_original_architecture(prompt: str) -> dict:
    """
    Simulate the original SpatialHero architecture.

    Original: LLM generates code → Render → GPT-4V scores (0-1) → Done

    Returns:
        Dictionary with minimal evaluation
    """
    print("\n" + "─"*60)
    print("ORIGINAL ARCHITECTURE")
    print("─"*60)

    start_time = time.time()

    # Step 1: Generate code
    print("1. Generating code...")
    generator = CodeGenerator()
    result = generator.generate(prompt)

    if not result.success:
        return {
            'success': False,
            'reward': 0.0,
            'feedback': 'Generation failed',
            'time': time.time() - start_time
        }

    # Step 2: Render (simulated)
    print("2. Rendering views...")
    print("   (Would render isometric, front, top, side views)")

    # Step 3: GPT-4V evaluation (simulated with GPT-5-Pro)
    print("3. Evaluating with vision model...")
    print("   (In original: single 0-1 score from GPT-4V)")

    # Simplified evaluation - just visual
    from openai import OpenAI
    client = OpenAI()

    eval_prompt = f"""Rate this CAD model description on a scale of 0.0 to 1.0:
Prompt: {prompt}
Generated code: {result.code[:500]}...

Provide only a number between 0.0 and 1.0."""

    try:
        # GPT-5 models use max_completion_tokens and only support temperature=1.0
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",  # Using mini for cost
            messages=[{"role": "user", "content": eval_prompt}],
            max_completion_tokens=10
            # Note: No temperature parameter - GPT-5 only supports default (1.0)
        )
        score_text = response.choices[0].message.content
        score = float(score_text.strip())
    except:
        score = 0.5

    elapsed = time.time() - start_time

    print(f"\n   Reward: {score:.3f}")
    print(f"   Time: {elapsed:.2f}s")

    return {
        'success': result.success,
        'reward': score,
        'feedback': f'Visual quality score: {score}',
        'time': elapsed,
        'details': 'Single scalar reward, no validation'
    }


def improved_architecture(prompt: str, expected_dims: dict) -> dict:
    """
    Run the improved multi-modal architecture.

    Improved: LLM generates code → Multi-stage validation → Composite reward

    Returns:
        Dictionary with comprehensive evaluation
    """
    print("\n" + "─"*60)
    print("IMPROVED ARCHITECTURE")
    print("─"*60)

    start_time = time.time()

    # Step 1: Generate code
    print("1. Generating code...")
    generator = CodeGenerator()
    result = generator.generate(prompt)

    if not result.success:
        return {
            'success': False,
            'total_reward': 0.0,
            'components': {},
            'feedback': 'Generation failed',
            'time': time.time() - start_time
        }

    # Step 2-5: Multi-modal evaluation
    print("2. Code validation (syntax, execution)...")
    print("3. Geometric verification (topology, dimensions)...")
    print("4. Visual evaluation (selective GPT-5-Pro)...")
    print("5. Computing composite reward...")

    reward_model = RewardModel()
    evaluation = reward_model.evaluate(
        code=result.code,
        prompt=prompt,
        expected_dimensions=expected_dims
    )

    elapsed = time.time() - start_time

    print(f"\n   Total Reward: {evaluation.total_reward:.3f}")
    print(f"   Components:")
    print(f"     - Code Valid: {evaluation.components.code_valid:.3f}")
    print(f"     - Dimension Accuracy: {evaluation.components.dimension_accuracy:.3f}")
    print(f"     - Visual Quality: {evaluation.components.visual_quality:.3f}")
    print(f"     - Topology Valid: {evaluation.components.topology_valid:.3f}")
    print(f"   Time: {elapsed:.2f}s")

    return {
        'success': evaluation.success,
        'total_reward': evaluation.total_reward,
        'components': {
            'code_valid': evaluation.components.code_valid,
            'dimension_accuracy': evaluation.components.dimension_accuracy,
            'visual_quality': evaluation.components.visual_quality,
            'topology_valid': evaluation.components.topology_valid
        },
        'feedback': evaluation.feedback,
        'time': elapsed,
        'validation_errors': len(evaluation.validation.errors),
        'validation_warnings': len(evaluation.validation.warnings)
    }


def main():
    """Compare architectures."""
    print_banner(compact=False, show_version=True)
    print_header("Architecture Comparison")

    prompt = "Create a simple chair with four legs, a seat, and a backrest"
    expected_dims = {
        'width': 40.0,
        'height': 85.0,
        'depth': 40.0
    }

    print(f"\nTest prompt: '{prompt}'")
    print(f"Expected dimensions: {expected_dims}")

    # Run original architecture
    original_result = simulate_original_architecture(prompt)

    # Run improved architecture
    improved_result = improved_architecture(prompt, expected_dims)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print("\n┌─ Original Architecture")
    print("│  Reward signal: Single 0-1 score")
    print(f"│  Reward: {original_result['reward']:.3f}")
    print(f"│  Time: {original_result['time']:.2f}s")
    print("│  Validation: None")
    print("│  Feedback: Minimal")
    print("│  Cost: High (GPT-4V every sample)")
    print("│")
    print("│  Issues:")
    print("│    ✗ No code validation")
    print("│    ✗ No dimensional verification")
    print("│    ✗ Black box evaluation")
    print("│    ✗ Poor training signal")
    print("└─")

    print("\n┌─ Improved Architecture")
    print("│  Reward signal: Multi-dimensional composite")
    print(f"│  Total Reward: {improved_result['total_reward']:.3f}")
    print(f"│  Time: {improved_result['time']:.2f}s")
    print(f"│  Validation: {improved_result['validation_errors']} errors, "
          f"{improved_result['validation_warnings']} warnings")
    print("│  Feedback: Detailed and actionable")
    print("│  Cost: Lower (custom model + selective GPT-5)")
    print("│")
    print("│  Components:")
    for name, value in improved_result['components'].items():
        print(f"│    • {name}: {value:.3f}")
    print("│")
    print("│  Advantages:")
    print("│    ✓ Multi-modal validation")
    print("│    ✓ Programmatic verification")
    print("│    ✓ Rich feedback for training")
    print("│    ✓ Better training signal")
    print("│    ✓ Production-ready error handling")
    print("└─")

    print("\n" + "="*60)
    print("KEY IMPROVEMENTS")
    print("="*60)
    print("\n1. REWARD SIGNAL QUALITY")
    print("   Original: 1D scalar (visual only)")
    print("   Improved: 4D composite (code, dimensions, visual, topology)")
    print("\n2. VALIDATION")
    print("   Original: None")
    print("   Improved: Comprehensive (syntax, execution, geometry)")
    print("\n3. FEEDBACK")
    print("   Original: Vague score")
    print("   Improved: Actionable error messages")
    print("\n4. SCALABILITY")
    print("   Original: Expensive GPT-4V every time")
    print("   Improved: Custom model + selective GPT-5-Pro")
    print("\n5. PRODUCTION READINESS")
    print("   Original: Research prototype")
    print("   Improved: Robust error handling, extensible")


if __name__ == "__main__":
    main()
