"""
Demo with full 3D rendering enabled.

This demonstrates the complete system with PyVista rendering.
Requires: pip install pyvista numpy-stl
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(parent_dir, '.env'))
except:
    pass

from core import CodeGenerator, RewardModel
from core.renderer_3d import PYVISTA_AVAILABLE, get_renderer
from utils import load_config


def main():
    """Run demo with full rendering."""
    print("="*60)
    print("SpatialHero Demo - WITH 3D Rendering")
    print("="*60)

    # Check rendering availability
    print(f"\nPyVista available: {PYVISTA_AVAILABLE}")

    if not PYVISTA_AVAILABLE:
        print("\n⚠ PyVista not installed!")
        print("Install it for real 3D rendering:")
        print("  pip install pyvista numpy-stl")
        print("\nContinuing with placeholder rendering...\n")

    # Initialize
    print("\nInitializing components...")
    generator = CodeGenerator()
    reward_model = RewardModel(use_visual_eval=PYVISTA_AVAILABLE)

    if PYVISTA_AVAILABLE:
        print("  ✓ Real 3D rendering enabled")
    else:
        print("  ℹ Using placeholder rendering")

    # Generate code
    prompt = "Create a simple chair with four legs, a seat, and a backrest. Use millimeters."
    print(f"\nGenerating CAD code...")

    result = generator.generate(prompt)

    if not result.success:
        print(f"  ✗ Generation failed: {result.error}")
        return

    print(f"  ✓ Generated {len(result.code)} characters")

    # Evaluate with visual component
    print("\nEvaluating with multi-modal reward model...")

    expected_dims = {
        'width': 420.0,
        'height': 800.0,
        'depth': 420.0
    }

    evaluation = reward_model.evaluate(
        code=result.code,
        prompt=prompt,
        expected_dimensions=expected_dims
    )

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print("="*60)
    print(f"Total Reward:          {evaluation.total_reward:.3f}")
    print(f"├─ Code Valid:         {evaluation.components.code_valid:.3f}")
    print(f"├─ Execution Valid:    {evaluation.components.execution_valid:.3f}")
    print(f"├─ Geometry Valid:     {evaluation.components.geometry_valid:.3f}")
    print(f"├─ Dimension Accuracy: {evaluation.components.dimension_accuracy:.3f}")
    print(f"├─ Visual Quality:     {evaluation.components.visual_quality:.3f}")
    print(f"└─ Topology Valid:     {evaluation.components.topology_valid:.3f}")

    if PYVISTA_AVAILABLE:
        print(f"\n✅ Visual quality from REAL 3D renders!")
    else:
        print(f"\n⚠ Visual quality from placeholder (install PyVista for real renders)")

    print(f"\nFeedback: {evaluation.feedback}")

    # Show measurements
    if evaluation.validation.measurements:
        measurements = evaluation.validation.measurements
        if measurements.get('success') and 'comparisons' in measurements:
            print(f"\nDimensional Measurements:")
            for dim, comp in measurements['comparisons'].items():
                status = "✓" if comp['within_tolerance'] else "✗"
                print(f"  {status} {dim}: {comp['actual']:.1f}mm "
                      f"(expected: {comp['expected']:.1f}mm, error: {comp['relative_error']:.1%})")

    print("\n" + "="*60)

    # Compare with/without rendering
    print("\nImpact of Real 3D Rendering:")
    print("="*60)

    if PYVISTA_AVAILABLE:
        print("With PyVista:")
        print(f"  ✅ Visual Quality: {evaluation.components.visual_quality:.3f} (from actual renders)")
        print(f"  ✅ Total Reward: {evaluation.total_reward:.3f}")
        print(f"  ✅ Better training signal!")
    else:
        print("Without PyVista (current):")
        print(f"  ⚠ Visual Quality: {evaluation.components.visual_quality:.3f} (placeholder)")
        print(f"  ⚠ Total Reward: {evaluation.total_reward:.3f}")
        print(f"\nWith PyVista (estimated):")
        estimated_visual = 0.85
        estimated_total = (
            0.20 * 1.0 +  # code_valid
            0.30 * evaluation.components.dimension_accuracy +
            0.30 * estimated_visual +  # improved visual
            0.20 * 1.0   # topology_valid
        )
        print(f"  ✅ Visual Quality: ~{estimated_visual:.3f} (from renders)")
        print(f"  ✅ Total Reward: ~{estimated_total:.3f} (+{estimated_total - evaluation.total_reward:.3f})")
        print(f"\n  To enable: pip install pyvista numpy-stl")

    print("="*60)


if __name__ == "__main__":
    main()
