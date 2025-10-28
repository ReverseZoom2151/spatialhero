"""
Simple demo of the SpatialHero pipeline.

This demonstrates the improved architecture with multi-modal evaluation.
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Load .env file from parent directory
try:
    from dotenv import load_dotenv
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # Manual loading as fallback
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from core import CodeGenerator, CADRenderer, RewardModel
from utils import load_config


def main():
    """Run a simple demo."""
    print("="*60)
    print("SpatialHero Demo - Improved Architecture")
    print("="*60)

    # Load configuration
    print("\n1. Loading configuration...")
    try:
        config = load_config()
        print("   ✓ Configuration loaded")
    except Exception as e:
        print(f"   ✗ Failed to load config: {e}")
        return

    # Initialize components
    print("\n2. Initializing components...")

    try:
        generator = CodeGenerator()
        print("   ✓ Code generator initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize generator: {e}")
        print("   Make sure OPENAI_API_KEY is set in your .env file")
        return

    renderer = CADRenderer()
    print("   ✓ Renderer initialized")

    try:
        # Check if PyVista is available for real 3D rendering
        from core.renderer_3d import PYVISTA_AVAILABLE

        # Only enable visual eval if we have real 3D rendering
        use_visual = PYVISTA_AVAILABLE

        reward_model = RewardModel(use_visual_eval=use_visual)

        if use_visual:
            print("   ✓ Reward model initialized (visual eval: enabled with PyVista)")
        else:
            print("   ✓ Reward model initialized (visual eval: disabled)")
            print("      For visual evaluation, install: pip install pyvista numpy-stl")
    except Exception as e:
        print(f"   ✗ Failed to initialize reward model: {e}")
        print("   Make sure OPENAI_API_KEY is set in your .env file")
        return

    # Generate code
    prompt = "Create a simple chair with four legs, a seat, and a backrest. Use millimeters for all dimensions."
    print(f"\n3. Generating CAD code for: '{prompt}'")

    result = generator.generate(prompt)

    if not result.success:
        print(f"   ✗ Generation failed: {result.error}")
        return

    print("   ✓ Code generated successfully")
    print(f"\n   Generated code:")
    print("   " + "-"*56)
    for line in result.code.split('\n')[:15]:  # Show first 15 lines
        print(f"   {line}")
    if len(result.code.split('\n')) > 15:
        print("   ...")
    print("   " + "-"*56)

    # Evaluate with multi-modal reward model
    print("\n4. Evaluating with multi-modal reward model...")

    # Note: Expected dimensions in centimeters (cm)
    # The generated code used millimeters, so we need to match units
    expected_dims = {
        'width': 420.0,   # ~42cm in mm
        'height': 1000.0,  # ~100cm in mm
        'depth': 420.0     # ~42cm in mm
    }

    evaluation = reward_model.evaluate(
        code=result.code,
        prompt=prompt,
        expected_dimensions=expected_dims
    )

    print(f"\n   Evaluation Results:")
    print(f"   {'─'*56}")
    print(f"   Total Reward:          {evaluation.total_reward:.3f}")
    print(f"   Code Valid:            {evaluation.components.code_valid:.3f}")
    print(f"   Execution Valid:       {evaluation.components.execution_valid:.3f}")
    print(f"   Geometry Valid:        {evaluation.components.geometry_valid:.3f}")
    print(f"   Dimension Accuracy:    {evaluation.components.dimension_accuracy:.3f}")
    print(f"   Visual Quality:        {evaluation.components.visual_quality:.3f}")
    print(f"   Topology Valid:        {evaluation.components.topology_valid:.3f}")
    print(f"   {'─'*56}")

    print(f"\n   Feedback: {evaluation.feedback}")

    # Show validation details
    if evaluation.validation.errors:
        print(f"\n   Errors:")
        for error in evaluation.validation.errors:
            print(f"     - {error}")

    if evaluation.validation.warnings:
        print(f"\n   Warnings:")
        for warning in evaluation.validation.warnings:
            print(f"     - {warning}")

    # Show measurements if available
    if evaluation.validation.measurements:
        measurements = evaluation.validation.measurements
        if measurements.get('success') and 'comparisons' in measurements:
            print(f"\n   Dimensional Measurements:")
            for dim, comp in measurements['comparisons'].items():
                status = "[PASS]" if comp['within_tolerance'] else "[FAIL]"
                print(f"     {status} {dim}: {comp['actual']:.1f} "
                      f"(expected: {comp['expected']:.1f}, "
                      f"error: {comp['relative_error']:.1%})")

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)

    # Compare with original architecture
    print("\n" + "="*60)
    print("Comparison with Original Architecture")
    print("="*60)
    print("\nOriginal approach would only provide:")
    print("  - Single visual quality score from GPT-4V")
    print("  - No code validation")
    print("  - No dimensional verification")
    print("  - No actionable feedback")
    print("\nOur improved approach provides:")
    print("  - Multi-dimensional reward signal")
    print("  - Code syntax and execution validation")
    print("  - Precise dimensional measurements")
    print("  - Geometric topology checks")
    print("  - Detailed, actionable feedback")
    print("  - Cost-effective evaluation (selective LLM usage)")

    # Show rendering status
    from core.renderer_3d import PYVISTA_AVAILABLE
    if PYVISTA_AVAILABLE:
        print("  - Real 3D rendering with PyVista")
    else:
        print("\n  NOTE: Install PyVista for real 3D rendering and visual evaluation:")
        print("    pip install pyvista numpy-stl")

    print("="*60)

    # Summary and next steps
    print("\n" + "="*60)
    print("Summary and Next Steps")
    print("="*60)

    print(f"\nYour SpatialHero system achieved {evaluation.total_reward:.1%} quality!")

    # Interpretation
    if evaluation.total_reward >= 0.9:
        print("EXCELLENT - Production-ready CAD code")
    elif evaluation.total_reward >= 0.8:
        print("GOOD - High-quality CAD code with minor refinements possible")
    elif evaluation.total_reward >= 0.7:
        print("ACCEPTABLE - Valid CAD code, some improvements recommended")
    else:
        print("NEEDS WORK - Review errors and refine")

    # Show what's working
    print("\nWhat's working:")
    if evaluation.components.code_valid == 1.0:
        print("  [PASS] Code generation and execution")
    if evaluation.components.geometry_valid == 1.0:
        print("  [PASS] Geometric topology validation")
    if evaluation.components.dimension_accuracy > 0.85:
        print(f"  [PASS] Dimensional accuracy ({evaluation.components.dimension_accuracy:.1%})")

    # Show what could improve
    improvements = []
    if evaluation.components.dimension_accuracy < 0.95:
        improvements.append("  - Fine-tune dimensional constraints in prompt")
    if evaluation.components.visual_quality < 0.8:
        if not PYVISTA_AVAILABLE:
            improvements.append("  - Install PyVista for visual evaluation: pip install pyvista numpy-stl")
        else:
            improvements.append("  - Adjust design for better visual appeal")

    if improvements:
        print("\nSuggested improvements:")
        for imp in improvements:
            print(imp)

    # Next steps
    print("\nNext steps:")
    print("  1. Try: python test_rendering.py (test 3D rendering)")
    print("  2. Try: python compare_architectures.py (see all improvements)")
    print("  3. Try: python evaluate.py --dataset ../data/seed_prompts.json")

    if not PYVISTA_AVAILABLE:
        print("\n  BOOST YOUR SCORES: Install PyVista for +5-10% reward improvement")
        print("    pip install pyvista numpy-stl")

    print("="*60)


if __name__ == "__main__":
    main()
