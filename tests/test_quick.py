"""
Quick manual tests - run these first!

These are simple smoke tests to verify basic functionality.
Run this file directly: python tests/test_quick.py
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Load .env file from parent directory
env_path = os.path.join(parent_dir, '.env')
if os.path.exists(env_path):
    try:
        # Try with python-dotenv first
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        # Manual loading as fallback
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from core import CodeGenerator, CADRenderer, CodeVerifier, RewardModel
        from training import CADDataset
        from utils import load_config
        print("  ✓ Core imports successful")

        # Try importing training (optional)
        try:
            from training import _TRAINING_AVAILABLE
            if _TRAINING_AVAILABLE:
                print("  ✓ Training modules available")
            else:
                print("  ℹ Training modules not available (optional - only needed for fine-tuning)")
        except:
            pass

        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from utils import load_config
        config = load_config()
        print(f"  ✓ Config loaded")
        print(f"    - Code gen model: {config.models['code_generator'].model}")
        print(f"    - Reward model: {config.models['reward_model'].model}")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False


def test_verifier():
    """Test code verifier without API."""
    print("\nTesting code verifier...")
    try:
        from core import CodeVerifier

        verifier = CodeVerifier()

        # Test valid code
        valid_code = "import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)"
        result = verifier.verify(valid_code)

        print(f"  ✓ Verifier works")
        print(f"    - Code valid: {result.code_valid}")
        print(f"    - Execution valid: {result.execution_valid}")
        print(f"    - Geometry valid: {result.geometry_valid}")

        # Test invalid code
        invalid_code = "this is not valid python"
        result = verifier.verify(invalid_code)
        print(f"  ✓ Invalid code caught: {result.errors[0] if result.errors else 'syntax error'}")

        return True
    except Exception as e:
        print(f"  ✗ Verifier failed: {e}")
        return False


def test_dataset():
    """Test dataset creation."""
    print("\nTesting dataset...")
    try:
        from training import CADDataset, CADSample, _TRAINING_AVAILABLE

        samples = [
            CADSample(prompt="Create a box", category="basic"),
            CADSample(prompt="Create a cylinder", category="basic"),
        ]
        dataset = CADDataset(samples)

        print(f"  ✓ Dataset created with {len(dataset)} samples")
        print(f"    - Sample 0: {dataset[0]['prompt']}")

        if not _TRAINING_AVAILABLE:
            print(f"  ℹ Training module not available (optional)")

        return True
    except Exception as e:
        print(f"  ✗ Dataset failed: {e}")
        return False


def test_api_key():
    """Test if API key is available."""
    print("\nTesting API key...")
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("  ⚠ No API key found")
        print("    Set OPENAI_API_KEY in .env file for full tests")
        return False
    else:
        print(f"  ✓ API key found: {api_key[:7]}...{api_key[-4:]}")
        return True


def test_code_generator():
    """Test code generator (requires API key)."""
    print("\nTesting code generator (requires API)...")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ⊘ Skipped (no API key)")
        return None

    try:
        from core import CodeGenerator

        generator = CodeGenerator()
        result = generator.generate("Create a simple box")

        if result.success:
            print(f"  ✓ Generated {len(result.code)} characters of code")
            print(f"    - Has imports: {'import' in result.code.lower()}")
            print(f"    - Has result: {'result' in result.code.lower()}")
            return True
        else:
            print(f"  ✗ Generation failed: {result.error}")
            return False

    except Exception as e:
        print(f"  ✗ Generator failed: {e}")
        return False


def test_reward_model():
    """Test reward model (requires API key)."""
    print("\nTesting reward model (requires API)...")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ⊘ Skipped (no API key)")
        return None

    try:
        from core import RewardModel

        reward_model = RewardModel(use_visual_eval=False)

        simple_code = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
"""

        evaluation = reward_model.evaluate(
            code=simple_code,
            prompt="Create a box"
        )

        print(f"  ✓ Evaluation completed")
        print(f"    - Total reward: {evaluation.total_reward:.3f}")
        print(f"    - Code valid: {evaluation.components.code_valid}")
        print(f"    - Execution valid: {evaluation.components.execution_valid}")
        print(f"    - Success: {evaluation.success}")

        return True

    except Exception as e:
        print(f"  ✗ Reward model failed: {e}")
        return False


def main():
    """Run all quick tests."""
    print("="*60)
    print("SPATIALHERO QUICK TESTS")
    print("="*60)

    results = []

    # Tests that don't require API
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Verifier", test_verifier()))
    results.append(("Dataset", test_dataset()))
    results.append(("API Key", test_api_key()))

    # Tests that require API
    results.append(("Code Generator", test_code_generator()))
    results.append(("Reward Model", test_reward_model()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status:8} {name}")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠ Some tests failed. Check errors above.")
        return 1
    elif skipped > 0:
        print("\n⚠ Some tests skipped (likely need API key).")
        print("  Set OPENAI_API_KEY in .env for full testing.")
        return 0
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
