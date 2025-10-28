"""
Simple test runner for SpatialHero.

Usage:
    python run_tests.py          # Run quick tests
    python run_tests.py --unit   # Run unit tests only
    python run_tests.py --all    # Run all tests (including API tests)
"""

import sys
import subprocess
import os


def check_env():
    """Check if environment is set up."""
    print("Checking environment...")

    # Check .env file
    if not os.path.exists('.env'):
        print("  ⚠ .env file not found")
        print("  → Copy .env.example to .env and add your API key")
        return False

    # Check API key
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ⚠ OPENAI_API_KEY not set in .env")
        print("  → Unit tests will run, but integration tests will be skipped")
    else:
        print(f"  ✓ API key found: {api_key[:7]}...{api_key[-4:]}")

    return True


def run_quick_tests():
    """Run quick manual tests."""
    print("\n" + "="*60)
    print("RUNNING QUICK TESTS")
    print("="*60 + "\n")

    result = subprocess.run([sys.executable, "tests/test_quick.py"])
    return result.returncode


def run_unit_tests():
    """Run unit tests (no API required)."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS (No API required)")
    print("="*60 + "\n")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_unit.py",
        "-v",
        "--tb=short"
    ])
    return result.returncode


def run_integration_tests():
    """Run integration tests (requires API)."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS (Requires API - costs money!)")
    print("="*60 + "\n")

    # Confirm before running
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("✗ Cannot run integration tests without API key")
        print("  Set OPENAI_API_KEY in .env file")
        return 1

    print("⚠ WARNING: These tests will make API calls and incur costs!")
    print("  Estimated cost: $0.05 - $0.20")

    response = input("\nContinue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_integration.py",
        "-v",
        "--tb=short",
        "-s"
    ])
    return result.returncode


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60 + "\n")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ])
    return result.returncode


def main():
    """Main test runner."""
    # Check environment
    check_env()

    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg in ['--quick', '-q']:
            return run_quick_tests()
        elif arg in ['--unit', '-u']:
            return run_unit_tests()
        elif arg in ['--integration', '-i']:
            return run_integration_tests()
        elif arg in ['--all', '-a']:
            return run_all_tests()
        elif arg in ['--help', '-h']:
            print(__doc__)
            return 0
        else:
            print(f"Unknown option: {arg}")
            print(__doc__)
            return 1
    else:
        # Default: run quick tests
        return run_quick_tests()


if __name__ == "__main__":
    exit(main())
