# Scripts

Utility scripts for testing and development.

## Contents

### run_tests.py

Test runner script with multiple modes:

```bash
# Quick tests (no API)
python scripts/run_tests.py --quick

# Unit tests (no API)
python scripts/run_tests.py --unit

# Integration tests (requires API key, costs ~$0.20)
python scripts/run_tests.py --integration

# All tests
python scripts/run_tests.py --all
```

## Running from Root

All scripts are designed to be run from the project root directory:

```bash
cd spatialhero
python scripts/run_tests.py
```

## Alternative Test Commands

You can also run tests directly:

```bash
# Quick smoke test
python tests/test_quick.py

# Unit tests
pytest tests/test_unit.py -v

# Integration tests
pytest tests/test_integration.py -v
```
