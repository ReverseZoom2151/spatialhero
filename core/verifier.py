"""
Geometric verification and validation system.

This module provides comprehensive validation of generated CAD code,
including syntax checking, execution verification, and geometric validation.
"""

import ast
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from utils.cad_utils import (
    CADQueryExecutor,
    validate_topology,
    measure_dimensions,
    GeometricProperties
)


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    code_valid: bool
    execution_valid: bool
    geometry_valid: bool
    errors: List[str]
    warnings: List[str]
    properties: Optional[GeometricProperties] = None
    measurements: Optional[Dict[str, Any]] = None


class CodeVerifier:
    """
    Verifies CADQuery code at multiple levels:
    1. Syntax validation
    2. Execution validation
    3. Geometric validation
    """

    # Allowed imports for safe execution
    ALLOWED_IMPORTS = {
        'cadquery',
        'cq',
        'math',
        'numpy',
        'np'
    }

    # Required CadQuery patterns
    REQUIRED_PATTERNS = [
        'cq.Workplane',
        'result',
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        timeout: int = 30
    ):
        """
        Initialize verifier.

        Args:
            strict_mode: If True, applies stricter validation rules
            timeout: Maximum execution time for code in seconds
        """
        self.strict_mode = strict_mode
        self.timeout = timeout
        self.executor = CADQueryExecutor()

    def verify(
        self,
        code: str,
        expected_dimensions: Optional[Dict[str, float]] = None,
        tolerance: float = 0.05
    ) -> ValidationResult:
        """
        Comprehensive verification of CADQuery code.

        Args:
            code: Python code string to verify
            expected_dimensions: Expected dimensions for validation
            tolerance: Tolerance for dimensional accuracy

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []

        # 1. Syntax validation
        syntax_valid, syntax_errors = self._validate_syntax(code)
        if not syntax_valid:
            return ValidationResult(
                valid=False,
                code_valid=False,
                execution_valid=False,
                geometry_valid=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # 2. Static analysis
        static_warnings = self._static_analysis(code)
        warnings.extend(static_warnings)

        # 3. Execution validation
        workplane = self.executor.execute(code, timeout=self.timeout)
        execution_valid = workplane is not None

        if not execution_valid:
            errors.append("Code failed to execute or did not produce a valid Workplane")
            return ValidationResult(
                valid=False,
                code_valid=True,
                execution_valid=False,
                geometry_valid=False,
                errors=errors,
                warnings=warnings
            )

        # 4. Geometric validation
        topology_result = validate_topology(workplane)
        geometry_valid = topology_result['valid']

        if not geometry_valid:
            errors.extend(topology_result['errors'])
        warnings.extend(topology_result['warnings'])

        properties = topology_result.get('properties')

        # 5. Dimensional measurement
        measurements = None
        if expected_dimensions:
            measurements = measure_dimensions(workplane, expected_dimensions, tolerance)
            if not measurements.get('success'):
                warnings.append(f"Dimensional measurement failed: {measurements.get('error')}")
            elif not measurements.get('all_within_tolerance', True):
                warnings.append("Some dimensions are outside acceptable tolerance")

        # Overall validity
        overall_valid = syntax_valid and execution_valid and geometry_valid

        return ValidationResult(
            valid=overall_valid,
            code_valid=syntax_valid,
            execution_valid=execution_valid,
            geometry_valid=geometry_valid,
            errors=errors,
            warnings=warnings,
            properties=properties,
            measurements=measurements
        )

    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python syntax.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return False, errors

        return True, errors

    def _static_analysis(self, code: str) -> List[str]:
        """
        Perform static analysis on the code.

        Args:
            code: Python code string

        Returns:
            List of warning messages
        """
        warnings = []

        try:
            tree = ast.parse(code)

            # Check for required imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)

            # Check if cadquery is imported
            if 'cadquery' not in imports and 'cq' not in [alias.asname for n in ast.walk(tree)
                                                           if isinstance(n, ast.Import)
                                                           for alias in n.names]:
                warnings.append("cadquery not imported")

            # Check for result variable
            has_result = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'result':
                            has_result = True
                            break

            if not has_result:
                warnings.append("No 'result' variable assigned - may fail execution")

            # Check for dangerous operations
            dangerous_ops = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            dangerous_ops.append(node.func.id)

            if dangerous_ops:
                warnings.append(f"Potentially dangerous operations detected: {', '.join(dangerous_ops)}")

            # Check for file operations
            file_ops = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['open', 'read', 'write']:
                            file_ops.append(node.func.id)

            if file_ops:
                warnings.append(f"File operations detected: {', '.join(file_ops)}")

        except Exception as e:
            warnings.append(f"Static analysis failed: {str(e)}")

        return warnings

    def verify_batch(
        self,
        codes: List[str],
        expected_dimensions_list: Optional[List[Dict[str, float]]] = None
    ) -> List[ValidationResult]:
        """
        Verify multiple code samples in batch.

        Args:
            codes: List of code strings
            expected_dimensions_list: List of expected dimensions dicts

        Returns:
            List of ValidationResults
        """
        results = []

        if expected_dimensions_list is None:
            expected_dimensions_list = [None] * len(codes)

        for code, expected_dims in zip(codes, expected_dimensions_list):
            result = self.verify(code, expected_dims)
            results.append(result)

        return results


class GeometricConstraintChecker:
    """
    Checks geometric constraints and physical plausibility.
    """

    @staticmethod
    def check_physical_plausibility(properties: GeometricProperties) -> Dict[str, Any]:
        """
        Check if geometry is physically plausible.

        Args:
            properties: Geometric properties to check

        Returns:
            Dictionary with plausibility check results
        """
        issues = []
        warnings = []

        # Check for extremely small or large dimensions
        bbox = properties.bounding_box
        min_dimension = min(bbox.width, bbox.height, bbox.depth)
        max_dimension = max(bbox.width, bbox.height, bbox.depth)

        if min_dimension < 0.1:
            warnings.append(f"Very small dimension detected: {min_dimension}mm")

        if max_dimension > 10000:
            warnings.append(f"Very large dimension detected: {max_dimension}mm")

        # Check aspect ratio
        aspect_ratios = [
            bbox.width / bbox.height if bbox.height > 0 else float('inf'),
            bbox.width / bbox.depth if bbox.depth > 0 else float('inf'),
            bbox.height / bbox.depth if bbox.depth > 0 else float('inf'),
        ]

        extreme_aspect = [ar for ar in aspect_ratios if ar > 100 or ar < 0.01]
        if extreme_aspect:
            warnings.append(f"Extreme aspect ratio detected: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f}")

        # Check volume vs bounding box ratio
        if properties.volume > 0 and bbox.volume > 0:
            fill_ratio = properties.volume / bbox.volume
            if fill_ratio < 0.01:
                warnings.append(f"Very low fill ratio: {fill_ratio:.2%} - geometry may be too sparse")
            elif fill_ratio > 0.95:
                issues.append("Fill ratio suspiciously high - may indicate errors in calculation")

        # Check surface area to volume ratio
        if properties.volume > 0:
            sa_vol_ratio = properties.surface_area / properties.volume
            # This is object-dependent, but can flag obvious issues
            if sa_vol_ratio > 1000:
                warnings.append("High surface area to volume ratio - may indicate thin or complex geometry")

        return {
            'plausible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    @staticmethod
    def check_constraints(
        properties: GeometricProperties,
        constraints: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if geometry meets specified constraints.

        Args:
            properties: Geometric properties
            constraints: Dictionary of constraints to check

        Returns:
            Dictionary mapping constraint names to pass/fail
        """
        results = {}

        bbox = properties.bounding_box

        # Check dimensional constraints
        if 'max_width' in constraints:
            results['max_width'] = bbox.width <= constraints['max_width']

        if 'max_height' in constraints:
            results['max_height'] = bbox.height <= constraints['max_height']

        if 'max_depth' in constraints:
            results['max_depth'] = bbox.depth <= constraints['max_depth']

        if 'min_volume' in constraints:
            results['min_volume'] = properties.volume >= constraints['min_volume']

        if 'max_volume' in constraints:
            results['max_volume'] = properties.volume <= constraints['max_volume']

        # Check topology constraints
        if 'must_be_closed' in constraints:
            results['must_be_closed'] = properties.is_closed == constraints['must_be_closed']

        if 'min_faces' in constraints:
            results['min_faces'] = properties.num_faces >= constraints['min_faces']

        return results
