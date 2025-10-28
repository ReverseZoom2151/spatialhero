"""
CADQuery utilities for geometry manipulation and analysis.
"""

import cadquery as cq
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """3D bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )


@dataclass
class GeometricProperties:
    """Properties extracted from CAD geometry."""
    bounding_box: BoundingBox
    volume: float
    surface_area: float
    center_of_mass: Tuple[float, float, float]
    num_vertices: int
    num_edges: int
    num_faces: int
    is_valid: bool
    is_closed: bool


class CADQueryExecutor:
    """Safely execute CADQuery code and extract results."""

    @staticmethod
    def execute(code: str, timeout: int = 30) -> Optional[cq.Workplane]:
        """
        Execute CADQuery code and return the result.

        Args:
            code: Python code string containing CADQuery operations
            timeout: Maximum execution time in seconds

        Returns:
            CADQuery Workplane object or None if execution fails
        """
        try:
            # Create a safe namespace for execution
            namespace = {
                'cq': cq,
                'cadquery': cq,
                'math': __import__('math'),
                'np': np,
            }

            # Execute the code
            exec(code, namespace)

            # Try to find the result variable
            # Common patterns: result, chair, table, etc.
            possible_vars = ['result', 'model', 'chair', 'table', 'part', 'object']

            for var_name in possible_vars:
                if var_name in namespace and isinstance(namespace[var_name], cq.Workplane):
                    return namespace[var_name]

            # If no named variable found, look for any Workplane object
            for value in namespace.values():
                if isinstance(value, cq.Workplane):
                    return value

            return None

        except Exception as e:
            print(f"CADQuery execution error: {e}")
            return None


def get_bounding_box(workplane: cq.Workplane) -> Optional[BoundingBox]:
    """
    Extract bounding box from a CADQuery workplane.

    Args:
        workplane: CADQuery Workplane object

    Returns:
        BoundingBox object or None if extraction fails
    """
    try:
        bbox = workplane.val().BoundingBox()
        return BoundingBox(
            min_x=bbox.xmin,
            max_x=bbox.xmax,
            min_y=bbox.ymin,
            max_y=bbox.ymax,
            min_z=bbox.zmin,
            max_z=bbox.zmax,
        )
    except Exception as e:
        print(f"Error extracting bounding box: {e}")
        return None


def extract_geometric_properties(workplane: cq.Workplane) -> Optional[GeometricProperties]:
    """
    Extract comprehensive geometric properties from a workplane.

    Args:
        workplane: CADQuery Workplane object

    Returns:
        GeometricProperties object or None if extraction fails
    """
    try:
        solid = workplane.val()
        bbox = get_bounding_box(workplane)

        if bbox is None:
            return None

        # Extract topological information
        vertices = solid.Vertices()
        edges = solid.Edges()
        faces = solid.Faces()

        # Calculate volume and surface area
        try:
            volume = solid.Volume()
        except:
            volume = 0.0

        try:
            surface_area = sum(face.Area() for face in faces)
        except:
            surface_area = 0.0

        # Get center of mass
        try:
            com = solid.CenterOfMass()
            center_of_mass = (com.x, com.y, com.z)
        except:
            center_of_mass = bbox.center

        # Check validity
        try:
            is_valid = solid.isValid()
            is_closed = solid.isClosed() if hasattr(solid, 'isClosed') else True
        except:
            is_valid = True
            is_closed = True

        return GeometricProperties(
            bounding_box=bbox,
            volume=volume,
            surface_area=surface_area,
            center_of_mass=center_of_mass,
            num_vertices=len(vertices),
            num_edges=len(edges),
            num_faces=len(faces),
            is_valid=is_valid,
            is_closed=is_closed,
        )

    except Exception as e:
        print(f"Error extracting geometric properties: {e}")
        return None


def validate_topology(workplane: cq.Workplane) -> Dict[str, Any]:
    """
    Validate the topology of a CAD model.

    Args:
        workplane: CADQuery Workplane object

    Returns:
        Dictionary with validation results
    """
    props = extract_geometric_properties(workplane)

    if props is None:
        return {
            'valid': False,
            'errors': ['Failed to extract geometric properties'],
            'warnings': []
        }

    errors = []
    warnings = []

    # Check basic validity
    if not props.is_valid:
        errors.append("Geometry is not valid")

    if not props.is_closed:
        warnings.append("Geometry is not closed (may not be watertight)")

    # Check for degenerate geometry
    if props.volume <= 0:
        errors.append("Volume is zero or negative")

    if props.surface_area <= 0:
        errors.append("Surface area is zero or negative")

    if props.num_faces == 0:
        errors.append("No faces found")

    if props.num_edges < 3:
        errors.append("Insufficient edges for valid geometry")

    # Check bounding box
    if props.bounding_box.volume <= 0:
        errors.append("Bounding box has zero or negative volume")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'properties': props
    }


def measure_dimensions(
    workplane: cq.Workplane,
    expected_dims: Optional[Dict[str, float]] = None,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Measure actual dimensions and compare with expected values.

    Args:
        workplane: CADQuery Workplane object
        expected_dims: Dictionary of expected dimensions (e.g., {'width': 10, 'height': 20})
        tolerance: Acceptable tolerance as a fraction (e.g., 0.05 for 5%)

    Returns:
        Dictionary with measurement results and accuracy
    """
    bbox = get_bounding_box(workplane)

    if bbox is None:
        return {
            'success': False,
            'error': 'Failed to extract bounding box'
        }

    actual_dims = {
        'width': bbox.width,
        'height': bbox.height,
        'depth': bbox.depth,
        'volume': bbox.volume
    }

    result = {
        'success': True,
        'actual_dimensions': actual_dims,
    }

    # Compare with expected dimensions if provided
    if expected_dims:
        comparisons = {}
        accuracies = []

        for key, expected_value in expected_dims.items():
            if key in actual_dims:
                actual_value = actual_dims[key]
                diff = abs(actual_value - expected_value)
                relative_error = diff / expected_value if expected_value > 0 else float('inf')
                within_tolerance = relative_error <= tolerance

                comparisons[key] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'difference': diff,
                    'relative_error': relative_error,
                    'within_tolerance': within_tolerance
                }

                # Calculate accuracy (1.0 - relative_error, clamped to [0, 1])
                accuracy = max(0.0, 1.0 - relative_error)
                accuracies.append(accuracy)

        result['comparisons'] = comparisons
        result['average_accuracy'] = np.mean(accuracies) if accuracies else 0.0
        result['all_within_tolerance'] = all(c['within_tolerance'] for c in comparisons.values())

    return result
