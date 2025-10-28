"""
Test 3D rendering capabilities.

This script tests the PyVista-based rendering system.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cadquery as cq
from core.renderer_3d import get_renderer, PYVISTA_AVAILABLE


def test_simple_rendering():
    """Test rendering a simple shape."""
    print("="*60)
    print("Testing 3D Rendering")
    print("="*60)

    # Check if PyVista is available
    print(f"\nPyVista available: {PYVISTA_AVAILABLE}")

    if not PYVISTA_AVAILABLE:
        print("\n⚠ PyVista not installed!")
        print("\nTo enable real 3D rendering:")
        print("  pip install pyvista numpy-stl")
        print("\nUsing fallback renderer for now...")
        print()

    # Create a simple model
    print("\n1. Creating test model (simple chair)...")

    # Simple chair
    seat_width = 40
    seat_depth = 40
    seat_height = 2
    leg_height = 45
    leg_width = 3

    seat = cq.Workplane("XY").box(seat_width, seat_depth, seat_height)
    leg = cq.Workplane("XY").box(leg_width, leg_width, leg_height).translate((0, 0, -leg_height/2))

    # Add four legs
    legs = seat
    for x in [-1, 1]:
        for y in [-1, 1]:
            offset_x = x * (seat_width/2 - leg_width/2)
            offset_y = y * (seat_depth/2 - leg_width/2)
            legs = legs.union(leg.translate((offset_x, offset_y, 0)))

    # Add backrest
    backrest = cq.Workplane("XY").box(4, seat_depth, 40).translate((-seat_width/2 + 2, 0, seat_height/2 + 20))
    chair = legs.union(backrest)

    print("   ✓ Chair model created")

    # Create renderer
    print("\n2. Initializing renderer...")
    renderer = get_renderer(width=800, height=600)

    if PYVISTA_AVAILABLE:
        print("   ✓ PyVista renderer initialized")
    else:
        print("   ✓ Fallback renderer initialized")

    # Render multiple views
    print("\n3. Rendering views...")
    views = ['isometric', 'front', 'top', 'side']

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test_renders')
    renders = renderer.render_multiview(chair, views=views, output_dir=output_dir)

    print(f"\n   ✓ Rendered {len(renders)} views")

    if output_dir and os.path.exists(output_dir):
        print(f"\n   Images saved to: {output_dir}")
        for view_name in renders.keys():
            img_path = os.path.join(output_dir, f"{view_name}.png")
            if os.path.exists(img_path):
                print(f"     - {view_name}.png")

    # Create grid
    print("\n4. Creating multi-view grid...")
    if hasattr(renderer, 'create_comparison_grid'):
        grid = renderer.create_comparison_grid(renders, grid_cols=2)
        grid_path = os.path.join(output_dir, 'grid.png')
        grid.save(grid_path)
        print(f"   ✓ Grid saved to: {grid_path}")

    print("\n" + "="*60)
    print("Rendering test complete!")
    print("="*60)

    if PYVISTA_AVAILABLE:
        print("\n✅ Real 3D rendering is working!")
        print(f"Check the images in: {output_dir}")
    else:
        print("\n⚠ Using placeholder rendering")
        print("Install PyVista for actual 3D renders:")
        print("  pip install pyvista numpy-stl")

    return renders


if __name__ == "__main__":
    test_simple_rendering()
