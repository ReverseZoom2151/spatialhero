"""
Test OCCT-based rendering and compare with PyVista.

This script tests the professional-grade OCCT renderer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cadquery as cq
from utils.banner import print_banner, print_header


def test_occt_rendering():
    """Test OCCT rendering capabilities."""
    print_banner()
    print_header("OCCT Rendering Test")

    # Check availability
    print("Checking rendering backends...")

    from core.renderer_occt import OCCT_AVAILABLE
    from core.renderer_3d import PYVISTA_AVAILABLE

    print(f"  OCCT (pythonOCC) available: {OCCT_AVAILABLE}")
    print(f"  PyVista available: {PYVISTA_AVAILABLE}")

    if not OCCT_AVAILABLE and not PYVISTA_AVAILABLE:
        print("\n[ERROR] No rendering backends available!")
        print("\nInstall options:")
        print("  1. OCCT (recommended): pip install pythonOCC-core")
        print("  2. PyVista (fallback): pip install pyvista numpy-stl")
        return

    # Create test model
    print("\n1. Creating test model (chair)...")

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

    print("   [OK] Chair model created")

    # Test with best available renderer
    print("\n2. Testing rendering...")

    from core.renderer_occt import get_best_renderer

    renderer = get_best_renderer(width=800, height=600)

    # Render multiple views
    print("\n3. Rendering views...")
    views = ['isometric', 'front', 'top', 'side']

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'occt_test_renders')
    renders = renderer.render_multiview(chair, views=views, output_dir=output_dir)

    print(f"\n   [OK] Rendered {len(renders)} views")

    if output_dir and os.path.exists(output_dir):
        print(f"\n   Images saved to: {output_dir}")
        for view_name in renders.keys():
            img_path = os.path.join(output_dir, f"{view_name}.png")
            if os.path.exists(img_path):
                size = os.path.getsize(img_path)
                print(f"     - {view_name}.png ({size:,} bytes)")

    # Create grid
    print("\n4. Creating comparison grid...")
    if hasattr(renderer, 'create_comparison_grid'):
        grid = renderer.create_comparison_grid(renders, grid_cols=2)
        grid_path = os.path.join(output_dir, 'grid.png')
        grid.save(grid_path)
        print(f"   [OK] Grid saved to: {grid_path}")

    print_header("Rendering Test Complete")

    # Summary
    if OCCT_AVAILABLE:
        print("SUCCESS: Using OCCT (professional CAD-grade rendering)")
        print("\nOCCT provides:")
        print("  - Industry-standard visualization")
        print("  - Superior quality for technical CAD")
        print("  - Better edge rendering")
        print("  - More accurate geometry representation")
    elif PYVISTA_AVAILABLE:
        print("SUCCESS: Using PyVista (good quality rendering)")
        print("\nFor even better quality, install OCCT:")
        print("  pip install pythonOCC-core")
    else:
        print("Using fallback placeholder rendering")
        print("\nFor real 3D rendering, install:")
        print("  pip install pythonOCC-core  (recommended)")
        print("  pip install pyvista          (alternative)")

    print(f"\nCheck your renders at: {output_dir}")

    return renders


if __name__ == "__main__":
    test_occt_rendering()
