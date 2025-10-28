"""
Real 3D rendering implementation using PyVista.

This replaces the placeholder renderer with actual 3D visualization.
"""

import os
import tempfile
import cadquery as cq
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")


@dataclass
class Camera:
    """Camera configuration for rendering."""
    position: Tuple[float, float, float]
    focal_point: Tuple[float, float, float]
    view_up: Tuple[float, float, float]


# Standard camera positions for different views
CAMERA_CONFIGS = {
    'isometric': Camera(
        position=(1, -1, 1),  # Isometric angle
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1)
    ),
    'front': Camera(
        position=(0, -1, 0),  # Looking from front
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1)
    ),
    'top': Camera(
        position=(0, 0, 1),  # Looking from above
        focal_point=(0, 0, 0),
        view_up=(0, 1, 0)
    ),
    'side': Camera(
        position=(1, 0, 0),  # Looking from right side
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1)
    ),
    'back': Camera(
        position=(0, 1, 0),  # Looking from back
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1)
    ),
    'bottom': Camera(
        position=(0, 0, -1),  # Looking from below
        focal_point=(0, 0, 0),
        view_up=(0, -1, 0)
    ),
}


class PyVistaRenderer:
    """
    3D renderer using PyVista for high-quality CAD visualization.

    This provides actual 3D rendering (not placeholder) for visual evaluation.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        background: str = 'white',
        lighting: bool = True,
        anti_aliasing: bool = True
    ):
        """
        Initialize PyVista renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            background: Background color
            lighting: Enable lighting
            anti_aliasing: Enable anti-aliasing for smoother edges
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is required for 3D rendering. "
                "Install it with: pip install pyvista"
            )

        self.width = width
        self.height = height
        self.background = background
        self.lighting = lighting
        self.anti_aliasing = anti_aliasing

    def render_view(
        self,
        workplane: cq.Workplane,
        view_name: str = 'isometric',
        color: str = 'gold',
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Render a single view of a CADQuery workplane.

        Args:
            workplane: CADQuery Workplane object
            view_name: Name of view ('isometric', 'front', 'top', etc.)
            color: Mesh color
            output_path: Optional path to save image

        Returns:
            PIL Image of the rendered view
        """
        if view_name not in CAMERA_CONFIGS:
            raise ValueError(f"Unknown view: {view_name}. Options: {list(CAMERA_CONFIGS.keys())}")

        # Create temporary STL file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            stl_path = tmp.name

        try:
            # Export CADQuery model to STL
            cq.exporters.export(workplane, stl_path)

            # Load mesh with PyVista
            mesh = pv.read(stl_path)

            # Create off-screen plotter
            plotter = pv.Plotter(
                off_screen=True,
                window_size=(self.width, self.height)
            )

            # Add mesh with styling
            plotter.add_mesh(
                mesh,
                color=color,
                show_edges=False,
                smooth_shading=True,
                specular=0.5,
                specular_power=15
            )

            # Set background
            plotter.set_background(self.background)

            # Configure camera
            camera_config = CAMERA_CONFIGS[view_name]

            # Get mesh bounds for proper camera distance
            bounds = mesh.bounds
            center = mesh.center
            max_dim = max(
                bounds[1] - bounds[0],  # x range
                bounds[3] - bounds[2],  # y range
                bounds[5] - bounds[4]   # z range
            )

            # Scale camera position based on model size
            scale = max_dim * 2.5
            camera_pos = tuple(p * scale for p in camera_config.position)
            camera_pos = tuple(camera_pos[i] + center[i] for i in range(3))

            plotter.camera_position = [
                camera_pos,
                center,  # Focal point at model center
                camera_config.view_up
            ]

            # Add lighting if enabled
            if self.lighting:
                plotter.add_light(pv.Light(
                    position=(scale, -scale, scale),
                    focal_point=center,
                    color='white',
                    intensity=0.8
                ))

            # Enable anti-aliasing
            if self.anti_aliasing:
                plotter.enable_anti_aliasing()

            # Render to image
            if output_path:
                plotter.screenshot(output_path)
                img = Image.open(output_path)
            else:
                # Render to numpy array then convert to PIL
                img_array = plotter.screenshot(return_img=True)
                img = Image.fromarray(img_array)

            plotter.close()

            return img

        finally:
            # Clean up temporary STL file
            if os.path.exists(stl_path):
                os.remove(stl_path)

    def render_multiview(
        self,
        workplane: cq.Workplane,
        views: Optional[List[str]] = None,
        color: str = 'gold',
        output_dir: Optional[str] = None
    ) -> Dict[str, Image.Image]:
        """
        Render multiple views of a CADQuery workplane.

        Args:
            workplane: CADQuery Workplane object
            views: List of view names. If None, uses default views
            color: Mesh color
            output_dir: Optional directory to save images

        Returns:
            Dictionary mapping view names to PIL Images
        """
        if views is None:
            views = ['isometric', 'front', 'top', 'side']

        results = {}

        for view_name in views:
            try:
                output_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{view_name}.png")

                img = self.render_view(workplane, view_name, color, output_path)
                results[view_name] = img

                print(f"  ✓ Rendered {view_name} view")

            except Exception as e:
                print(f"  ✗ Failed to render {view_name} view: {e}")

        return results

    def create_comparison_grid(
        self,
        renders: Dict[str, Image.Image],
        grid_cols: int = 2
    ) -> Image.Image:
        """
        Create a grid layout of multiple rendered views.

        Args:
            renders: Dictionary of view name to PIL Image
            grid_cols: Number of columns in grid

        Returns:
            Combined PIL Image in grid layout
        """
        if not renders:
            raise ValueError("No renders provided")

        images = list(renders.values())
        view_names = list(renders.keys())

        # Get dimensions from first image
        img_width, img_height = images[0].size

        # Calculate grid dimensions
        num_images = len(images)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # Create grid image
        grid_width = grid_cols * img_width
        grid_height = grid_rows * img_height
        grid_img = Image.new('RGB', (grid_width, grid_height), 'white')

        # Paste images into grid with labels
        for idx, (view_name, img) in enumerate(zip(view_names, images)):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * img_width
            y = row * img_height

            grid_img.paste(img, (x, y))

        return grid_img


class FallbackRenderer:
    """
    Fallback renderer when PyVista is not available.

    Creates simple labeled placeholders for testing.
    """

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

    def render_view(
        self,
        workplane: cq.Workplane,
        view_name: str = 'isometric',
        **kwargs
    ) -> Image.Image:
        """Create placeholder image with view label."""
        from PIL import ImageDraw, ImageFont

        img = Image.new('RGB', (self.width, self.height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Add view label
        label = f"{view_name.upper()} VIEW"
        draw.text((10, 10), label, fill=(50, 50, 50))

        # Add note about real rendering
        note = "(Install PyVista for actual 3D rendering)"
        draw.text((10, 40), note, fill=(100, 100, 100))

        return img

    def render_multiview(
        self,
        workplane: cq.Workplane,
        views: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Image.Image]:
        """Render multiple placeholder views."""
        if views is None:
            views = ['isometric', 'front', 'top', 'side']

        return {view: self.render_view(workplane, view) for view in views}


def get_renderer(width: int = 800, height: int = 600, **kwargs):
    """
    Get the best available renderer.

    Returns PyVistaRenderer if available, otherwise FallbackRenderer.

    Args:
        width: Image width
        height: Image height
        **kwargs: Additional arguments for PyVistaRenderer

    Returns:
        Renderer instance
    """
    if PYVISTA_AVAILABLE:
        return PyVistaRenderer(width=width, height=height, **kwargs)
    else:
        print("⚠ PyVista not available, using fallback renderer")
        print("  For actual 3D rendering, install: pip install pyvista")
        return FallbackRenderer(width=width, height=height)
