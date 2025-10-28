"""
CADQuery rendering system for generating multi-view images.
"""

import cadquery as cq
from PIL import Image, ImageDraw
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import io
import base64


@dataclass
class ViewConfig:
    """Configuration for a camera view."""
    name: str
    rotation: Tuple[float, float, float]  # (rx, ry, rz) in degrees
    description: str


# Standard view configurations
STANDARD_VIEWS = {
    'isometric': ViewConfig(
        name='isometric',
        rotation=(35.264, 0, 45),
        description='Isometric view (3D perspective)'
    ),
    'front': ViewConfig(
        name='front',
        rotation=(0, 0, 0),
        description='Front orthographic view'
    ),
    'top': ViewConfig(
        name='top',
        rotation=(90, 0, 0),
        description='Top orthographic view'
    ),
    'side': ViewConfig(
        name='side',
        rotation=(0, 0, 90),
        description='Side orthographic view'
    ),
    'back': ViewConfig(
        name='back',
        rotation=(0, 0, 180),
        description='Back orthographic view'
    ),
    'bottom': ViewConfig(
        name='bottom',
        rotation=(-90, 0, 0),
        description='Bottom orthographic view'
    ),
}


class CADRenderer:
    """
    Renders CADQuery models from multiple viewpoints.

    This renderer generates orthographic and isometric projections
    for evaluating spatial qualities of generated CAD models.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        background_color: Tuple[int, int, int] = (240, 240, 240),
    ):
        """
        Initialize renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            background_color: RGB background color
        """
        self.width = width
        self.height = height
        self.background_color = background_color

    def render_view(
        self,
        workplane: cq.Workplane,
        view: ViewConfig,
        format: str = 'PNG'
    ) -> Optional[Image.Image]:
        """
        Render a single view of a CADQuery workplane.

        Args:
            workplane: CADQuery Workplane object
            view: ViewConfig specifying camera position
            format: Image format (PNG, JPEG, etc.)

        Returns:
            PIL Image or None if rendering fails
        """
        try:
            # Try renderers in order of quality: OCCT > PyVista > Fallback
            try:
                # First try OCCT (best quality for CAD)
                from core.renderer_occt import get_best_renderer
                renderer_3d = get_best_renderer(
                    width=self.width,
                    height=self.height,
                    background='white'
                )
                img = renderer_3d.render_view(workplane, view.name, color='gold')
                return img

            except ImportError:
                # Fallback to placeholder if no renderers available
                img = Image.new('RGB', (self.width, self.height), self.background_color)
                draw = ImageDraw.Draw(img)

                # Add view label
                label = f"{view.name.upper()} VIEW"
                draw.text((10, 10), label, fill=(50, 50, 50))

                # Add note
                note = "(Install pythonOCC-core or PyVista for 3D rendering)"
                draw.text((10, 40), note, fill=(100, 100, 100))

                return img

        except Exception as e:
            print(f"Error rendering {view.name} view: {e}")
            return None

    def render_multiview(
        self,
        workplane: cq.Workplane,
        views: Optional[List[str]] = None
    ) -> Dict[str, Image.Image]:
        """
        Render multiple views of a CADQuery workplane.

        Args:
            workplane: CADQuery Workplane object
            views: List of view names (e.g., ['isometric', 'front', 'top'])
                   If None, uses default views

        Returns:
            Dictionary mapping view names to PIL Images
        """
        if views is None:
            views = ['isometric', 'front', 'top', 'side']

        results = {}

        for view_name in views:
            if view_name not in STANDARD_VIEWS:
                print(f"Warning: Unknown view '{view_name}', skipping")
                continue

            view_config = STANDARD_VIEWS[view_name]
            img = self.render_view(workplane, view_config)

            if img is not None:
                results[view_name] = img

        return results

    def export_to_step(
        self,
        workplane: cq.Workplane,
        filepath: str
    ) -> bool:
        """
        Export CADQuery workplane to STEP file.

        Args:
            workplane: CADQuery Workplane object
            filepath: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            cq.exporters.export(workplane, filepath)
            return True
        except Exception as e:
            print(f"Error exporting to STEP: {e}")
            return False

    def export_to_stl(
        self,
        workplane: cq.Workplane,
        filepath: str,
        tolerance: float = 0.01
    ) -> bool:
        """
        Export CADQuery workplane to STL file.

        Args:
            workplane: CADQuery Workplane object
            filepath: Output file path
            tolerance: Meshing tolerance

        Returns:
            True if successful, False otherwise
        """
        try:
            cq.exporters.export(workplane, filepath, tolerance=tolerance)
            return True
        except Exception as e:
            print(f"Error exporting to STL: {e}")
            return False

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string for API transmission.

        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    @staticmethod
    def create_multiview_grid(
        images: Dict[str, Image.Image],
        grid_layout: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Combine multiple view images into a single grid layout.

        Args:
            images: Dictionary mapping view names to PIL Images
            grid_layout: (rows, cols) tuple. If None, auto-calculated

        Returns:
            Combined PIL Image in grid layout
        """
        if not images:
            raise ValueError("No images provided")

        # Get dimensions from first image
        first_img = next(iter(images.values()))
        img_width, img_height = first_img.size

        # Calculate grid layout
        num_images = len(images)
        if grid_layout is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_layout

        # Create grid image
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        # Paste images into grid
        for idx, (view_name, img) in enumerate(images.items()):
            row = idx // cols
            col = idx % cols
            x = col * img_width
            y = row * img_height
            grid_img.paste(img, (x, y))

            # Add label
            draw = ImageDraw.Draw(grid_img)
            draw.text((x + 10, y + 10), view_name.upper(), fill=(50, 50, 50))

        return grid_img


class RenderCache:
    """
    Cache for rendered images to avoid redundant rendering.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize render cache.

        Args:
            max_size: Maximum number of cached renders
        """
        self.cache: Dict[str, Dict[str, Image.Image]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def get(self, code_hash: str) -> Optional[Dict[str, Image.Image]]:
        """
        Get cached renders for a code hash.

        Args:
            code_hash: Hash of the CADQuery code

        Returns:
            Dictionary of rendered images or None if not cached
        """
        if code_hash in self.cache:
            # Update access order (LRU)
            self.access_order.remove(code_hash)
            self.access_order.append(code_hash)
            return self.cache[code_hash]
        return None

    def put(self, code_hash: str, renders: Dict[str, Image.Image]) -> None:
        """
        Cache rendered images for a code hash.

        Args:
            code_hash: Hash of the CADQuery code
            renders: Dictionary of rendered images
        """
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and code_hash not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[code_hash] = renders
        if code_hash not in self.access_order:
            self.access_order.append(code_hash)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
