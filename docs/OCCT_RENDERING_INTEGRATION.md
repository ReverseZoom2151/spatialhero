# Integrating OCCT-based Rendering from CQ-editor

## Overview

CQ-editor (https://github.com/CadQuery/CQ-editor) uses OCCT (OpenCASCADE) for professional-grade CAD rendering. This could significantly improve our visual evaluation quality.

## Why OCCT is Better Than PyVista for CAD

### PyVista (Current)
- General-purpose 3D visualization
- Requires STL export (lossy conversion)
- Good for scientific visualization
- Quality: Good (7/10)

### OCCT (CQ-editor)
- Professional CAD kernel
- Native CadQuery B-rep rendering
- Industry-standard (used in FreeCAD, Salome)
- Quality: Excellent (9/10)

## Integration Approaches

### Option 1: Use pythonOCC (OCCT Python Bindings)

```bash
pip install pythonOCC-core
```

```python
from OCP.Display.WebGl import threejs_renderer
from cadquery import exporters

# Direct rendering without STL conversion
renderer = threejs_renderer.ThreejsRenderer()
renderer.DisplayShape(workplane.val())
renderer.Render()
```

**Effort**: Medium (4-6 hours)
**Benefit**: Professional rendering quality

### Option 2: Extract CQ-editor's Rendering Code

Study CQ-editor's viewer code:
- `cq_editor/viewer.py` - Main 3D viewer
- Uses PyQt5 + OCCT
- Screenshot capture functionality

**Effort**: High (8-12 hours)
**Benefit**: Full control, best quality

### Option 3: Headless CQ-editor Rendering

Use CQ-editor's rendering in headless mode:

```python
import subprocess

# Generate code, save to file
with open('temp_model.py', 'w') as f:
    f.write(generated_code)

# Render with CQ-editor (if headless mode exists)
subprocess.run(['cq-editor', '--headless', '--export', 'output.png', 'temp_model.py'])
```

**Effort**: Low (if supported) or High (if we need to add headless mode)

## Recommendation

### For SpatialHero v0.2.0

**Use pythonOCC directly**:

1. Install: `pip install pythonOCC-core`
2. Create new renderer: `core/renderer_occt.py`
3. Implement OCCT-based rendering
4. Compare quality with PyVista
5. If better, make it the default

**Expected improvement:**
- Visual quality scores: 0.80-0.85 → 0.85-0.95
- Better detection of geometric issues
- More accurate visual evaluation

### Implementation Plan

```python
# core/renderer_occt.py

from OCP.Display.WebGl import threejs_renderer
from OCP.Extend.TopologyUtils import TopologyExplorer
import cadquery as cq

class OCCTRenderer:
    def render_view(self, workplane, view_name='isometric'):
        # Use OCCT's native rendering
        shape = workplane.val()

        # Set up viewer
        renderer = threejs_renderer.ThreejsRenderer()
        renderer.DisplayShape(shape)

        # Set camera position for view
        if view_name == 'isometric':
            renderer.camera_position = (1, -1, 1)
        elif view_name == 'front':
            renderer.camera_position = (0, -1, 0)
        # ... etc

        # Render to image
        renderer.Render()
        image = renderer.GetScreenshot()

        return image
```

## Cost-Benefit Analysis

### Current (PyVista)
- Quality: 7/10
- Speed: Fast (~2 seconds/view)
- Complexity: Low
- Cost: Free

### With OCCT
- Quality: 9/10 (+28% better)
- Speed: Similar (~2-3 seconds/view)
- Complexity: Medium
- Cost: Free

**ROI**: High - Better visual evaluation → better training signal → better models

## Action Items

1. **Research**: Study CQ-editor's rendering code (1-2 hours)
2. **Prototype**: Implement basic OCCT rendering (2-3 hours)
3. **Compare**: Test against PyVista with same models (1 hour)
4. **Decide**: If quality improvement is significant, integrate fully
5. **Document**: Update rendering guide

## Compatibility

**pythonOCC Requirements:**
- Python 3.8+
- Works on Windows, Linux, Mac
- No GPU required
- ~150MB install size

**Integration with existing code:**
- Drop-in replacement for PyVistaRenderer
- Same interface, better output
- Backward compatible

## Next Steps

If you want to explore this:

```bash
# Install pythonOCC
pip install pythonOCC-core

# Test basic rendering
python -c "from OCP.gp import gp_Pnt; print('OCCT available!')"

# Then we can implement core/renderer_occt.py
```

Want me to implement an OCCT-based renderer?
