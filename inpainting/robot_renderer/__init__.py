"""Offline Dexmate Vega + Sharpa renderer for visual inpainting.

The package keeps its validation layer NumPy-only.  Heavy rendering and IK
dependencies are imported lazily by :mod:`inpainting.robot_renderer.backend` so
dry runs and unit tests do not require a GPU, OpenGL, or robot assets.
"""

from .inputs import RenderInputs, load_render_inputs

__all__ = ["RenderInputs", "load_render_inputs"]
