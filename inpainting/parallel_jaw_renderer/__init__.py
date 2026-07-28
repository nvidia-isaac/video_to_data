"""Generic offline renderer for bimanual parallel-jaw embodiments.

This package is intentionally separate from :mod:`inpainting.robot_renderer`,
which remains the Vega/Sharpa implementation.  The public surface here is
small: load a strict semantic target archive and an explicit robot bundle,
solve the shared-root dual-arm trajectory, then render the complete URDF.
"""

from .bundle import RobotBundle, load_robot_bundle
from .inputs import ParallelJawInputs, load_parallel_jaw_inputs
from .render import render_parallel_jaw_robot

__all__ = [
    "ParallelJawInputs",
    "RobotBundle",
    "load_parallel_jaw_inputs",
    "load_robot_bundle",
    "render_parallel_jaw_robot",
]
