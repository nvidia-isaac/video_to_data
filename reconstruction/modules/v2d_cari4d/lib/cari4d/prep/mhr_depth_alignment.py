from __future__ import annotations

from typing import Any

import numpy as np

from lib_mhr.contact import sample_mesh_surface_points


MHR_ALIGNMENT_SURFACE_SAMPLE_COUNT = 8000


def sample_mhr_alignment_surface(vertices_cam: Any, faces: Any, *, seed: int) -> np.ndarray:
    return sample_mesh_surface_points(vertices_cam, faces, MHR_ALIGNMENT_SURFACE_SAMPLE_COUNT, seed=seed)
