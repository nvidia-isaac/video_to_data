"""
Stabilise per-frame camera intrinsics produced by monocular depth models.

Monocular models (MoGe, UniDepth, etc.) estimate focal length independently
for every frame. Small per-frame errors create apparent scale jitter and, in
particular, cause close-up objects to be placed at the wrong depth when the
estimated focal length deviates from the true value.

This module computes a single stable intrinsics estimate by taking the
temporal median of fx, fy, cx, cy across all frames. The median is robust to
occasional outlier predictions. cx/cy are optionally fixed to the image centre
instead (they rarely drift meaningfully and the median may be biased by
composition).
"""
import json
import os

import numpy as np

from v2d.common.datatypes import CameraIntrinsics


def stabilize_intrinsics(
    intrinsics_folder: str,
    output_path: str,
    fix_principal_point: bool = False,
) -> CameraIntrinsics:
    """
    Load all per-frame intrinsics from intrinsics_folder, compute the temporal
    median of fx, fy, cx, cy, and write a single stable JSON to output_path.

    Args:
        intrinsics_folder:    Directory of per-frame intrinsics JSON files
                              ({000000.json}, {000001.json}, ...).
        output_path:          Path to write the stable intrinsics JSON.
        fix_principal_point:  If True, set cx = width/2, cy = height/2 instead
                              of using the median estimated values. Useful when
                              the model's principal-point estimates are noisy.
                              Default False.

    Returns:
        The stable CameraIntrinsics object (also written to output_path).
    """
    files = sorted(
        f for f in os.listdir(intrinsics_folder) if f.endswith('.json')
    )
    if not files:
        raise RuntimeError(f"No intrinsics JSON files found in {intrinsics_folder}")

    fx_vals, fy_vals, cx_vals, cy_vals = [], [], [], []
    widths, heights = [], []

    for fname in files:
        intr = CameraIntrinsics.load(os.path.join(intrinsics_folder, fname))
        fx_vals.append(intr.fx)
        fy_vals.append(intr.fy)
        cx_vals.append(intr.cx)
        cy_vals.append(intr.cy)
        widths.append(intr.width)
        heights.append(intr.height)

    width  = int(np.median(widths))
    height = int(np.median(heights))

    fx = float(np.median(fx_vals))
    fy = float(np.median(fy_vals))
    cx = width  / 2.0 if fix_principal_point else float(np.median(cx_vals))
    cy = height / 2.0 if fix_principal_point else float(np.median(cy_vals))

    stable = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    stable.save(output_path)

    return stable
