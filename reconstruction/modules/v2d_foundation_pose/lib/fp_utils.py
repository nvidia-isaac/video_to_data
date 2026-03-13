"""
Re-export commonly used FoundationPose utility functions.

This wrapper lets other packages import GPU-accelerated depth filtering
and visualization helpers via a clean package path instead of manipulating
sys.path themselves:

    from v2d.foundation_pose.lib.fp_utils import erode_depth, bilateral_filter_depth, depth2xyzmap
"""
import os
import sys

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationPose")
if _FP_DIR not in sys.path:
    sys.path.insert(0, _FP_DIR)

from Utils import (  # noqa: E402
    erode_depth,
    bilateral_filter_depth,
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
)
