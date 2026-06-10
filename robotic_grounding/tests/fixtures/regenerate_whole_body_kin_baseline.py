# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Regenerate the WholeBodyKinematics numerical-baseline fixture.

Captures the exact per-frame IK outputs (``q``, ``frame_pose``,
``frame_task_errors``, ``num_optimization_iterations``) that
``WholeBodyKinematics`` produces on a synthetic SOMA-shaped input.
The companion pytest ``tests/test_whole_body_kinematics_baseline.py``
replays the same input and asserts allclose at ``atol=1e-10``.

The committed ``whole_body_kin_baseline.npz`` was captured against
the pre-merge class layout; the post-merge unified class must
reproduce it bit-for-bit. Re-run this helper only when the
canonical IK output is intentionally changed (e.g. a config schema
bump that legitimately alters the numbers), and update the PR
description accordingly.

Usage (inside the robotic-grounding-latest-gpu0 container):
    python tests/fixtures/regenerate_whole_body_kin_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.whole_body_kinematics import WholeBodyKinematics

FIXTURE_PATH = Path(__file__).parent / "whole_body_kin_baseline.npz"
SEED = 42
N_FRAMES = 5
SCALE = 1.0


def _build_synthetic_soma_input(
    n_frames: int, n_joints: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, wxyz) of shape (T, J, 3) and (T, J, 4).

    Synthetic input is sufficient for a numerical-regression baseline:
    the test only cares that the same input produces the same output
    before and after the refactor. We pick a deterministic shape
    loosely resembling a humanoid (vertical spread along z) so the
    IK runs realistic convergence iterations rather than terminating
    in one step on a degenerate input.
    """
    rng = np.random.default_rng(seed)
    # Rough body shape: joints spread vertically from 0 to 1.7m, then
    # spread laterally by a seeded offset. Hips at index 0 sits near
    # the bottom (the IK uses the lowest joint as ground anchor).
    base_z = np.linspace(0.0, 1.7, n_joints)
    base_x = 0.15 * rng.standard_normal(n_joints)
    base_y = 0.15 * rng.standard_normal(n_joints)
    base_pose = np.stack([base_x, base_y, base_z], axis=-1)
    base_pose[0] = np.array([0.0, 0.0, 0.05])  # Hips near ground

    # Per-frame perturbation: small drift around the base pose so the
    # frame-task targets actually change between frames (exercises the
    # qpos-threading and posture-task-q_prev paths).
    drift = 0.02 * rng.standard_normal((n_frames, n_joints, 3))
    positions = base_pose[None, :, :] + drift

    # Quaternions: small random perturbations from identity (wxyz),
    # normalized.
    wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, n_joints, 1))
    wxyz[..., 1:] += 0.05 * rng.standard_normal((n_frames, n_joints, 3))
    wxyz /= np.linalg.norm(wxyz, axis=-1, keepdims=True)
    return positions, wxyz


def main() -> int:
    """Capture the WholeBodyKinematics baseline fixture and print a sanity record."""
    n_joints = len(SOMA_JOINTS_ORDER)
    positions, wxyz = _build_synthetic_soma_input(
        n_frames=N_FRAMES, n_joints=n_joints, seed=SEED
    )

    config = load_robot_config("g1")
    kin = WholeBodyKinematics(config=config)

    q = kin.robot.q0.copy()
    q_per_frame: list[np.ndarray] = []
    frame_pose_per_frame: list[np.ndarray] = []
    frame_task_errors_per_frame: list[np.ndarray] = []
    num_iters_per_frame: list[int] = []
    for t in range(N_FRAMES):
        result = kin.compute(
            source_joints=positions[t],
            source_joints_wxyz=wxyz[t],
            source_to_robot_scale=SCALE,
            qpos=q,
        )
        q_per_frame.append(result["q"].copy())
        frame_pose_per_frame.append(np.asarray(result["frame_pose"]).copy())
        frame_task_errors_per_frame.append(
            np.asarray(result["frame_task_errors"], dtype=np.float64).copy()
        )
        num_iters_per_frame.append(int(result["num_optimization_iterations"]))
        q = result["q"].copy()

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE_PATH,
        positions=positions,
        wxyz=wxyz,
        q=np.stack(q_per_frame),
        frame_pose=np.stack(frame_pose_per_frame),
        frame_task_errors=np.stack(frame_task_errors_per_frame),
        num_iters=np.asarray(num_iters_per_frame, dtype=np.int64),
        scale=np.float64(SCALE),
        seed=np.int64(SEED),
    )

    # Sanity record for the PR description.
    print(f"Wrote {FIXTURE_PATH}")
    print(f"  n_frames={N_FRAMES}  n_joints={n_joints}  seed={SEED}")
    print(f"  q[0, :7]            = {q_per_frame[0][:7]}")
    print(f"  frame_task_errors[0]= {frame_task_errors_per_frame[0]}")
    print(f"  num_iters_per_frame = {num_iters_per_frame}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
