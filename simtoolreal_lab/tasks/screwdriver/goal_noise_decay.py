"""Task-AGNOSTIC goal-pose NOISE schedule for BC-collection diversity (cfg.goal_noise_module).

Adds per-env, per-waypoint N(0, sigma) noise to the generated goal trajectory so the EXPERT visits a
WIDER tube of states. The BC student never sees the goal (its obs is keypoints / proprio, not the
goal), so this purely broadens the recorded state distribution -> helps the cloned policy recover
from the drift it accumulates at test time (compounding error).

Unlike screwdriver043/goal_noise.py (which keys magnitudes off NAMED task phases LIFT/REORIENT/.../
TURN), this schedule needs only the trajectory length T: the std-dev is LARGE at the start of the
trajectory and decays MONOTONICALLY (linearly) to a small value at the end. Rationale -- precision
matters most near the END of any manipulation trajectory (insertion / tightening / striking), where a
noisy goal would teach the expert to mis-seat; the coarse approach at the start is forgiving, so it
can absorb large diversity. Works for ANY task whose goal generator exposes T (tighten_traj,
nail_traj, tighten_traj043, ...).

The env (ScrewdriverEnv) calls sigma_schedule(T, phase_counts) ONCE at init, scales the result by
cfg.goal_noise_scale, then samples fresh N(0, sigma) per env each reset. phase_counts is accepted for
signature compatibility with the phase-aware schedules but is IGNORED here (task-agnostic)."""
import numpy as np

# Trajectory-START vs trajectory-END std-devs, linearly interpolated across the T waypoints. These are
# the BASE magnitudes (cfg.goal_noise_scale multiplies them); the endpoints match the proven
# screwdriver043 schedule's coarse-approach max and precision-phase min.
POS_SIGMA_START, POS_SIGMA_END = 0.030, 0.002    # position: 3.0 cm  -> 2 mm
ROT_SIGMA_START, ROT_SIGMA_END = 0.150, 0.010    # rotation: 8.6 deg -> 0.6 deg


def sigma_schedule(T, phase_counts=None):
    """Per-waypoint (pos_sigma (T,), rot_sigma (T,)) in meters / radians, decaying linearly from
    *_START at index 0 to *_END at index T-1. phase_counts is ignored (task-agnostic)."""
    p = np.linspace(0.0, 1.0, T, dtype=np.float32) if T > 1 else np.zeros(1, dtype=np.float32)
    pos = POS_SIGMA_START + (POS_SIGMA_END - POS_SIGMA_START) * p
    rot = ROT_SIGMA_START + (ROT_SIGMA_END - ROT_SIGMA_START) * p
    return pos.astype(np.float32), rot.astype(np.float32)
