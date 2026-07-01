"""Modular goal-pose NOISE schedule for finetune DIVERSITY (loaded via cfg.goal_noise_module).

Adds per-env, per-goal-index noise to the generated tighten trajectory so the policy sees varied
goal poses (robustness), with a schedule that is LARGE in the lift / reorient / move-over phases and
DECAYS to ~0 in the insertion (lower) + rotation (tighten) phases -- because that's where precision
is crucial (a noisy goal there would teach the policy to mis-seat). Off by default; the env applies
it only during training (finetune), never at eval/viz.

The env (ScrewdriverEnv) calls `sigma_schedule(T, phase_counts)` ONCE at init to get per-index
position (m) and rotation (rad) std-devs, then samples fresh N(0, sigma) noise per env each reset and
adds it to that env's goal poses. phase_counts = (N_LIFT, N_REORIENT, N_OVER, N_LOWER, N_TURN)."""
import numpy as np


def sigma_schedule(T, phase_counts):
    """Return (pos_sigma (T,), rot_sigma (T,)): per-goal-index position (m) and rotation (rad) noise."""
    nl, nr, no, nlo, nt = phase_counts
    pos = np.zeros(T, dtype=np.float32)
    rot = np.zeros(T, dtype=np.float32)
    i = 0
    # lift / reorient / over (move to screw): LARGE diversity
    for n, ps, rs in [(nl, 0.030, 0.15),      # lift:     3.0 cm / 8.6 deg
                      (nr, 0.022, 0.12),      # reorient: 2.2 cm / 6.9 deg
                      (no, 0.020, 0.08)]:     # over:     2.0 cm / 4.6 deg
        pos[i:i + n] = ps; rot[i:i + n] = rs; i += n
    # lower (insertion): decay 8 mm -> 1.5 mm, 2.9 deg -> 0.6 deg across the descent
    for k in range(nlo):
        f = k / max(1, nlo - 1)
        pos[i] = 0.008 * (1 - f) + 0.0015 * f
        rot[i] = 0.05 * (1 - f) + 0.01 * f
        i += 1
    # rotate (tighten): TINY -- precision-critical
    pos[i:i + nt] = 0.0015; rot[i:i + nt] = 0.01   # 1.5 mm / 0.6 deg
    return pos, rot
