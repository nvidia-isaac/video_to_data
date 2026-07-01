"""TRAINING config to teach the native Vega-right policy the NAIL-DRIVING strike (#2: warm-start from
the grasp+lift expert `00_vega_right_v4`, then learn to bring the hammer over the nail and strike down).

Same training recipe/fixes as `VegaHammerRightEnvCfg` (compat=False, grip friction, keypoint fix,
move_scene, tolerance curriculum, reset noise, no cameras) but the GOAL is the nail-driving TIGHTEN
trajectory (lift->reorient->over->strike) with a physical drivable screw + `nail_driven` termination, so
the reward's reach-goal/strike terms train the strike. The eval/deploy genuine-strike FAILURE guards
(nail_move_eps / strike_contact) are left OFF during training (they'd prematurely fail exploration);
they're re-enabled in the deploy cfg for validation/collection.

Warm-start:  train.py --task Isaac-SimToolReal-Vega-Hammer-Right-Strike-Direct-v0 \
   --checkpoint .../00_vega_right_v4/nn/00_vega_right_v4.pth --resume_mode weights --num_envs 6144 --headless
"""
from __future__ import annotations

from isaaclab.utils import configclass

from .vega_hammer_right_env_cfg import VegaHammerRightEnvCfg


@configclass
class VegaHammerRightStrikeEnvCfg(VegaHammerRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()   # full training recipe + fixes (delta goals, curriculum, reset noise, no cams)
        # --- switch the GOAL from delta lift to the nail-driving tighten trajectory ---
        self.use_tighten_goals = True
        self.use_fixed_goal_trajectory = False
        self.physical_screw = True                # a real drivable nail (so the strike is rewarded)
        self.terminate_on_nail_driven = -0.006    # episode ends (success) when the nail is seated
        # eval-only genuine-strike FAILURE guards OFF during training (avoid premature fail-terminations
        # while the policy explores the strike); re-enabled in the deploy cfg.
        self.nail_strike_contact_dist = None
        self.nail_move_eps = None
        self.nail_hand_reject_dist = None
        self.episode_length_s = 800 / 60.0        # room to lift->reorient->over->strike (13 s)
        # the PHYSICAL screw at many envs generates far more contacts than the kinematic-screw training
        # (v3/v4) -> the default 2**30 collision stack overflows ("contacts dropped"). Bump it so nail
        # contacts are computed correctly (train at a moderate num_envs, e.g. 3072, to keep it tractable).
        self.sim.physx.gpu_collision_stack_size = 2 ** 31
        # GOAL DIVERSIFY: per-env variation of the nail_traj shape (lift_height / swing_angle / n_strikes,
        # sample_diversify_params). Train WITH it so the policy is robust to the diversified trajectories
        # we use in data collection (this is one of the collection perturbations; the 'all-perturbation'
        # eval enables it too). The clean 'no-perturbation' eval leaves it off (base trajectory).
        self.goal_diversify = True
        self.goal_diversify_scale = 1.0
        # keep from the parent: curriculum (start tol 0.075 -> 0.01), reset noise, fixed_size_success,
        # per_env_camera=False, compat=False, grip friction, keypoint fix, move_scene, table_dist=0.
