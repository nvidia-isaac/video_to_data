"""DEPLOY / data-collection config for the native Vega-right SimToolReal policy.

`VegaHammerRightEnvCfg` is the TRAINING config (delta goals, curriculum, no cameras, kinematic screw).
This subclass reverts those to the nail-driving DEPLOY / collection setup (the same regime the retarget
collection used) so the trained policy can be validated / used to collect data: tighten (nail-driving)
goals, physical drivable screw, per-env camera, `nail_driven` success + genuine-strike guards, NO
perturbation. It KEEPS the parent's native convention (`pretrained_compat=False`), grip friction, the
keypoint-scale fix, the raised/shifted scene (move_scene), and `table_dist=0`.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from .vega_hammer_right_env_cfg import VegaHammerRightEnvCfg


@configclass
class VegaHammerRightDeployEnvCfg(VegaHammerRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()   # robot swap, compat=False, friction, keypoint fix, move_scene, table_dist=0 + TRAINING overrides
        # --- revert the training-recipe overrides to the nail-driving DEPLOY / collection regime ---
        self.use_tighten_goals = True             # nail-driving goal trajectory (lift->reorient->over->strike)
        self.use_fixed_goal_trajectory = False
        self.use_tolerance_curriculum = False
        self.success_tolerance = 0.03             # goal-advance tol used by the retarget collection
        self.success_steps = 1
        self.max_consecutive_successes = 0
        self.fixed_size_success = False           # HammerEnv deploy scores on (now-correct) object-scale keypoints
        self.force_consecutive_near_goal_steps = True   # HammerEnvCfg deploy default
        self.physical_screw = True                # a real drivable nail
        self.terminate_on_nail_driven = -0.006    # success = nail seated
        self.nail_strike_contact_dist = 0.030     # genuine-strike guard (hammer head near the nail)
        self.nail_move_eps = 0.001                # nail only moves when struck
        self.episode_length_s = 1200 / 60.0       # collection budget (20 s)
        self.per_env_camera = True                # cameras for videos / image obs
        self.domain_randomization = False         # NO perturbation
        self.eval_append_expl_coef = True         # SAPG coef_cond fed at play (the trained net expects it)
        # deterministic-ish deploy reset (no start-pose noise)
        self.reset_dof_pos_noise_arm = 0.0
        self.reset_dof_pos_noise_fingers = 0.0
