"""Config for the Vega hammer task driven by the pretrained policy via shadow-IIWA retarget.

The RIGHT arm+hand are the canonical (controlled) 29 -- the env builds obs/reward/success off the
right palm + the object, and the retarget env feeds the policy a MIRRORED shadow-IIWA obs and applies
the mirrored+IK'd action to the right arm. The LEFT arm is parked (held). `pretrained_compat=True` so
the env uses the original Q-limits / friction / object-scale conventions the policy expects.
"""
from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..hammer.hammer_env_cfg import HammerEnvCfg
from ..vega_sharpa_robot import (
    VEGA_BASE_POS, VEGA_LEFT_ARM, VEGA_PALM_OFFSET, VEGA_RIGHT_ARM,
    VEGA_RIGHT_FINGERTIP_BODIES, VEGA_RIGHT_JOINT_NAMES, VEGA_RIGHT_PALM_BODY,
    make_vega_robot_cfg_bimanual, make_vega_robot_cfg_right, move_scene,
)


@configclass
class VegaHammerRetargetEnvCfg(HammerEnvCfg):
    robot_cfg: ArticulationCfg = make_vega_robot_cfg_bimanual()
    joint_names: list = VEGA_RIGHT_JOINT_NAMES          # canonical 29 = right arm + right hand
    palm_body: str = VEGA_RIGHT_PALM_BODY
    fingertip_bodies: list = VEGA_RIGHT_FINGERTIP_BODIES
    palm_offset: tuple = VEGA_PALM_OFFSET

    # --- LEFT arm+hand: hold the thread_tester via the SAME pretrained policy (NO mirror; the Vega left
    # hand is the policy's trained morphology). The env runs a 2nd policy instance internally whose obs
    # is built around the thread_tester (object) with a constant small horizontal goal offset, so the
    # policy grasps + holds it. Enabled when a left policy is attached (deploy/collect set it). -------
    left_hold: bool = True
    left_hold_offset: tuple = (0.0, 0.0, 0.0)      # goal = the CURRENT thread_tester keypoints exactly
    #                                                (kp_rel_goal == 0) -> the left hand just HOLDS the bar
    left_object_scale: tuple = (2.0, 0.6, 0.6)     # the size cue the policy sees for the thread_tester
    # The 4 left keypoints span the SCREW-FREE portion of the bar, in the fixture LOCAL frame. Values are
    # the MEASURED thread_test bar-mesh extents (/base/geom/ref/mesh, assembly-root frame): x in
    # [-0.024, +0.283], y in [-0.039, +0.039], z in [0, 0.0488]; the screw mesh is at x~+0.029 (NEAR the
    # -x end) so the far-end face is at x=+0.283. So:
    #   - 2 keypoints = the DIAGONAL corners of the FAR-END face (x=far_x), farthest from the screw,
    #   - 2 keypoints = the MIDDLE of the bar (x=mid_x = bar mid = (-0.024+0.283)/2 ~ 0.129).
    left_kp_far_x: float = 0.283
    left_kp_mid_x: float = 0.129
    left_kp_y: float = 0.039                       # bar half-width (y), from the mesh
    left_kp_z_center: float = 0.024                # bar mid-height above the (bottom) origin (0..0.0488)
    left_kp_z_half: float = 0.024

    def __post_init__(self):
        super().__post_init__()                         # full hammer-task config (HammerEnvCfg sets compat=True)
        # the pretrained policy needs the original obs/action convention -> keep compat ON (don't flip off
        # like the train-from-scratch Vega tasks). HammerEnvCfg.__post_init__ already set it True; the
        # parent left the IIWA startArmHigher keys on robot_cfg -> reset to the Vega init pose.
        self.pretrained_compat = True
        self.eval_append_expl_coef = True               # SAPG coef_cond at obs idx 140
        self.robot_cfg.init_state.pos = VEGA_BASE_POS
        self.robot_cfg.init_state.joint_pos = dict({**VEGA_LEFT_ARM, **VEGA_RIGHT_ARM})
        # table+objects: front 0.4, UP 0.45 (raised +0.15 vs the original 0.30 so the right arm reaches the
        # workspace higher -> less downward over-extension into a near-singularity, which caused the s3e1
        # IK spike / out-of-reach). The retarget re-anchors per reset, so the policy just reaches up a bit less.
        move_scene(self, dx=0.0, dy=-0.50, dz=0.45)
        # the approved "front camera" (the perfect view): directly in front of the robot (its front is
        # -y), elevated, telephoto. Raised +0.15 in eye-z (same look-at) per the higher table.
        self.cam_eye = (0.0, -2.5, 1.75)
        self.cam_lookat = (0.0, -0.40, 1.02)
        self.cam_z_far = 7.0
        self.cam_focal = 41.0          # FoV 15% smaller than 34.56 (33.7deg -> 28.7deg horiz; zoomed in)
        # widen the work-table in the robot's LEFT/RIGHT (x) direction by 0.2 m EACH side (+0.4 total);
        # the object/fixture SAMPLING (layout ranges + init xy) is unchanged -- just more table surface.
        sx, sy, sz = self.table_cfg.spawn.size
        self.table_cfg.spawn.size = (sx + 0.4, sy, sz)
        # halve the teleport probabilities (tool teleport + joint/"hand" teleport)
        self.tool_displace_prob = self.tool_displace_prob * 0.5
        self.joint_displace_prob = self.joint_displace_prob * 0.5
