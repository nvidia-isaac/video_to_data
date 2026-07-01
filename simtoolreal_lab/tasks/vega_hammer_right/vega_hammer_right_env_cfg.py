"""Config for a NATIVE SimToolReal expert on the Vega RIGHT arm + RIGHT Sharpa hand (train-from-scratch).

Unlike `vega_hammer_retarget` (which runs the pretrained IIWA policy on the Vega right arm via a
shadow-IIWA / IK / mirror retarget), this task trains a policy DIRECTLY in the Vega right arm's joint
space with the SimToolReal SAPG recipe -- no shadow, no IK, no mirror. The policy outputs the 29-dim
delta-joint action that the base SimToolReal action path applies to the Vega right arm (0:7) + right
hand (7:29); the LEFT arm+hand are parked (`make_vega_robot_cfg_right`).

It reuses the proven Vega hammer SCENE (HammerEnvCfg + move_scene, validated by vega_hammer /
vega_hammer_retarget) but flips the deploy/eval defaults back to the ORIGINAL SimToolReal RL TRAINING
recipe: native convention (`pretrained_compat=False`), the high grip friction forced on
(`force_grasp_friction`), random DELTA goals in a lift-forcing target volume + the tolerance
curriculum, no cameras, and the eval-only genuine-strike termination/guards removed. The original
hammer/screwdriver tasks are untouched.

Train (SAPG):  train.py --task Isaac-SimToolReal-Vega-Hammer-Right-Direct-v0 \
                 --agent_cfg rl_games_sapg_cfg.yaml --num_envs 6144 --headless
"""
from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..hammer.hammer_env_cfg import HammerEnvCfg
from ..vega_sharpa_robot import (
    VEGA_BASE_POS,
    VEGA_INIT_JOINT_POS,
    VEGA_PALM_OFFSET,
    VEGA_RIGHT_FINGERTIP_BODIES,
    VEGA_RIGHT_JOINT_NAMES,
    VEGA_RIGHT_PALM_BODY,
    make_vega_robot_cfg_right,
    move_scene,
)


@configclass
class VegaHammerRightEnvCfg(HammerEnvCfg):
    # robot: Vega RIGHT arm + RIGHT Sharpa hand are the CONTROLLED canonical 29; LEFT arm+hand parked.
    robot_cfg: ArticulationCfg = make_vega_robot_cfg_right()
    joint_names: list = VEGA_RIGHT_JOINT_NAMES
    palm_body: str = VEGA_RIGHT_PALM_BODY
    fingertip_bodies: list = VEGA_RIGHT_FINGERTIP_BODIES
    palm_offset: tuple = VEGA_PALM_OFFSET

    def __post_init__(self):
        super().__post_init__()                          # full hammer-task cfg (compat=True, DEPLOY goals/guards)

        # --- native train-from-scratch on the Vega right arm (the IIWA checkpoint can't transfer) ---
        self.pretrained_compat = False                   # native wxyz quats + USD-derived joint limits
        self.force_grasp_friction = True                 # apply the original high fingertip grip (compat is off)
        # Vega init pose (the parent wrote IIWA startArmHigher keys that don't exist on Vega).
        self.robot_cfg.init_state.pos = VEGA_BASE_POS
        self.robot_cfg.init_state.joint_pos = dict(VEGA_INIT_JOINT_POS)
        # workspace UP +0.45 / FRONT -0.50: the right arm reaches the raised workspace without the
        # near-singular over-extension at +0.30 (the retarget's out-of-reach). Matches the retarget /
        # specialist scene so a distilled specialist is directly comparable.
        MOVE = (0.0, -0.50, 0.45)
        move_scene(self, dx=MOVE[0], dy=MOVE[1], dz=MOVE[2])
        # CRITICAL: move_scene shifts the table/objects/layout but NOT the absolute delta-goal box
        # (`target_volume`, used only by the train-from-scratch delta goals; the retarget/vega_hammer use
        # screw-relative tighten goals so they never hit this). Without this shift the hammer rests at
        # z~0.545+0.45=0.995 ABOVE the entire target_volume (z<=0.95) -> the lift goal is below the
        # table-resting hammer -> UNREACHABLE -> 0 successes forever. Shift the volume with the scene.
        tvm, tvx = self.target_volume_min, self.target_volume_max
        self.target_volume_min = tuple(tvm[i] + MOVE[i] for i in range(3))
        self.target_volume_max = tuple(tvx[i] + MOVE[i] for i in range(3))
        # --- ENVIRONMENT/scene: match the as-built VEGA env (NOT the inherited IIWA defaults) ---
        # collect_bc_data forces table_dist=0 for VEGA (the 0.15 IIWA shift pushes the workspace out of
        # the right arm's reach); train on the SAME table the native expert will be collected/eval'd on.
        self.table_dist = 0.0
        # the env we built widened the work-table +0.4 m in x (object sampling unchanged) -- match it.
        sx, sy, sz = self.table_cfg.spawn.size
        self.table_cfg.spawn.size = (sx + 0.4, sy, sz)

        # --- restore the ORIGINAL SimToolReal RL TRAINING recipe (HammerEnvCfg defaults are deploy/eval) ---
        # random delta goals in a lift-forcing target volume (NOT the fixed nail-driving trajectory).
        self.use_tighten_goals = False
        self.use_fixed_goal_trajectory = False
        # tolerance curriculum: start loose so the untrained policy advances goals + earns reward, then
        # tighten multiplicatively toward 0.01 (the original toleranceCurriculum).
        self.use_tolerance_curriculum = True
        self.success_tolerance = 0.075
        self.target_success_tolerance = 0.01
        self.success_steps = 10
        self.max_consecutive_successes = 50              # reset after many goal-reaches (training variety)
        self.episode_length_s = 600 / 60.0               # original SimToolReal episodeLength (10 s @ 60 Hz)
        # RESET NOISE (HammerEnvCfg zeroes it for deterministic eval/deploy; the original TRAINING relies
        # on it for start-state diversity / grasp exploration). Restore the base SimToolReal training
        # values -- the env now applies them as the original lerp-toward-full-range distribution.
        self.reset_dof_pos_noise_arm = 0.1
        self.reset_dof_pos_noise_fingers = 0.1
        # SUCCESS regime -> match the original (which the working policy used):
        #  - fixed_size_success: score the keypoint reward + success on the scale-invariant FIXED-SIZE box
        #    (original fixedSizeKeypointReward=True), not the object-scale keypoints.
        #  - force_consecutive_near_goal_steps=False: accumulate near-goal steps NON-consecutively + pay
        #    the goal bonus densely per step (True needs 10 uninterrupted in-tol steps -> too hard cold).
        self.fixed_size_success = True
        self.force_consecutive_near_goal_steps = False
        # the nail/screw is inert during delta-goal training (the hammer lifts into the TARGET VOLUME at
        # z 0.60-0.95, well above the table; it never strikes the nail) -> no nail-driven termination and
        # no eval-only genuine-strike FAILURE guards (an untrained policy would otherwise bump the nail and
        # fail-terminate instantly). These match the original SimToolReal training (no nail concept).
        self.terminate_on_nail_driven = None
        self.nail_strike_contact_dist = None
        self.nail_move_eps = None
        self.nail_hand_reject_dist = None
        # EXCEPTION to "match the built env" (which uses physical_screw=True): the screw is provably inert
        # during delta-goal lifting, so the kinematic screw is behaviorally identical here but far cheaper
        # at 1000s of envs. Re-enabled (True) for the eval/collection config, where the nail IS struck.
        self.physical_screw = False
        # train at scale headlessly: NO cameras (per-env cameras cripple throughput + OOM at 1000s of envs).
        self.per_env_camera = False
        self.wrist_camera = False
        # domain_randomization is set by train.py (--domain_randomization, default ON).
