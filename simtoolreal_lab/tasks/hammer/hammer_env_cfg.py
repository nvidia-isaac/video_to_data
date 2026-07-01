"""Config for the SimToolReal 'hammer' env — hammer a (prismatic-jointed) nail/screw.

A SEPARATE variant of the screwdriver task (reuses ScrewdriverEnv logic): the manipulated TOOL is
the claw_hammer, and the passive screw/thread_test are the SAME assets as screwdriver043 (the
cross-slot screw_new_sdf in the thread_test bar). The difference: instead of a REVOLUTE-jointed
screw (spun in), the physical screw is on a PRISMATIC joint along the screw axis -- so the hammer
drives ('nails') it in/out linearly. The nail-in goal trajectory is `nail_traj`.

Does NOT modify the screwdriver / screwdriver043 tasks -- it only subclasses the cfg + reuses the env.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS
from ..screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg

CLAW_HAMMER_USD = f"{ASSETS}/claw_hammer/claw_hammer.usd"
SCREW_NEW_USD = f"{ASSETS}/screw_new_sdf/screw_new_sdf.usd"          # cross-slot screw (043)
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"
# PRISMATIC physical assembly: thread_test FIXED base + screw on a Z-prismatic joint (nail in/out).
SCREW_ASM_PRISMATIC_USD = f"{ASSETS}/screw_assembly043_prismatic/screw_assembly043_prismatic.usd"


@configclass
class HammerEnvCfg(ScrewdriverEnvCfg):
    # nail-in goal generator (lift -> reorient face-down -> over the nail -> lower -> repeated strikes)
    goal_generator_module: str = "simtoolreal_lab.tasks.hammer.nail_traj"
    # claw_hammer keypoint box the policy sees (the pretrained policy's claw_hammer scale)
    pretrained_object_scale: tuple = (2.5, 0.5625, 0.375)

    # --- CLOSED-LOOP strikes (deploy --closed_loop sets responsive_goals=True) ---
    # When responsive_goals=True the hammer env re-aims every step at the nail's CURRENT head (which
    # SINKS as it's driven), so repeated strikes keep landing on the (sinking) nail and drive it in --
    # vs the open-loop trajectory that targets the original head and misses after the first blow.
    hammer_cl_swing: float = 0.9         # strike swing amplitude (rad): head raised this far between blows
    hammer_cl_phase_step: float = 0.4    # oscillation advance per control step (bigger -> faster blows)

    # When set (BC data collection), the episode TERMINATES once the nail prismatic joint is driven
    # to <= this value (screw seated in the hole / near the -0.008 joint limit). None = no such reset.
    terminate_on_nail_driven: float | None = None
    # the hammer's screw_asm is a PRISMATIC nail, not a revolute screw -> opt out of the inherited
    # screwdriver rotation-success condition (the hammer uses nail_driven instead).
    terminate_on_screw_rotated: float | None = None
    # TIGHTER success: in addition to the nail being driven in, require the hammer's striking face to be
    # within this distance (m) of the nail head at the seated moment -> the head is actually CONTACTING
    # the nail (a genuine strike), not the nail drifting in or a perturbation pushing it. None = looser
    # (nail-driven joint threshold only). Calibrated from the genuine-strike distance distribution.
    nail_strike_contact_dist: float | None = None
    # RULE OUT the HAND nailing the screw (a finger/palm pressing it in) instead of the hammer: if the
    # nearest fingertip is within this distance (m) of the nail head at the seated moment, treat it as a
    # failure. Genuine strikes grip the hammer by the handle so the nearest finger is ~a hammer-length
    # from the nail. None = no hand check. Calibrated from the clean genuine-strike hand distance.
    nail_hand_reject_dist: float | None = None
    # TIGHTEST: fail the episode if the nail joint moves by more than this (m) in a single step while the
    # hammer head is NOT near (> nail_strike_contact_dist) -> the nail only moves when struck. Threshold is
    # above the static-nail numerical jitter and below a real strike's mm-scale drive. None = off.
    nail_move_eps: float | None = None
    # opt out of the screwdriver's freeze-until-grasp (hammer behavior unchanged from its dataset run).
    freeze_until_grasp: bool = False

    # use the PRISMATIC physical screw (nail) by default -- the point of this task. The hammer
    # strikes it and it slides along its axis (free both directions; gravity disabled so it stays
    # put until struck). Set physical_screw=False to fall back to the kinematic screw + thread_test.
    physical_screw: bool = True

    # --- manipulated tool = the claw_hammer (drives all obs/reward/goals) ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CLAW_HAMMER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
            ),
        ),
        # claw_hammer (z-extent ~0.028, root ~centered) rests on the table top (0.53) at root z~0.545.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.545)),
    )

    # how high the nail starts above its fully-seated pose (m): it sticks UP out of the hole so the
    # hammer can drive it DOWN. Applied to BOTH the kinematic screw (+goal target) and the physical
    # prismatic screw (initial joint position), so they stay consistent.
    nail_start_height: float = 0.012

    # screw (nail) + thread_test: SAME as screwdriver043 (kinematic fallback when physical_screw=False).
    # Raised by nail_start_height so it protrudes (not fully in the hole).
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_NEW_USD,
            scale=(0.012948, 0.012948, 0.012948),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0763, 0.0, 0.5898),   # 0.5778 seated + 0.012 raised
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD,
            scale=(0.004875, 0.004875, 0.004875),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53)),
    )

    # nail head relative to the screw root (same screw asset as 043); strike target = head + clearance.
    screw_head_offset_nominal: tuple = (0.0, 0.0, 0.0091)
    # strike point seats AT the nail head (0 = contact); the hammer head's collision drives the nail.
    screw_contact_clearance: float = 0.0

    # FRICTION so the raised nail HOLDS (doesn't slide down) under gravity until the hammer drives it.
    # The screw mass is 0.05 kg -> gravity force ~0.49 N; friction 2.0 N >> that, so it stays put, but
    # a hammer blow overcomes it (and it stays at the new depth -- no pop-back).
    screw_joint_friction: float = 2.0
    screw_joint_damping: float = 0.2
    screw_joint_armature: float = 0.0

    # PRISMATIC physical screw articulation: thread_test FIXED base + screw on a Z-prismatic joint.
    # Gravity ENABLED (so 'doesn't drop' is real) but the joint friction (above) holds the raised nail
    # in place. Initial joint = nail_start_height so the nail starts proud; the hammer drives it down
    # (joint goes negative). The env keys the actuator by "spin" (kept) -> drives the prismatic joint.
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(usd_path=SCREW_ASM_PRISMATIC_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"nail_slide": 0.012}),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["nail_slide"], stiffness=0.0, damping=0.0)},
    )

    def __post_init__(self):
        # The DEFAULT hammer-task config is the one used for pretrained-policy deploy / BC data
        # collection / eval (see scripts/collect_bc_data.py, scripts/eval_specialist_client.py).
        # NOTE: per_env_camera=True -> run with cameras enabled (--enable_cameras / the deploy/collect
        # scripts set it). For training-from-scratch flip pretrained_compat/domain_randomization back.
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # --- mode: zero-shot deploy of the original pretrained SAPG policy, clean eval ---
        self.pretrained_compat = True
        self.domain_randomization = False
        self.use_tolerance_curriculum = False
        # --- success = screw seated; track each goal tightly; goal-successes don't reset the episode ---
        self.success_tolerance = 0.01
        self.success_steps = 1
        self.max_consecutive_successes = 0
        self.terminate_on_nail_driven = -0.006       # episode ends when the nail is driven in (BC success)
        # TIGHTER: also require the striking face within 30mm of the nail head at the seated moment, so
        # the success is a GENUINE strike (head contacting the nail). Calibrated from clean-hammer
        # genuine strikes (face-to-head 3-30mm; non-contact/kicked-in seatings are ~50mm+). Reject the
        # latter -> the recorded successes are real hammer blows. Override via --strike_contact.
        self.nail_strike_contact_dist = 0.030        # <- this 3cm "hammer head contacting the nail" IS the
        #   "the hammer (not the hand) nailed it" criterion: if the head is this close at the seated moment,
        #   the hammer did it. Preferred over a hand-distance requirement (rejecting valid strikes where the
        #   hand happens to be near). Hand check therefore OFF by default; enable via --hand_reject if wanted.
        self.nail_hand_reject_dist = None
        # fail if the nail moves >1mm in a step while the hammer is far (only-struck nail). 1mm >> static
        # jitter (um) and << a strike's mm-scale drive. Tune via --nail_move_eps; verify with the logged move_far.
        self.nail_move_eps = 0.001
        self.screw_contact_clearance = -0.04         # strike overshoots 4 cm below the head (drives it in)
        self.episode_length_s = 800 / 60.0           # per-episode budget (time_out)
        # --- goals: tighten-style nail-in goals, per-layout randomization, physical (prismatic) screw ---
        self.use_fixed_goal_trajectory = False
        self.use_tighten_goals = True
        self.randomize_layout = True
        self.physical_screw = True
        # shrink the HAMMER's pose-randomization range along the robot-facing (+y) axis by 0.05 on the
        # NEAR-robot side (0.10 -> 0.05): don't spawn the tool as close to the robot base. Far side (-y) kept.
        self.layout_screwdriver_y_range = (-0.10, 0.05)
        # --- work-table moved 0.15 m further from the robot (DEFAULT; objects stay put + supported). Matches
        #     the BC datasets + eval (collect_bc_data.py / eval_simtoolreal_client.py also default to 0.15). ---
        self.table_dist = 0.15
        # --- deterministic reset: startArmHigher pose, no reset noise (matches eval_interactive.py) ---
        self.reset_dof_pos_noise_arm = 0.0
        self.reset_dof_pos_noise_fingers = 0.0
        self.reset_position_noise_x = self.reset_position_noise_y = self.reset_position_noise_z = 0.0
        self.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
        self.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
        # --- robot-facing recording camera + clean visuals used for the dataset ---
        self.per_env_camera = True
        self.cam_width, self.cam_height = 640, 480
        self.cam_eye = (0.0, -0.65, 0.85)
        self.cam_lookat = (0.0, 0.30, 0.55)
        self.cam_z_far = 2.5
        self.scene.env_spacing = 4.0                 # isolate each sub-env in its (z_far-clipped) view
        self.ground_color = (0.12, 0.12, 0.12)
        self.screw_color = (0.55, 0.72, 0.82)
        # --- physical-screw contact needs a bigger GPU collision stack for ~100 envs ---
        self.sim.physx.gpu_collision_stack_size = 2 ** 30
