"""Config for the GR1-Screwdriver SCENE/SCAFFOLD env.

A fresh DirectRLEnv (NOT a SimToolReal subclass -- the GR-1 is a different morphology, so the
IIWA+Sharpa 29-DOF obs/action and the pretrained policy do NOT apply). It simply stands up the
scene: a FIXED-BASE Fourier GR1T2 humanoid (right arm + right 6-DOF hand controllable) plus the
screwdriver-task objects (044 screwdriver + flat screw + thread_test) on a table. Observations are
the controlled-joint state + object poses; the action drives the right arm + right hand joint
position targets. Reward is a placeholder (0) -- this is a starting point to teleop / train / build
a real task on, not a working controller.

The GR1T2 USD is on the Isaac nucleus server (ISAAC_NUCLEUS_DIR), so this needs nucleus access.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # the Fourier GR1T2 humanoid (nucleus USD)

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
SCREWDRIVER_USD = f"{ASSETS}/044_screwdriver/044_screwdriver.usd"
SCREW_USD = f"{ASSETS}/flat_screw/flat_screw.usd"
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"

# The 18 right-side joints we control (from the GR1T2 probe): right arm (7) + right hand (11).
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
]
RIGHT_HAND_JOINTS = [
    "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint", "R_index_intermediate_joint", "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint", "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_distal_joint",
]
CONTROLLED_JOINTS = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS  # 18
RIGHT_HAND_LINK = "right_hand_pitch_link"                 # the right wrist/palm body (for obs)

# --- fixed-base GR1T2: pick_place manipulation pose + fix the pelvis to the world + hold the lower
# body (the HIGH_PD cfg drops the leg/head actuators, so re-add a holding one). Gravity is already
# disabled in the GR1T2 cfg, so the free-standing fixed manipulator keeps its pose.
_GR1 = copy.deepcopy(GR1T2_HIGH_PD_CFG)
_GR1.prim_path = "/World/envs/env_.*/Robot"
_GR1.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.93),
    rot=(0.7071, 0.0, 0.0, 0.7071),   # face +y (toward the table), as in pick_place
    joint_pos={
        "right_elbow_pitch_joint": -1.5708, "left_elbow_pitch_joint": -1.5708,  # elbows bent, forearms forward
        "right_shoulder_.*": 0.0, "right_wrist_.*": 0.0,
        "left_shoulder_.*": 0.0, "left_wrist_.*": 0.0,
        "head_.*": 0.0, "waist_.*": 0.0, ".*_hip_.*": 0.0, ".*_knee_.*": 0.0, ".*_ankle_.*": 0.0,
        "R_.*": 0.0, "L_.*": 0.0,
    },
    joint_vel={".*": 0.0},
)
_GR1.spawn.articulation_props.fix_root_link = True            # FIXED base (no balancing needed)
_GR1.actuators = dict(_GR1.actuators)
_GR1.actuators["lower_body"] = ImplicitActuatorCfg(           # hold legs+head at init (HIGH_PD dropped these)
    joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "head_.*"],
    stiffness=200.0, damping=20.0,
)


@configclass
class GR1ScrewdriverEnvCfg(DirectRLEnvCfg):
    decimation = 2                       # 1/120 sim * 2 = 60 Hz control
    episode_length_s = 10.0
    # 18 controlled joints (right arm 7 + right hand 11)
    action_space = 18
    # obs: controlled joint pos(18) + vel(18) + screwdriver pose(7) + screw pose(7) + right-hand pose(7)
    observation_space = 57
    state_space = 0

    action_scale = 0.5                   # joint-pos target = default + action * action_scale (rad)

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_collision_stack_size=2 ** 28),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = _GR1

    # screwdriver = a dynamic, graspable tool resting on the table in front of the right hand
    screwdriver_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREWDRIVER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.45, 0.96)),
    )
    # screw + thread_test = a static fixture on the table (kinematic)
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_USD, scale=(0.013, 0.013, 0.013),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.60, 0.95)),
    )
    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD, scale=(0.005, 0.005, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.60, 0.95)),
    )
    # static table (kinematic cuboid; top at z = 0.45 + 0.95/2 ... -> set so top ~= 0.95)
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.7, 0.95),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, 0.475)),  # top at z=0.95
    )
