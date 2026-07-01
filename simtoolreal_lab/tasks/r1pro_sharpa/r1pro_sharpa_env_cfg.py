"""Config for the R1 Pro + Sharpa GR00T-N1.7 task env.

A fixed-base Galaxea R1 Pro (BEHAVIOR asset) with the 22-DOF Sharpa hands grafted on (see
scripts/build_r1pro_sharpa_urdf.py), set up for GR00T N1.7 `REAL_R1_PRO_SHARPA` inference:
  - GR00T controls only the wrists (relative-EEF -> differential IK on the 7-DOF arms) + the
    two 22-DOF hands (absolute joint targets); the base/torso are held (fix_root_link).
  - 3 cameras matching the embodiment's video keys: ego (head ZED), left/right wrist (RealSense),
    all 320x240.
This file defines the SCENE (robot + table + object + cameras); the GR00T obs/action wiring lives
in the env class.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

ROBOT_USD = "/home/cning/simtoolreal_isaaclab/assets/r1pro_sharpa/r1pro_sharpa.usd"

# the 22 Sharpa hand joints per side (articulation order), for obs/action mapping
LEFT_HAND_JOINTS = [
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
RIGHT_HAND_JOINTS = [n.replace("left_", "right_", 1) for n in LEFT_HAND_JOINTS]
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
EEF_BODIES = {"left": "left_arm_link7", "right": "right_arm_link7"}


def _rpy_to_quat(r, p, y):
    """xyz-euler (URDF rpy) -> (w,x,y,z)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                              math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _quat_mul(a, b):
    """Hamilton product of two (w,x,y,z) quats."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


# camera mount offsets relative to their parent bodies (computed from the R1 Pro URDF)
EGO_OFFSET = ((0.0655, 0.060, 0.476), _rpy_to_quat(-1.9199, 0.0, -1.5708))
WRIST_OFFSET = ((0.021, 0.0029, -0.1555), _rpy_to_quat(-3.1376, -0.4363, 3.1399))
# right wrist cam: spin 200 deg about the arm axis (arm_link7 +z) so it looks down the arm.
_RZ200 = (math.cos(math.radians(200) / 2), 0.0, 0.0, math.sin(math.radians(200) / 2))
RIGHT_WRIST_OFFSET = (WRIST_OFFSET[0], _quat_mul(_RZ200, WRIST_OFFSET[1]))


# left wrist cam = the proper sagittal MIRROR of the right. Both arm_link7 frames are ~identity
# orientation, so a naive offset mirror gives the wrong view (it ignores the camera optical-frame
# convention). This offset was computed so the LEFT cam's WORLD pose is the y-reflection of the
# RIGHT cam's world pose (pos.y negated; quat solved from the probed right-cam world orientation).
LEFT_WRIST_OFFSET = ((WRIST_OFFSET[0][0], -WRIST_OFFSET[0][1], WRIST_OFFSET[0][2]),
                     (-0.21287, -0.16914, -0.96153, 0.03933))
CAM_W, CAM_H = 320, 240


def _cam(prim_path, offset):
    return CameraCfg(
        prim_path=prim_path, width=CAM_W, height=CAM_H, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=offset[0], rot=offset[1], convention="ros"),
    )


@configclass
class R1ProSharpaEnvCfg(DirectRLEnvCfg):
    decimation = 4                       # 1/120 * 4 = 30 Hz control
    episode_length_s = 20.0
    # GR00T action: left_wrist_eef(9) + right_wrist_eef(9) + left_hand_joints(22) + right_hand_joints(22)
    action_space = 62
    observation_space = 62               # placeholder (GR00T obs is a dict from get_gr00t_obs())
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0, render_interval=decimation,
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_collision_stack_size=2 ** 28),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # --- fixed-base R1 Pro + Sharpa ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False,
                solver_position_iteration_count=12, solver_velocity_iteration_count=1),
        ),
        # joint_pos left at the USD defaults (0) for now -- a guessed manipulation pose exceeded the
        # R1 Pro arm joint limits; tune within limits once the scene is verified.
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arms": ImplicitActuatorCfg(joint_names_expr=[".*_arm_joint.*"], stiffness=600.0, damping=50.0),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*", "(left|right)_[1-5]_.*"],
                stiffness=8.0, damping=0.8),
            "body": ImplicitActuatorCfg(joint_names_expr=["torso.*", "steer.*", "wheel.*"],
                                        stiffness=300.0, damping=30.0),
        },
    )

    # --- tabletop scene: a table + a manipulable cube (default task) ---
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.74),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.37)),  # in front of the robot, top z=0.74
    )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.79)),
    )

    # --- GR00T cameras (mounted on the robot; map to ego/left_wrist/right_wrist video keys) ---
    ego_cam_cfg: CameraCfg = _cam("/World/envs/env_.*/Robot/torso_link4/ego_cam", EGO_OFFSET)
    left_wrist_cam_cfg: CameraCfg = _cam("/World/envs/env_.*/Robot/left_arm_link7/left_wrist_cam", LEFT_WRIST_OFFSET)
    right_wrist_cam_cfg: CameraCfg = _cam("/World/envs/env_.*/Robot/right_arm_link7/right_wrist_cam", RIGHT_WRIST_OFFSET)

    # GR00T inference task prompt
    task_prompt: str = "pick up the red cube"
