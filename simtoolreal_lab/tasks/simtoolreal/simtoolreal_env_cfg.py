"""Config for the SimToolReal DirectRLEnv (IIWA14 + left Sharpa, claw_hammer / swing_down).

Ported from the Isaac Gym `SimToolReal` task (cfg/task/SimToolReal.yaml). Values here mirror
that YAML; fidelity TODOs (exact PD gains, perturbation forces) are flagged inline.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .robot_gains import (
    ARM_DAMPING,
    ARM_EFFORT,
    ARM_STIFFNESS,
    HAND_ARMATURE,
    HAND_DAMPING,
    HAND_EFFORT,
    HAND_FRICTION,
    HAND_STIFFNESS,
)

# Resolved at import; vertical-slice paths are absolute for now.
ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
ROBOT_USD = f"{ASSETS}/iiwa14_left_sharpa/robot.usd"
OBJECT_USD = f"{ASSETS}/claw_hammer/claw_hammer.usd"
TRAJECTORY = "/home/cning/simtoolreal_isaaclab/trajectories/hammer/claw_hammer/swing_down.json"

# Canonical SimToolReal 29-DOF joint order (arm 0:7, hand 7:29).
JOINT_NAMES_ISAACGYM = [
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
PALM_BODY = "iiwa14_link_7"
FINGERTIP_BODIES = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]

# Robot base offset in env frame (T_W_R translation), from observation_action_utils_sharpa.py.
ROBOT_BASE_POS = (0.0, 0.8, 0.0)


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    # --- env / spaces ---
    decimation = 2  # sim dt 1/120 * 2 = 1/60 control (controlFrequencyInv: 1 @ 60 Hz)
    episode_length_s = 600 / 60.0  # episodeLength 600 steps @ 60 Hz = 10 s
    action_space = 29
    observation_space = 140  # N_OBS from ported obs layout
    # asymmetric critic state: obs(140) + obj_linvel(3) + obj_angvel(3) + lifted(1)
    # + keypoints_max_dist(1) + closest_keypoint_max_dist(1) + closest_fingertip_dist(5) + successes(1)
    state_space = 155

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=1.2, replicate_physics=True
    )

    # --- optional per-env camera (one TiledCamera view per sub-env, to record ALL envs in a single
    # rollout). OFF by default so training/normal eval pays no render cost. Resolution is per-env. ---
    per_env_camera: bool = False
    cam_width: int = 1280
    cam_height: int = 720
    cam_eye: tuple = (-0.10, -0.62, 0.90)    # per-env camera eye (env-local)
    cam_lookat: tuple = (0.15, 0.0, 0.64)    # per-env camera look-at (env-local)

    # --- robot (fix_base baked into USD at conversion; placed at base offset) ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            # Arm pose hovering the hand over the table object (env.py:476 Sharpa variant).
            # Only the 7 arm joints are listed; the 22 hand joints keep their USD default (0).
            joint_pos={
                "iiwa14_joint_1": -1.571,
                "iiwa14_joint_2": 1.571,
                "iiwa14_joint_3": 0.0,
                "iiwa14_joint_4": 1.376,
                "iiwa14_joint_5": 0.0,
                "iiwa14_joint_6": 1.485,
                "iiwa14_joint_7": 1.308,
            },
        ),
        actuators={
            # Per-joint Kp/Kd/effort ported verbatim from the original SimToolReal (robot_gains.py).
            # Arm: no armature (matches real KUKA). Hand: + armature + joint friction.
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["iiwa14_joint_.*"],
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
                effort_limit=ARM_EFFORT,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["left_.*"],
                stiffness=HAND_STIFFNESS,
                damping=HAND_DAMPING,
                effort_limit=HAND_EFFORT,
                armature=HAND_ARMATURE,
                friction=HAND_FRICTION,
            ),
        },
    )

    # --- manipulated object (free rigid body) ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
            ),
        ),
        # object centered under the hand (palm ~(0,0,0.67)), spawned just ABOVE the table
        # top (0.53) so its convex hull does not penetrate the kinematic table on spawn
        # (penetration was ejecting it laterally off the table).
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.60)),
    )

    # --- static table (kinematic cuboid; top surface at z=0.53) ---
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
    )

    # --- action mapping (observation_action_utils_sharpa.compute_joint_pos_targets) ---
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    dof_speed_scale = 1.5
    control_dt = 1.0 / 60.0

    # --- keypoints ---
    object_base_size = 0.04
    keypoint_scale = 1.5

    # --- reward scales (SimToolReal.yaml) ---
    lifting_rew_scale = 20.0
    lifting_bonus = 300.0
    lifting_bonus_threshold = 0.15
    keypoint_rew_scale = 200.0
    distance_delta_rew_scale = 50.0
    reach_goal_bonus = 1000.0
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003
    fall_distance = 0.24
    fall_penalty = 0.0
    object_lin_vel_penalty_scale = 0.0
    object_ang_vel_penalty_scale = 0.0
    reward_shaper_scale = 0.01  # rl_games reward_shaper.scale_value

    # --- success / resets ---
    success_tolerance = 0.075
    max_consecutive_successes = 50
    success_steps = 10
    force_consecutive_near_goal_steps = True  # launch_training override

    # --- reset noise (SimToolReal.yaml) ---
    reset_position_noise_x = 0.1
    reset_position_noise_y = 0.1
    reset_position_noise_z = 0.02
    reset_dof_pos_noise_fingers = 0.1
    reset_dof_pos_noise_arm = 0.1
    reset_dof_vel_noise = 0.5

    clamp_abs_observations = 10.0

    # --- goal sampling (training): random delta goals (goalSamplingType: delta) ---
    use_fixed_goal_trajectory = False  # True at eval: step through the trajectory goals
    trajectory_path = TRAJECTORY

    # SAPG eval: append the exploit exploration-coefficient (0.0) at obs index 140 so the
    # coef_cond network selects the exploit block (training adds this in a2c_common; the
    # stock rl_games player does not). Leave False for training and for non-SAPG (PPO) eval.
    eval_append_expl_coef = False
    expl_exploit_coef = 0.0  # = linspace(50,0,6)[-1] -> exploit block

    # --- pretrained Isaac Gym checkpoint compatibility (zero-shot deploy) ---
    # When True, the env emits obs / consumes actions in the EXACT original convention so the
    # Isaac Gym pretrained checkpoint can run zero-shot (see env.__init__ / _get_observations):
    #  - object_scales = dextoolbench "scale given to policy" (objects.py), NOT metric;
    #  - palm_rot/object_rot in XYZW; joint_pos unscale + hand-action scale via Q_{LOWER,UPPER}.
    pretrained_compat = False
    # claw_hammer: rescale_by_factor((0.10,0.0225,0.015), 25) = bbox / object_base_size(0.04).
    pretrained_object_scale = (2.5, 0.5625, 0.375)

    # --- demo mode (reproduce dextoolbench/eval_interactive.py scenario) ---
    # When True: object initialized at the trajectory's recorded start_pose (+ demo_z_offset, no
    # reset noise) and SUCCESS is measured on the fixed-size keypoints (fixedSizeKeypointReward).
    # The demo script also sets: success_tolerance=0.01, success_steps=1, use_tolerance_curriculum
    # =False, domain_randomization=False, reset noise=0, max_consecutive_successes=#goals, and the
    # startArmHigher arm pose (arm[1]-=10deg, arm[3]+=10deg).
    demo_mode = False
    demo_z_offset = 0.03  # eval_interactive.py: traj_data["start_pose"][2] += Z_OFFSET

    # --- friction (original modifyAssetFrictions; applied when pretrained_compat) ---
    # The pretrained policy was trained with HIGH fingertip friction (grip) and low elsewhere.
    # Without this the hand contacts the object but can't hold/lift it (grasp slips) -> unreliable.
    apply_compat_friction = True
    fingertip_friction = 1.5  # the 5 *_DP distal links (self.fingertips in the original)
    robot_friction = 0.5      # all other robot links
    object_friction = 0.5
    table_friction = 0.5
    delta_goal_distance = 0.1
    delta_rotation_degrees = 90.0
    # Goal target volume (env-local), elevated above the table top (~0.53) so the FIRST goal of
    # each episode forces lifting (matches the original SimToolReal targetVolumeMins/Maxs).
    target_volume_min = (-0.35, -0.2, 0.60)
    target_volume_max = (0.35, 0.2, 0.95)

    # --- domain randomization (default ON for training; scripts disable it for eval) ---
    domain_randomization = True
    randomize_object_yaw = True            # random object yaw at reset
    object_scale_noise_range = (0.9, 1.1)  # per-env keypoint/scale multiplier (objectScaleNoiseMultiplierRange)
    perturb_force_scale = 20.0             # random force on the object when lifted (forceScale)
    perturb_torque_scale = 2.0             # random torque (torqueScale)
    perturb_prob = 0.25                    # per-control-step probability of a perturbation kick
    obs_noise_std = 0.01                   # Gaussian observation noise (std, on the policy obs)
    randomize_pd_gains = True              # per-env actuator stiffness/damping jitter
    pd_gain_noise = 0.2                    # +/- fraction on default gains

    # --- tolerance curriculum (anneal success tolerance as the policy improves) ---
    use_tolerance_curriculum = True
    target_success_tolerance = 0.01            # tightest tolerance (targetSuccessTolerance)
    tolerance_curriculum_increment = 0.9       # multiplicative tighten (toleranceCurriculumIncrement)
    tolerance_curriculum_interval = 3000       # control steps between checks (original frame_since_restart units)
    curriculum_success_threshold = 3.0         # advance when mean successes/episode >= this (original hardcodes 3.0)
