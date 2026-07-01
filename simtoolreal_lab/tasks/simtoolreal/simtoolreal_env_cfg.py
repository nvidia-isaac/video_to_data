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

    # --- goal-free teacher (distillation-friendly actor) ---
    # When True, the ACTOR obs drops keypoints_rel_goal(12) and instead gets the dynamic SCREW
    # keypoints rel-palm(12): the actor must INFER the goal from the layout (the screw pose), so it
    # observes only what the BC student can. The CRITIC keeps the goal (added back as privileged) -> a
    # teacher cfg using this MUST set state_space = obs(140) + goal(12) + privileged(15) = 167.
    actor_infer_goal_from_screw: bool = False

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
    cam_z_far: float = 50.0                  # per-env camera far clip (m). Lower it (+ raise env_spacing)
    #                                          so the view clips out neighbor sub-envs (data collection).
    cam_focal: float = 24.0                  # per-env camera focal length (mm). Larger = zoomed-in / narrower FOV.
    # optional WRIST camera: a TiledCamera mounted on PALM_BODY (iiwa14_link_7) looking at the palm/
    # fingers (moves with the hand). eye/lookat are in the LINK-LOCAL frame; calibrate by rendering.
    wrist_camera: bool = False
    wrist_cam_width: int = 640
    wrist_cam_height: int = 480
    # Calibrated wrist cam (SIDE view): ~8 cm off to the +x side of the wrist base (just clears the
    # bulky palm-base mesh), aimed at the grasp region (fingers gripping the object). Shows the
    # fingers wrapping the tool + contact, arm out of frame. (Hand geometry probe: fingers at z~+0.27,
    # thumb at (-0.096,-0.042,0.185), palm normal ~-y.)
    wrist_cam_eye: tuple = (0.08, -0.02, 0.08)       # link-local camera position (side of the wrist)
    wrist_cam_lookat: tuple = (-0.02, -0.015, 0.18)  # link-local target = grasp region
    wrist_cam_up: str = "Y"
    wrist_cam_focal: float = 14.0
    # if set, spawn a TEXTURED floor (this MDL material) instead of the default grid -- changes the
    # camera background (e.g. an in-house Isaac Lab tile/carpet material). None = default grid.
    ground_mdl: str | None = None
    ground_texture_scale: tuple = (1.0, 1.0)
    # if set, the floor + backdrop use a plain solid diffuse color (e.g. white) instead of the grid /
    # an MDL texture -- a clean uniform background. Takes precedence over ground_mdl. None = off.
    ground_color: tuple | None = None
    # if set, override the work-table's diffuse color (default mid-grey). Use to match the table to a
    # solid ground_color for a uniform scene. None = keep the table_cfg default.
    table_color: tuple | None = None
    # if set, recolor the physical screw/nail link (the driven part of screw_asm, not the board) with
    # this diffuse color, overriding the baked USD material. None = keep the asset's own color.
    screw_color: tuple | None = None
    # if set, the environment DomeLight uses this HDRI as its texture (an existing Isaac Lab/Sim sky
    # asset). This lights the scene AND renders as the camera background (replaces the gray dome / the
    # backdrop wall). None = plain gray dome. dome_intensity scales the HDRI's brightness.
    dome_texture: str | None = None
    dome_intensity: float = 2000.0

    # --- robot identifier overrides (default None -> the module-level IIWA14 + left-Sharpa
    # constants JOINT_NAMES_ISAACGYM / PALM_BODY / FINGERTIP_BODIES / PALM_OFFSET / FINGERTIP_OFFSET).
    # A robot-SWAP task (e.g. the Vega tasks) sets these so the SAME obs/action/reward/goal pipeline
    # drives a different arm + the (un-prefixed) left Sharpa hand. Left None, the original tasks are
    # byte-identical -- the env falls back to the module constants. ---
    joint_names: list | None = None        # 29 canonical joints, arm 0:7 then hand 7:29
    palm_body: str | None = None           # palm/wrist body (palm_pos/palm_rot + wrist-cam mount)
    fingertip_bodies: list | None = None   # 5 fingertip bodies (index/middle/ring/thumb/pinky *_DP)
    palm_offset: tuple | None = None       # palm-center offset in the palm-body frame (m)
    fingertip_offset: tuple | None = None  # fingertip-center offset in the fingertip-body frame (m)

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

    # move the WORK-TABLE this far (m) FURTHER from the robot (the robot is at +y, so the table slides
    # -y). The objects (hammer/screw/...) are NOT moved -> they stay where the robot reaches; the table
    # just slides under them away from the robot. The env CAPS this so >=10cm of table stays under the
    # objects (else they'd fall off the receding near edge). 0 = default position.
    table_dist: float = 0.0

    # --- static table (kinematic cuboid; top surface at z=0.53) ---
    # y (depth) narrowed 0.6 -> 0.5 toward the original simtoolreal table (0.4): the near edge moves
    # from y=0.3 to y=0.25 (away from the robot at y=0.8), giving the arm more clearance so it clips
    # the table less. x kept wide (0.6) for the long thread_test bar. Center/top unchanged (0,0,0.38 /
    # 0.53) so the robot->table offset still matches the original.
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.5, 0.3),
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

    # --- pluggable reward augmentation (modular; default off) ---
    # Module exporting augment_reward(env) -> (reward_add (N,), success_gate (N,) bool). The env ADDS
    # reward_add to the pre-shaper reward and AND-s success_gate into the keypoint near-goal test. Used
    # to add task-specific shaping/precision (e.g. the cross-slot TIP tolerance) without editing the env.
    reward_module: str | None = None
    # score keypoint success on the FIXED-SIZE keypoints even when NOT demo_mode (matches the original
    # fixedSizeKeypointReward=True, used in training AND eval). Set by finetune so train==eval metric.
    fixed_size_success: bool = False
    # anneal the reward-module's TIP tolerance (TIP_TOL_START -> TIP_TOL_TARGET) on its own success-gated
    # curriculum, INDEPENDENT of the keypoint tolerance curriculum (which the finetune fixes at 0.01).
    tip_tol_curriculum: bool = False

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
    # The original SimToolReal RL TRAINING (Isaac Gym modifyAssetFrictions) also used the high fingertip
    # friction -- but the Isaac Lab port only applied it under pretrained_compat. Set this True to apply
    # the same grip friction when training from scratch (compat OFF). Default False keeps every existing
    # task's behavior unchanged (originals never set it).
    force_grasp_friction = False
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
    # per-env perturbation trigger probability: sampled LOG-UNIFORM in this range at each reset and
    # rolled independently for force vs torque each control step (original random_force_prob /
    # random_torque_prob = log_uniform(forceProbRange)). The force/torque are mass-scaled (randn *
    # object_mass * scale), i.e. a fixed-acceleration kick, faithful to the Isaac Gym env.
    perturb_prob_range = (0.001, 0.1)      # forceProbRange / torqueProbRange
    # enable the force/torque perturbation independently of full domain_randomization (so eval / BC
    # data collection can opt into perturbations only). Perturbation fires when this OR DR is on.
    force_perturbation = False
    # --- tool-displacement perturbation: directly TELEPORT the tool by a random delta pose (only once
    #     lifted, respecting a cooldown) + zero its velocity -> the tool slips / falls out of the grasp.
    #     Simulates failure cases (tool drops, grasp slips); the expert must recover (re-grasp / re-
    #     position), and success-filtering keeps the RECOVERED episodes (DART/DAgger-style failure data).
    #     Much more disruptive than force kicks -> expect a larger yield drop. Off by default. ---
    tool_displacement = False
    tool_displace_prob = 0.004         # per-env per-step probability of a teleport (gated on lifted)
    # Per-teleport magnitude is sampled HALF-NORMAL over [min, max] = min + |N(0,(max-min)/2)| clamped to max
    # -> mode at min, decaying tail -> MORE small slips than big drops, with a RANDOM direction/axis. Calibrated by a 3/5/8cm x 800/1500 sweep:
    # recoverability is gated by EPISODE LENGTH (at 800 steps recoveries get cut off mid-recovery; >=1500
    # steps recovers ~2x more). With max_ep_steps>=1500, the [2,10]cm range (mean ~6cm, near the 5cm sweet
    # spot) keeps recovery high while spanning gentle-to-severe failures. Override via --tool_displace_*.
    tool_displace_pos_min = 0.02       # min position offset per teleport (m): small slips
    tool_displace_pos = 0.10           # max position offset per teleport (m): big drops
    tool_displace_rot_min = 0.10       # min rotation per teleport (rad, ~6 deg)
    tool_displace_rot = 0.50           # max rotation per teleport (rad, ~29 deg)
    tool_displace_cooldown = 60        # min control steps between teleports (let the expert recover)
    # also teleport BEFORE the grasp (default only-when-lifted). True -> fire any time the tool is on the
    # table too, simulating "tool isn't where expected / failed grasp" (the expert re-approaches). NOTE:
    # if freeze_until_grasp is on (screwdriver), pre-grasp teleports are overridden by the freeze -> use
    # on the hammer (freeze off). Off by default.
    tool_displace_pregrasp = False
    # reject task success that occurs within this many control steps AFTER a teleport (60 = 1s @ 60Hz): a
    # teleport can accidentally drop the tool onto the screw/goal pose and fake a "success", so a seating
    # this soon after a teleport is treated as a FAILURE instead. 0 = no such guard.
    tool_displace_success_block_steps = 60
    # --- joint-displacement perturbation: TELEPORT the robot's 29 arm+hand joint positions by a random
    #     per-joint delta (+ zero joint velocity), INDEPENDENTLY sampled from the tool teleport (its own
    #     prob / cooldown / magnitude). Simulates a control glitch / the robot getting bumped; the PD +
    #     expert recover. cur/prev TARGETS are left unchanged -> the recorded action stays the clean expert
    #     command (the jump is a state disturbance, not an action), and it feeds the SAME per-step teleport
    #     flag as the tool teleport (so the chunk-loss masking covers it). Fires any time (no lifted gate).
    #     Off by default; enable via collect_bc_data.py --joint_displacement. ---
    joint_displacement = False
    joint_displace_prob = 0.004        # per-env per-step probability of a joint teleport (independent)
    # SEPARATE arm vs hand magnitudes: an arm joint moves the end-effector ~10x more per radian than a
    # finger joint, so the ARM gets a SMALLER delta. Each is a per-event HALF-NORMAL scale over its own
    # [min,max] (= min+|N(0,(max-min)/2)| clamped -> more small jolts than big); per-joint delta = randn*scale.
    joint_displace_arm_scale_min = 0.02    # arm (7 joints): min per-joint delta std (rad, ~1 deg)
    joint_displace_arm_scale = 0.10        # arm (7 joints): max per-joint delta std (rad, ~6 deg) -- small (big EE move)
    joint_displace_hand_scale_min = 0.05   # hand (22 joints): min per-joint delta std (rad, ~3 deg)
    joint_displace_hand_scale = 0.30       # hand (22 joints): max per-joint delta std (rad, ~17 deg) -- larger
    joint_displace_cooldown = 60       # min control steps between joint teleports (let the expert recover)

    # --- random-action-burst perturbation (DART/DAgger): with random_action_prob/step the robot EXECUTES
    #     random delta actions for a BURST of N control steps, driving it off-distribution; the RECORDED
    #     action stays the expert correction + the burst is flagged for chunk-loss masking, and the expert
    #     re-predicts + recovers after. Off by default; enable via collect_bc_data.py --random_action. ---
    random_action = False
    random_action_prob = 0.007         # per-env per-step probability of STARTING a random-action burst (NO cooldown)
    random_action_std = 0.5            # std of the random 29-dim delta action (action space ~[-1,1])
    random_action_steps_std = 27.0     # burst length N = round(|N(0, this^2)|) control steps (mean ~21 = 0.36s @60Hz)
    obs_noise_std = 0.01                   # Gaussian observation noise (std, on the policy obs)
    randomize_pd_gains = True              # per-env actuator stiffness/damping jitter
    pd_gain_noise = 0.2                    # +/- fraction on default gains

    # --- tolerance curriculum (anneal success tolerance as the policy improves) ---
    use_tolerance_curriculum = True
    target_success_tolerance = 0.01            # tightest tolerance (targetSuccessTolerance)
    tolerance_curriculum_increment = 0.9       # multiplicative tighten (toleranceCurriculumIncrement)
    tolerance_curriculum_interval = 3000       # control steps between checks (original frame_since_restart units)
    curriculum_success_threshold = 3.0         # advance when mean successes/episode >= this (original hardcodes 3.0)
