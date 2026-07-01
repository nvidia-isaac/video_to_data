"""Config for the SimToolReal 'screwdriver' env (IIWA14 + left Sharpa).

Mirrors `SimToolRealEnvCfg`: the manipulated TOOL is the `044_screwdriver` (replacing the
claw_hammer), and a `flat_screw` is added as a PASSIVE rigid body resting on the table
(loaded + reset each episode, but NOT part of the observation / reward / goal logic). The
robot, table, reward, observations, actions, and goal sampling are all inherited unchanged.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS, SimToolRealEnvCfg

# DEFAULT = the SDF-collider screwdriver: its thin blade physically ENTERS the slot (the convex-decomp
# 044_screwdriver.usd has a blunt blade that can't, so it just shoves the head). Required for real
# slot engagement; this is the collider that worked in best10_screwdriver_6_14.
SCREWDRIVER_USD = f"{ASSETS}/044_screwdriver_sdf/044_screwdriver_sdf.usd"
SCREW_USD = f"{ASSETS}/flat_screw/flat_screw.usd"
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"
SCREW_ASM_USD = f"{ASSETS}/screw_assembly/screw_assembly.usd"  # thread_test base + revolute-jointed screw


@configclass
class ScrewdriverEnvCfg(SimToolRealEnvCfg):
    # --- pluggable goal generation (modular: swap per env/task) ---
    # open-loop tighten-goal generator MODULE; must export TOOL, BLADE, TIP, T, compute_goals_batch
    # (same interface as tighten_traj). Override per env to drive a different trajectory. The
    # responsive closed-loop generator is selected separately via `responsive_goals`.
    goal_generator_module: str = "simtoolreal_lab.tasks.screwdriver.tighten_traj"
    # screwdriver-like keypoint box the policy sees in its obs (overrides the base claw-hammer default).
    # Matches what deploy/viz use, so the finetune trains on the SAME observation as eval.
    pretrained_object_scale: tuple = (2.5, 0.75, 0.75)
    # generate + advance the per-env tighten goals during TRAINING (not just demo_mode) so a policy
    # can be finetuned to FOLLOW the trajectory. Pair with randomize_layout=True (goals built per layout).
    use_tighten_goals: bool = False
    # pluggable goal-pose NOISE schedule (modular; default off -> clean goals for eval/viz). Module
    # exports sigma_schedule(T, phase_counts) -> (pos_sigma (T,), rot_sigma (T,)); the env adds fresh
    # per-env N(0,sigma) noise to each reset's goal poses. Set by finetune for training DIVERSITY.
    goal_noise_module: str | None = None
    # global multiplier on the goal-noise schedule's per-waypoint sigma (1.0 = the module's base
    # magnitudes; <1 = milder, >1 = wider). Lets you dial coverage vs success-yield without editing
    # the schedule module. Only used when goal_noise_module is set.
    goal_noise_scale: float = 1.0

    # --- goal-TRAJECTORY diversification (training; off by default -> clean goals for eval/viz). Unlike
    #     goal_noise (independent per-waypoint jitter), this diversifies the trajectory SHAPE + PATH:
    #     (a) per-episode random GENERATION parameters (the generator's sample_diversify_params, e.g. the
    #     hammer's lift_height/swing_angle/n_strikes) -> coherently different shapes, and (b) a SMOOTH
    #     correlated offset added to the APPROACH phases (decays to 0 by the strike/insert so it stays
    #     clean) -> a varied path. Both regenerated fresh each reset. Success-filtering keeps the good ones.
    goal_diversify: bool = False
    goal_diversify_scale: float = 1.0          # global multiplier on the param-noise ranges + offset magnitude
    goal_diversify_offset_std: float = 0.03    # std (m) of the smooth approach-path offset control points

    # spawn the passive screw + thread_test? Default True (eval/viz). Finetune sets False: the reward
    # is pure tool pose-reaching and the goals only need screw_head_world (a computed point), so the
    # screw/fixture are unnecessary -> drop them for speed + clean tracking. The screwdriver (Object)
    # and its SDF collider are UNCHANGED either way, so the tool's own collision matches train<->eval.
    spawn_passive_screw: bool = True

    # FREEZE the manipulated object at its reset pose until the hand reaches it, then release to full
    # UNMODIFIED dynamics. Settles the resting SDF-collider-on-table jitter (PhysX can't sleep it on the
    # kinematic table; damping can't drain the self-sustaining contact) without touching post-contact
    # physics. Release latches once a fingertip is within grasp_release_dist of the object.
    freeze_until_grasp: bool = True
    grasp_release_dist: float = 0.06    # m: a fingertip this close -> release (then dynamics are untouched)

    # --- manipulated tool = the screwdriver (replaces claw_hammer; drives all obs/reward/goals) ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREWDRIVER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
                # NOTE: the resting jitter is a self-sustaining SDF-collider-on-table contact instability;
                # PhysX sleep can't engage (the table is KINEMATIC -> a body touching it stays awake) and
                # damping can't drain it (the contact re-injects energy each step). The reliable fix is to
                # FREEZE the object at its rest pose until the hand reaches it (see freeze_until_grasp).
                sleep_threshold=0.05,
                stabilization_threshold=0.02,
            ),
        ),
        # centered under the hand, spawned RESTING on the table: the 044 screwdriver settles at
        # root z ~0.547 on the table top (0.53); spawn 1 mm above so it starts on the table with
        # no drop (was 0.60, which dropped ~7 cm).
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.548)),
    )

    # --- passive screw (free rigid body resting on the table, offset from the tool) ---
    # Loaded + reset each episode for visual/physical presence only. NOT in obs/reward/goals
    # (per "other things are unchanged"); the screwdriver remains the single manipulated object.
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_USD,
            # The flat_screw mesh has NO real-world scale (raw bbox ~2.3 x 3.8 x 3.8 m), so
            # scale it down to a realistic screw (~5 cm longest dim: 3.81 m * 0.013 ~ 0.05 m).
            scale=(0.013, 0.013, 0.013),
            # Kinematic: the screw is posed INSERTED in a thread_test hole. The bar's
            # convex-decomposition collision fills the holes, so a dynamic screw would be
            # ejected; kinematic holds the inserted pose.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        # Inserted into the SMALLEST hole the screw fits: the ~1.88 cm-diameter hole at
        # mesh-x~6 (world ~(0.078, 0)). The screw's ~1.46 cm shaft clears it (the 1.42 cm
        # hole is too small); its wider ~3.1 cm head seats on the rim. Shaft points down,
        # head up. pos is the mesh origin (offset from the geometry centroid); rot (wxyz)
        # stands the shaft vertical.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0697, 0.0235, 0.555),
            rot=(0.3852, 0.9223, -0.0306, 0.0),
        ),
    )

    # --- thread_test fixture (kinematic base; the screw rests on top of it) ---
    # Raw mesh bbox is ~63 x 16 x 10 mm (millimeters) with its origin at the bottom face
    # (min z = 0) and offset in +x (x in [-5, 58] mm). Scale x0.005 (5x larger than the
    # original x0.001) -> ~31.5 x 8 x 5 cm. Placed so the bar is centered at x=0.18
    # (bbox-center offset 26.5 mm * 0.005 = 0.1325 m) and its base sits on the table top
    # (z=0.53). Kinematic so it stays put as a fixture under the screw.
    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD,
            scale=(0.005, 0.005, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53)),
    )

    # head of the screw relative to its root, in WORLD at the NOMINAL screw orientation (from the
    # screw-mesh PCA + the env's nominal screw pose). Used to target the screwdriver tip at the
    # screw head per env: head_world = screw_pos + Rz(yaw) @ this (yaw = the layout group yaw).
    # head of the screw relative to its root, in WORLD at the NOMINAL screw orientation. Used to target
    # the screwdriver tip at the screw head per env: head_world = screw_pos + Rz(yaw) @ this.
    screw_head_offset_nominal: tuple = (0.0072, -0.0238, 0.0436)

    # clearance (m) of the screwdriver TIP relative to the screw head along the screw axis (+z) at
    # the contact/rotate phase. Negative = the tip seats slightly INTO the head so it actually
    # touches the screw (vs hovering above). 0.004 hovered ~1.5 cm high, so -0.011 lowers it to contact.
    screw_contact_clearance: float = -0.011

    # --- driven screw (the screw turns + sinks as the screwdriver tightens it) ---
    # Kinematic coupling: when the screwdriver tip is engaged in the slot (tip near the head +
    # tip pointing down), the (kinematic) screw is rotated about its axis to TRACK the
    # screwdriver's rotation, and sunk into the hole by the thread pitch. This shows the screw
    # turning without fragile contact-torque physics (the thin blade barely seats + the policy
    # only hovers within tolerance). Default off (screw stays frozen as before).
    screw_turns_with_driver: bool = False
    screw_thread_pitch: float = 0.001      # m of axial sink per full (2*pi) turn
    screw_engage_radius: float = 0.008     # tip-to-head distance (m) for "tip in slot". Tightened
    #                                        0.03 -> 0.015 -> 0.008: successes reach 2-3 mm, so 8 mm means
    #                                        the blade is genuinely DEEP in the slot, not hovering at the rim
    screw_engage_tipdown: float = 0.8      # require tool-axis . (-z) > this (tip pointing steeply down;
    #                                        tightened from 0.6 ~ within ~37deg of vertical)
    # --- SLOT bounding box (`_tip_in_slot`): the tip is "in the slot" if it falls inside an oriented box
    # centered at the screw head -- long axis ALONG the slot (turns with the screw), narrow ACROSS it, and
    # a depth range BELOW the head top. Computed on the fly each step; more precise than a sphere. ---
    # sized to the engagement envelope during TURNING (the tip rides ~8mm off the head ref while turning,
    # vs ~2-4mm at deepest insertion -- the box must accept the turning pose, not just insertion). A clear
    # miss is ~22/59mm out, so these still reject disengaged tips:
    slot_half_length: float = 0.015        # half the slot's long extent (along), m
    slot_half_width: float = 0.005         # half the slot width (across), m  (engaged turning rides ~7mm)
    slot_depth: float = 0.009              # max depth below the head ref the tip may be, m
    slot_top_tol: float = 0.002            # max height above the head ref the tip may be, m
    screw_max_sink: float = 0.012          # cap total sink (m)

    # --- PHYSICAL screw (merge screw + thread_test into one articulation; screw spins via contact) ---
    # When True, the kinematic screw + thread_test RigidObjects are replaced by a single
    # ArticulationCfg: thread_test = FIXED base, screw = revolute-jointed link (axis = its shaft,
    # SDF collider) that the screwdriver blade turns via contact friction. Supersedes the kinematic
    # `screw_turns_with_driver` coupling. DEFAULT ON: the screwdriver task drives + measures the real
    # spinning screw (the working config; the kinematic path is opt-out via physical_screw=False).
    physical_screw: bool = True
    # screw_spin joint resistance (so the screw doesn't free-spin/coast like a frictionless bearing):
    #   damping  = velocity resistance (tau = -damping * omega; stops coasting when the blade lifts off)
    #   friction = static/Coulomb resistance (needs torque to overcome; mimics thread friction)
    #   armature = added joint inertia (smooths). Contact torque is tiny, so keep these small.
    screw_joint_damping: float = 0.005     # bigger: heavily damped, no coasting (blade still turns it)
    screw_joint_friction: float = 0.005    # static/Coulomb (thread-friction feel)
    screw_joint_armature: float = 0.0
    # The episode TERMINATES (success) once the screw has rotated CLOCKWISE-from-top by >= this many
    # RADIANS from its start WITH the tip in the slot. DEFAULT = 150 deg (2.618 rad). Requires
    # physical_screw=True. Set None to disable the terminal/success condition.
    terminate_on_screw_rotated: float | None = math.radians(150.0)
    # sign that maps the screw_spin joint delta to CLOCKWISE-from-top rotation. CW-from-top is NEGATIVE
    # about +z (right-hand rule), so -1.0; success counts only clockwise (tightening) rotation.
    screw_tighten_sign: float = -1.0
    # total tool rotation (DEGREES) in the tighten trajectory's TURN phase. 180 = the original (6_14)
    # trajectory; with the SDF blade actually in the slot the screw follows it well. Passed to
    # the goal generator's compute_goals_batch(turn_degrees=...).
    tighten_turn_degrees: float = 180.0

    # --- responsive (closed-loop) goal generator ---
    # When True, the goal pose is recomputed EACH control step from the CURRENT state instead of
    # advancing through the fixed `tighten_traj` trajectory (which stays as the backup, used when
    # this is False): the goal follows the screw's CURRENT rotation (slot tracks the screw angle),
    # and switches behavior by whether the tip is in the slot:
    #   tip OUT of slot -> guide it back over the slot + lower (re-insert from the top)
    #   tip IN  slot     -> rotate the blade ahead of the screw angle (drive the screw round)
    responsive_goals: bool = False
    resp_engage_radius: float = 0.025   # tip-to-head dist (m) below which the tip counts as "in the slot"
    resp_over_radius: float = 0.035     # tip HORIZONTAL (xy) dist to slot below which -> lower in; beyond -> go over
    resp_tipdown: float = 0.6           # require tool-axis . (-z) > this to count as engaged
    resp_rotate_step: float = 0.5       # when engaged, target blade = screw_angle + this (rad) -> turns the screw
    resp_approach_height: float = 0.08  # "over the slot" height above the head (m)
    # the goal is a CARROT: always a small step ahead of the CURRENT screwdriver pose toward the
    # responsive target (so it stays in the policy's in-distribution incremental-tracking range,
    # rather than jumping to the far final pose, which the zero-shot policy can't track).
    resp_pos_hop: float = 0.08          # max position lead ahead of the current tool root (m)
    resp_rot_alpha: float = 0.30        # orientation lead toward the target (nlerp fraction)
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(usd_path=SCREW_ASM_USD),
        # root = the thread_test base; placed at the nominal thread_test pose (overridden per reset).
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"screw_spin": 0.0}),
        # free-spinning joint (no position drive): the blade contact provides the torque.
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["screw_spin"], stiffness=0.0, damping=0.0)},
    )

    # --- layout randomization (applied every reset) ---
    # Randomizes the xy-plane pose (x, y, yaw) of: (a) the thread_test + screw as ONE rigid
    # group (so the screw stays inserted in its hole), and (b) the screwdriver tool. The tool
    # is rejection-sampled so its footprint never overlaps the thread_test bar.
    randomize_layout: bool = True
    layout_pivot_xy: tuple = (0.18, 0.0)                     # nominal bar center = rotation pivot
    layout_threadtest_half_extents: tuple = (0.1575, 0.04)   # bar OBB half-size in xy (~31.5 x 8 cm)
    # Tight ranges (per request): both the tool and the screw+thread_test group were shifted 0.15 to
    # the robot's RIGHT (-x; robot faces -y so +x is its LEFT, -x its RIGHT) -- the whole scene
    # translates together, preserving their relative layout + no-overlap clearance. Tool sampled
    # +/-0.1 around (-0.15, 0); screw group around bar center ~0.15 (-x) and forward in +y (nearer
    # the robot). Both get +/-35 deg yaw.
    layout_threadtest_center_x_range: tuple = (0.05, 0.25)   # shifted RIGHT (-x) 0.15 from 0.3 +/- 0.1
    # screw+thread_test group y-center: pulled back to ~centered so the long bar stays ON the
    # narrowed (0.5-deep) table under +/-35 deg yaw -- the bar spans ~cy +/-0.20 in y, and the table
    # half-depth is 0.25, so |cy| must stay <= ~0.05. (Earlier this was pushed forward to (0.10,0.25)
    # to sit nearer the robot, but that overhangs the table edge once the table is narrowed.)
    layout_threadtest_center_y_range: tuple = (-0.05, 0.05)
    layout_screwdriver_x_range: tuple = (-0.25, -0.05)       # shifted RIGHT (-x) 0.15 from 0 +/- 0.1
    layout_screwdriver_y_range: tuple = (-0.10, 0.10)        # tool y-range (hammer overrides the near-robot end)
    layout_yaw_range: tuple = (-0.6108652381980153, 0.6108652381980153)  # +/-35 deg
    layout_min_clearance: float = 0.15   # tool footprint radius (0.127) + margin (no-overlap)
    layout_max_reject_iters: int = 40

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # the SDF screwdriver collider (default) generates many more contacts than a convex hull, so
        # the default GPU collision stack overflows ("Contacts have been dropped"); bump it.
        self.sim.physx.gpu_collision_stack_size = 2 ** 28
        # default look (match the hammer task): dark-grey floor/backdrop + light-blue screw
        self.ground_color = (0.12, 0.12, 0.12)
        self.screw_color = (0.55, 0.72, 0.82)
