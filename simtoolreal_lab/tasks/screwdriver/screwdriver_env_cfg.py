"""Config for the SimToolReal 'screwdriver' env (IIWA14 + left Sharpa).

Mirrors `SimToolRealEnvCfg`: the manipulated TOOL is the `044_screwdriver` (replacing the
claw_hammer), and a `flat_screw` is added as a PASSIVE rigid body resting on the table
(loaded + reset each episode, but NOT part of the observation / reward / goal logic). The
robot, table, reward, observations, actions, and goal sampling are all inherited unchanged.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS, SimToolRealEnvCfg

SCREWDRIVER_USD = f"{ASSETS}/044_screwdriver/044_screwdriver.usd"
SCREW_USD = f"{ASSETS}/flat_screw/flat_screw.usd"
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"
SCREW_ASM_USD = f"{ASSETS}/screw_assembly/screw_assembly.usd"  # thread_test base + revolute-jointed screw


@configclass
class ScrewdriverEnvCfg(SimToolRealEnvCfg):
    # --- manipulated tool = the screwdriver (replaces claw_hammer; drives all obs/reward/goals) ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREWDRIVER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
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
    screw_engage_radius: float = 0.03      # tip-to-head distance (m) below which it engages
    screw_engage_tipdown: float = 0.6      # require tool-axis . (-z) > this (tip pointing down)
    screw_max_sink: float = 0.012          # cap total sink (m)

    # --- PHYSICAL screw (merge screw + thread_test into one articulation; screw spins via contact) ---
    # When True, the kinematic screw + thread_test RigidObjects are replaced by a single
    # ArticulationCfg: thread_test = FIXED base, screw = revolute-jointed link (axis = its shaft,
    # SDF collider) that the screwdriver blade turns via contact friction. Supersedes the kinematic
    # `screw_turns_with_driver` coupling. Default off (keeps the two kinematic RigidObjects).
    physical_screw: bool = False
    # screw_spin joint resistance (so the screw doesn't free-spin/coast like a frictionless bearing):
    #   damping  = velocity resistance (tau = -damping * omega; stops coasting when the blade lifts off)
    #   friction = static/Coulomb resistance (needs torque to overcome; mimics thread friction)
    #   armature = added joint inertia (smooths). Contact torque is tiny, so keep these small.
    screw_joint_damping: float = 0.005     # bigger: heavily damped, no coasting (blade still turns it)
    screw_joint_friction: float = 0.005    # static/Coulomb (thread-friction feel)
    screw_joint_armature: float = 0.0

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
    # Tight ranges (per request): tool sampled +/-0.1 around its demo pose (0,0); the screw +
    # thread_test group sampled +/-0.1 around an anchor 0.3 to the robot's LEFT (+x, robot faces
    # -y) of the tool's demo pose -> bar center (0.3, 0). Both get +/-35 deg yaw.
    layout_threadtest_center_x_range: tuple = (0.20, 0.40)   # 0.3 +/- 0.1
    layout_threadtest_center_y_range: tuple = (-0.10, 0.10)  # 0.0 +/- 0.1
    layout_screwdriver_x_range: tuple = (-0.10, 0.10)        # demo (0,0) +/- 0.1
    layout_screwdriver_y_range: tuple = (-0.10, 0.10)
    layout_yaw_range: tuple = (-0.6108652381980153, 0.6108652381980153)  # +/-35 deg
    layout_min_clearance: float = 0.15   # tool footprint radius (0.127) + margin (no-overlap)
    layout_max_reject_iters: int = 40
