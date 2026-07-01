"""Config for the SimToolReal 'screwdriver043' env — a SEPARATE variant of the screwdriver task.

Same env logic (reuses ScrewdriverEnv) but swaps the assets: the manipulated TOOL is the 043
(phillips) screwdriver, and the passive screw is a different one (converted from screw.obj),
resting in the SAME thread_test fixture. Kinematic screw, no trajectory/physics yet.

Does NOT modify the original screwdriver task — it only subclasses its cfg and reuses its env class.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS
from ..screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg

# PCA-aligned 043 (tool->+x, tip at x=0.134 -- matches the aligned-044 frame for zero-shot transfer),
# SDF collider so the thin Phillips cross tip is represented accurately + can enter the cross slot.
SCREWDRIVER_043_USD = f"{ASSETS}/043_screwdriver_aligned_sdf/043_screwdriver_aligned_sdf.usd"
# screw with an SDF collider so the CONCAVE cross slot is preserved (convexDecomp fills it; 'none'
# removes collision) -- lets the tip physically drop into the cross.
# DEFAULT = the 60%-WIDER cross slot (screw_new_wideslot_sdf): the original slot was already well-
# matched, but the wider slot + a slightly-lower tip seat (clearance -0.002 below) lets the perfect
# trajectory tighten to ~-166deg. (Original narrow slot: screw_new_sdf / screw_assembly043.)
SCREW_NEW_USD = f"{ASSETS}/screw_new_wideslot_sdf/screw_new_wideslot_sdf.usd"
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"
# physical articulation: 50% thread_test FIXED base + WIDER cross-slot screw on a revolute joint (SDF)
SCREW_ASM_043_USD = f"{ASSETS}/screw_assembly043_wideslot/screw_assembly043_wideslot.usd"


@configclass
class Screwdriver043EnvCfg(ScrewdriverEnvCfg):
    # cross-slot (Phillips) goal generator: minimal-rotation tip-down + nearest-arm (mod 90deg) roll snap
    goal_generator_module: str = "simtoolreal_lab.tasks.screwdriver043.tighten_traj043"

    # tool = 043 (phillips) screwdriver  (mirror the 044 object_cfg, just swap the USD)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREWDRIVER_043_USD,
            scale=(1.2, 1.2, 1.2),   # +20% bigger tip; screw/slot go +50% so the slot/tip ratio improves
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0, max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.548)),  # on the table, under the hand
    )

    # screw = converted screw.obj, kinematic, stood vertical (head up) in a thread_test hole.
    # SEPARATE object from thread_test (not combined); aligned to the hole the flat_screw used.
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_NEW_USD,
            scale=(0.012948, 0.012948, 0.012948),   # +50% bigger (was 0.008632) -> bigger slot, tip fits w/ clearance
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        # this mesh's origin IS its shaft axis (xz-centered). The thread_test (scale 0.004875) puts its
        # hole (mesh-x~5.9) at world x = 0.0475 + 5.9*0.004875 = 0.0763 and the bar top at
        # z = 0.53 + 10mm*0.004875 = 0.57875. pos.xy = the hole (0.0763, 0); z seats the head bottom
        # (mesh-y 0.07) on the bar top (0.5778 + 0.07*0.012948 = 0.57875); rot +90deg about x stands it up.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0763, 0.0, 0.5778),
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    # thread_test scale 0.004875 (50% then +30% then +50%): bar ~30 x 7.8 x 4.9 cm; base on the table
    # (z=0.53, mesh origin at the bottom face); bar top now z=0.57875. The screw above tracks it.
    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD,
            scale=(0.004875, 0.004875, 0.004875),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53)),
    )

    # cross-slot geometry for the goal generator. With the +50% screw, the CROSS slot sits ~9mm above
    # the screw root (root @ z=0.5778, slot @ z~0.587). Slot is now ~2mm deep + wider -> the (only +20%)
    # tip fits with clearance, so the first contact is gentle (no wedge/sudden snap).
    screw_head_offset_nominal: tuple = (0.0, 0.0, 0.0091)   # head/slot rel. to screw root (slot @ z~0.587)
    # tip target = head + clearance. The screw head is flat-ish out to r~5mm and the tip taper is wide,
    # so the tip first TOUCHES at the head top (z~0.5884); driving it deeper (the old -0.0012 -> tip_z
    # 0.5857) forced it ~2.3mm INTO the solid head -> the sudden-rotation-at-first-contact. Seat the
    # tip AT the slot level instead (tip_z ~0.5875): clearance +0.0006 (head 0.5869 + 0.0006).
    # seat the tip ~2.6mm LOWER than the slot level (was +0.0006): the 60%-wider slot's walls are out
    # of reach at the normal depth, but a little deeper the tip catches them (slot tapers). -0.002 gives
    # a clean ~-166deg tighten under the perfect trajectory; -0.003 a touch more; deeper over-seats.
    screw_contact_clearance: float = -0.002

    # PHYSICAL screw articulation for --physical_screw (cross-slot screw revolute-jointed to the 50%
    # thread_test base). Same pattern as the 044 screw_asm_cfg, swapping in the 043 assembly.
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(usd_path=SCREW_ASM_043_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"screw_spin": 0.0}),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["screw_spin"], stiffness=0.0, damping=0.0)},
    )
