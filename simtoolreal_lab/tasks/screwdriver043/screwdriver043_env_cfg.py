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
SCREW_NEW_USD = f"{ASSETS}/screw_new_sdf/screw_new_sdf.usd"
THREAD_TEST_USD = f"{ASSETS}/thread_test/thread_test.usd"
# physical articulation: 50% thread_test FIXED base + cross-slot screw on a revolute joint (SDF)
SCREW_ASM_043_USD = f"{ASSETS}/screw_assembly043/screw_assembly043.usd"


@configclass
class Screwdriver043EnvCfg(ScrewdriverEnvCfg):
    # tool = 043 (phillips) screwdriver  (mirror the 044 object_cfg, just swap the USD)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREWDRIVER_043_USD,
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
            scale=(0.00664, 0.00664, 0.00664),   # 60% smaller than the 0.0166 first guess
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        # this mesh's origin IS its shaft axis (xz-centered). The thread_test is shrunk 50% below,
        # which moves its hole (mesh-x~5.9) to world x = 0.0475 + 5.9*0.0025 = 0.0623 and lowers the
        # bar top to z = 0.53 + 10mm*0.0025 = 0.555. So pos.xy = the new hole (0.0623, 0), z puts the
        # head on the new bar top (head bottom @ 0.555); rot +90deg about x stands it head-up.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0623, 0.0, 0.5545),
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    # thread_test shrunk 50% (scale 0.005 -> 0.0025): bar ~15.75 x 4 x 2.5 cm; base still on the
    # table (z=0.53, mesh origin at the bottom face); bar top now z=0.555. The screw above tracks it.
    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD,
            scale=(0.0025, 0.0025, 0.0025),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53)),
    )

    # cross-slot geometry for the goal generator. The new screw's CROSS slot sits ~4.5mm above the
    # screw root (root @ z=0.5545, slot @ z~0.559); tip target a hair into the slot. (The inherited
    # flat-screw offset/clearance assume a much larger screw and would drive the tip below the bar.)
    screw_head_offset_nominal: tuple = (0.0, 0.0, 0.0045)   # head/slot rel. to screw root (slot @ z~0.559)
    # tip target = head + clearance; the cross slot is shallow (~1mm, bottom z=0.5588), so the tip must
    # seat IN the channels (tip target ~0.5584), NOT below them. Validated depth for real torque transfer.
    screw_contact_clearance: float = -0.0006

    # PHYSICAL screw articulation for --physical_screw (cross-slot screw revolute-jointed to the 50%
    # thread_test base). Same pattern as the 044 screw_asm_cfg, swapping in the 043 assembly.
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(usd_path=SCREW_ASM_043_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"screw_spin": 0.0}),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["screw_spin"], stiffness=0.0, damping=0.0)},
    )
