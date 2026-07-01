"""Config for 'screwdriver043big' — the 043 cross-slot task with the screw + thread_test +80% bigger.

A SEPARATE experiment variant (does NOT modify the base screwdriver043 task/assets). Used to test
screw-drive physics validity: with the bigger screw + cross slot, replay (a) the correct tip-in-slot
tighten trajectory vs (b) an outer-rim-contact trajectory, and check the screw only spins for (a).

Everything is the base 043 cfg EXCEPT the screw + thread_test are scaled by f=1.8 (the screwdriver/
tool is UNCHANGED at 1.2x). The kinematic constants below are derived from the SAME f as the rebuilt
physical assembly (screw_assembly043_180.usd), so screw_head_world (the goal target) == the physical
slot. The base anchor (thread_test/assembly origin -> world (0.0475,0,0.53)) is unchanged; the screw
sits higher/further out because the scaled-up thread_test bar is taller/wider.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS
from .screwdriver043_env_cfg import SCREW_NEW_USD, THREAD_TEST_USD, Screwdriver043EnvCfg

F = 1.8  # +80% screw + thread_test
# +80% physical assembly (rebuilt by scripts/build_screw_asm043_big.py with the same F)
SCREW_ASM_043_BIG_USD = f"{ASSETS}/screw_assembly043_180/screw_assembly043_180.usd"

# base-anchored geometry, scaled by F (see screwdriver043_env_cfg for the unscaled derivation):
#   thread_test base on the table at (0.0475,0,0.53); screw root = base + (0.0288,0,0.0478)*F
_SCREW_POS = (0.0475 + 0.0288 * F, 0.0, 0.53 + 0.0478 * F)   # = (0.09934, 0, 0.61604)


@configclass
class Screwdriver043BigEnvCfg(Screwdriver043EnvCfg):
    # screw +80% (slot/head grow with it; the goal generator + clearance below track the bigger slot)
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_NEW_USD,
            scale=(0.012948 * F, 0.012948 * F, 0.012948 * F),   # 0.0233064
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_SCREW_POS,
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    # thread_test +80% (taller/wider bar; base stays anchored on the table at z=0.53)
    thread_test_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ThreadTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=THREAD_TEST_USD,
            scale=(0.004875 * F, 0.004875 * F, 0.004875 * F),   # 0.008775
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53)),
    )

    # head/slot offset above the screw root, and tip-vs-head clearance, both scale with the screw
    screw_head_offset_nominal: tuple = (0.0, 0.0, 0.0091 * F)   # = (0,0,0.01638) -> slot @ z~0.6324
    screw_contact_clearance: float = 0.0006 * F                 # = 0.00108

    # +80% physical revolute screw (rebuilt assembly; base anchored at (0.0475,0,0.53) like the base 043)
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(usd_path=SCREW_ASM_043_BIG_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"screw_spin": 0.0}),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["screw_spin"], stiffness=0.0, damping=0.0)},
    )
