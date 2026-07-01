"""Config for 'screwdriver_aligned' — the 044 flat-slot task with a STABLE physical screw.

Same task as ScrewdriverEnvCfg (044 flat screwdriver + tighten_traj), but the passive screw is the
PCA-aligned + SDF-baked flat screw on a clean vertical revolute joint, so the physics viz
(--physical_screw) doesn't blow up. The original convexDecomposition screw filled the slot (blade
jammed -> NaN on the rotate phase) and its mesh origin was off the shaft axis. Here:
  - shaft is along +z, slot along +x (so the generator's slot=world-x assumption holds), re-centered
    on the shaft axis (revolute axis = +z through the screw origin -> trivial joint);
  - SDF collider preserves the ~10mm slot so the blade enters it (form closure drives the screw);
  - head_offset points to the slot opening; a gentle NEGATIVE clearance seats the blade into the slot.
Built by: scripts/build_screw_asm_flat.py (assembly) + convert_mesh.py (SDF). Does NOT modify the
base 044 task.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import ASSETS
from .screwdriver_env_cfg import ScrewdriverEnvCfg

SCREW_ALIGNED_SDF_USD = f"{ASSETS}/flat_screw_aligned_sdf/flat_screw_aligned_sdf.usd"
SCREW_ASM_FLAT_USD = f"{ASSETS}/screw_assembly_flat_sdf/screw_assembly_flat_sdf.usd"

# aligned flat screw: origin = shaft-axis centroid, shaft +z, slot +x. World placement matches the
# old (correctly-seated) screw: centroid = old_rot @ (0.013*mesh_centroid) + old_pos.
_SCREW_POS = (0.0793, 0.0, 0.5648)


@configclass
class ScrewdriverAlignedEnvCfg(ScrewdriverEnvCfg):
    # aligned + SDF flat screw (kinematic path; the physical path uses screw_asm_cfg below)
    screw_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_ALIGNED_SDF_USD,
            scale=(0.013, 0.013, 0.013),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_SCREW_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # head/slot offset above the screw root (= centroid): slot OPENING at +33.4mm (shaft +z).
    screw_head_offset_nominal: tuple = (0.0, 0.0, 0.0334)
    # the slot is ~10mm deep; seat the blade gently INTO it (negative = into the slot, +z up).
    screw_contact_clearance: float = -0.005

    # added joint inertia: smooths the screw's response to the kinematic blade's stiff SDF contact
    # (an unbounded-force teleport otherwise flings the narrow-slot screw -> NaN on the rotate phase).
    screw_joint_armature: float = 0.005

    # stable physical screw: aligned-SDF flat screw on a vertical revolute joint (base = thread_test).
    # CAP the screw's response (max_angular_velocity + max_depenetration_velocity) + more solver iters,
    # so the stiff kinematic-blade contact can't fling it to NaN.
    screw_asm_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/ScrewAsm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_ASM_FLAT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_angular_velocity=15.0, max_depenetration_velocity=0.5,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0475, 0.0, 0.53), joint_pos={"screw_spin": 0.0}),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["screw_spin"], stiffness=0.0, damping=0.0)},
    )
