"""SYSID-TUNED Vega arm robot config -- SEPARATE from vega_sharpa_robot.py (which is untouched).

The LEFT-arm stiffness/damping here are fit to the REAL harmonic-drive joints' chirp sysid
(logs/sysid/arm_sysid_amp_0.35_dur_60) so the IsaacSim implicit PD reproduces the measured
closed-loop response: a heavily-overdamped 2nd-order (wn~0.9-1.1 Hz, zeta~2.3-2.85) + ~40 ms delay,
i.e. a ~0.2 Hz -3dB tracking bandwidth -- MUCH softer/slower than the IK-tuned gains in
`vega_sharpa_robot.py` (_VEGA_ARM_STIFF=[900..180]).

Two pieces matter for matching the real arm:
  1. stiffness/damping (here) -> set the 2nd-order shape (wn, zeta).
  2. a COMMAND DELAY of `ARM_DELAY_STEPS` control steps -> the ImplicitActuator itself has no delay,
     so the env/action pipeline must buffer the arm position targets by this many steps (see
     `delay_steps_at` for rate conversion). Without it the sim leads the real arm by ~40 ms.

Values are loaded from logs/sysid/fit/tuned_gains.json (produced by scripts/tune_arm_sysid.py); the
baked SYSID_* dicts below are a snapshot fallback so the cfg is import-safe without that file.
Hand gains / right-arm / held groups are reused verbatim from vega_sharpa_robot.py.
"""

from __future__ import annotations

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import vega_sharpa_robot as _vsr
from .vega_sharpa_robot import (
    VEGA_ROBOT_USD, VEGA_BASE_POS, VEGA_ARM_JOINTS, VEGA_INIT_JOINT_POS, VEGA_LEFT_ARM, VEGA_RIGHT_ARM,
    VEGA_RIGHT_ARM_JOINTS,
    VEGA_HAND_STIFFNESS, VEGA_HAND_DAMPING, VEGA_HAND_ARMATURE, VEGA_HAND_FRICTION, VEGA_HAND_EFFORT,
    VEGA_ARM_EFFORT,
)

_TUNED_JSON = "/home/cning/simtoolreal_isaaclab/logs/sysid/fit/tuned_gains.json"

# --- baked snapshot fallback (FILLED from the tune_arm_sysid.py run) -------------------------------
# stiffness [N*m/rad], damping [N*m*s/rad] per L_arm joint; delay in control steps @ 100 Hz.
SYSID_ARM_STIFFNESS = {}   # filled below from JSON or snapshot
SYSID_ARM_DAMPING = {}
ARM_DELAY_STEPS = 0        # at SYSID_RATE_HZ
SYSID_RATE_HZ = 100.0

# snapshot constants from the tune_arm_sysid.py run (sim-vs-real VAF 97-99.9%; j7 set = j6 at load).
# stiffness [N*m/rad], damping [N*m*s/rad]; delay in 100 Hz steps. Used only if tuned_gains.json absent.
_SNAPSHOT = {
    "L_arm_j1": {"stiffness": 35.736, "damping": 29.608, "delay_steps": 4},
    "L_arm_j2": {"stiffness": 7.163, "damping": 5.951, "delay_steps": 4},
    "L_arm_j3": {"stiffness": 22.397, "damping": 18.497, "delay_steps": 4},
    "L_arm_j4": {"stiffness": 20.640, "damping": 17.016, "delay_steps": 4},
    "L_arm_j5": {"stiffness": 1.482, "damping": 1.239, "delay_steps": 3},
    "L_arm_j6": {"stiffness": 1.509, "damping": 1.261, "delay_steps": 4},
}


def _load_tuned():
    global SYSID_ARM_STIFFNESS, SYSID_ARM_DAMPING, ARM_DELAY_STEPS, SYSID_RATE_HZ
    src = None
    if os.path.exists(_TUNED_JSON):
        with open(_TUNED_JSON) as fh:
            blob = json.load(fh)
        src = blob["gains"]
        SYSID_RATE_HZ = float(blob.get("rate_hz", 100.0))
    elif _SNAPSHOT:
        src = _SNAPSHOT
    if not src:
        return False
    SYSID_ARM_STIFFNESS = {j: float(src[j]["stiffness"]) for j in src}
    SYSID_ARM_DAMPING = {j: float(src[j]["damping"]) for j in src}
    # j7 had NO sysid response (dead/no-feedback) -> use j6's gains for j7 (per request).
    if "L_arm_j7" not in SYSID_ARM_STIFFNESS and "L_arm_j6" in SYSID_ARM_STIFFNESS:
        SYSID_ARM_STIFFNESS["L_arm_j7"] = SYSID_ARM_STIFFNESS["L_arm_j6"]
        SYSID_ARM_DAMPING["L_arm_j7"] = SYSID_ARM_DAMPING["L_arm_j6"]
    # use the max per-joint delay as the (single) arm command delay
    ARM_DELAY_STEPS = max(int(src[j]["delay_steps"]) for j in src)
    return True


_load_tuned()


def delay_steps_at(control_hz: float) -> int:
    """Convert the sysid command delay (steps @ SYSID_RATE_HZ) to steps at the env control rate."""
    if not ARM_DELAY_STEPS:
        return 0
    return int(round(ARM_DELAY_STEPS / SYSID_RATE_HZ * control_hz))


def make_vega_robot_cfg_sysid() -> ArticulationCfg:
    """Vega + dual-Sharpa cfg with the SYSID-TUNED left-arm gains (left arm + left hand controlled;
    right side + body HELD). Drop-in replacement for make_vega_robot_cfg() for sim-to-real matching.
    NOTE: also apply a `delay_steps_at(control_hz)` command buffer on the arm targets (the implicit
    actuator has no built-in delay)."""
    if not SYSID_ARM_STIFFNESS:
        raise RuntimeError(
            f"no tuned gains: run scripts/tune_arm_sysid.py (expected {_TUNED_JSON}) or fill _SNAPSHOT")
    # any joints without a sysid fit (e.g. j7, no response) fall back to a soft hold
    arm_stiff = {j: SYSID_ARM_STIFFNESS.get(j, 100.0) for j in VEGA_ARM_JOINTS}
    arm_damp = {j: SYSID_ARM_DAMPING.get(j, 20.0) for j in VEGA_ARM_JOINTS}
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=VEGA_BASE_POS, joint_pos=dict(VEGA_INIT_JOINT_POS)),
        actuators={
            "left_arm": ImplicitActuatorCfg(joint_names_expr=["L_arm_j[1-7]"],
                stiffness=arm_stiff, damping=arm_damp, effort_limit=VEGA_ARM_EFFORT),
            "left_hand": ImplicitActuatorCfg(joint_names_expr=["left_.*"],
                stiffness=VEGA_HAND_STIFFNESS, damping=VEGA_HAND_DAMPING,
                effort_limit=VEGA_HAND_EFFORT, armature=VEGA_HAND_ARMATURE, friction=VEGA_HAND_FRICTION),
            "held": ImplicitActuatorCfg(joint_names_expr=["R_arm_j[1-7]", "right_.*"],
                stiffness=200.0, damping=20.0),
        },
    )


def _sysid_arm_gains(joints):
    """Re-key the sysid L_arm gains onto the given joint-name list (L_ or R_; same per-index values).
    Falls back to a soft hold for any joint without a sysid fit."""
    stiff, damp = {}, {}
    for j in joints:
        # map "R_arm_j3" or "L_arm_j3" -> the canonical "L_arm_j3" sysid entry
        key = "L_arm_j" + j.split("_j")[-1]
        stiff[j] = SYSID_ARM_STIFFNESS.get(key, 100.0)
        damp[j] = SYSID_ARM_DAMPING.get(key, 20.0)
    return stiff, damp


def make_vega_robot_cfg_bimanual_sysid() -> ArticulationCfg:
    """Bimanual cfg (BOTH arms controlled) with SYSID-TUNED gains on BOTH arms -- for the
    vega_hammer_retarget task (right arm hammers, left arm holds the thread-tester). The harmonic
    drives are symmetric, so the left-arm sysid gains re-key 1:1 onto the right arm. Mirrors
    vega_sharpa_robot.make_vega_robot_cfg_bimanual() except for the arm stiffness/damping."""
    if not SYSID_ARM_STIFFNESS:
        raise RuntimeError(
            f"no tuned gains: run scripts/tune_arm_sysid.py (expected {_TUNED_JSON}) or fill _SNAPSHOT")
    r_stiff, r_damp = _sysid_arm_gains(VEGA_RIGHT_ARM_JOINTS)
    l_stiff, l_damp = _sysid_arm_gains(VEGA_ARM_JOINTS)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VEGA_BASE_POS, joint_pos=dict({**VEGA_LEFT_ARM, **VEGA_RIGHT_ARM})),
        actuators={
            "right_arm": ImplicitActuatorCfg(joint_names_expr=["R_arm_j[1-7]"],
                stiffness=r_stiff, damping=r_damp, effort_limit=_vsr._VEGA_RIGHT_ARM_EFFORT),
            "right_hand": ImplicitActuatorCfg(joint_names_expr=["right_.*"],
                stiffness=_vsr._VEGA_RIGHT_HAND_STIFFNESS, damping=_vsr._VEGA_RIGHT_HAND_DAMPING,
                effort_limit=_vsr._VEGA_RIGHT_HAND_EFFORT, armature=_vsr._VEGA_RIGHT_HAND_ARMATURE,
                friction=_vsr._VEGA_RIGHT_HAND_FRICTION),
            "left_arm": ImplicitActuatorCfg(joint_names_expr=["L_arm_j[1-7]"],
                stiffness=l_stiff, damping=l_damp, effort_limit=VEGA_ARM_EFFORT),
            "left_hand": ImplicitActuatorCfg(joint_names_expr=["left_.*"],
                stiffness=VEGA_HAND_STIFFNESS, damping=VEGA_HAND_DAMPING,
                effort_limit=VEGA_HAND_EFFORT, armature=VEGA_HAND_ARMATURE, friction=VEGA_HAND_FRICTION),
        },
    )
