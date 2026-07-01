"""Native SimToolReal expert on the Vega RIGHT arm + RIGHT Sharpa hand, with REAL-WORLD SYSID arm PD.

Identical task + training recipe to `VegaHammerRightEnvCfg` (all the from-scratch fixes: correct
keypoint scale, grip friction, reset noise, fixed-size success, delta-goal curriculum) -- the ONLY
change is the robot's ARM stiffness/damping: instead of the stiffened analytic gains
(`_VEGA_ARM_STIFF=[900..180]`), use the gains FIT to the real Vega harmonic-drive joints' chirp sysid
(`vega_sharpa_sysid.make_vega_robot_cfg_bimanual_sysid` -> right arm gets the sysid gains via the
symmetric L->R re-key; the right Sharpa hand keeps its normal gains). Training a policy under the real
arm dynamics narrows the sim-to-real gap.

NOTE: the sysid file also documents a command DELAY (`delay_steps_at`) on the arm targets that the
implicit actuator does not model -- not applied here yet (gains are the primary sysid element); add a
target-delay buffer in the env for a fuller real match.

Train (SAPG):  train.py --task Isaac-SimToolReal-Vega-Hammer-Right-Sysid-Direct-v0 \
                 --agent_cfg rl_games_sapg_cfg.yaml --num_envs 6144 --headless
"""
from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..vega_hammer_right.vega_hammer_right_env_cfg import VegaHammerRightEnvCfg
from ..vega_sharpa_sysid import make_vega_robot_cfg_bimanual_sysid


@configclass
class VegaHammerRightSysidEnvCfg(VegaHammerRightEnvCfg):
    # swap ONLY the robot: right arm gets the real-world sysid PD (right hand + left side as before).
    robot_cfg: ArticulationCfg = make_vega_robot_cfg_bimanual_sysid()
    # everything else (joint_names / palm / fingertips / the full training recipe + fixes) is inherited
    # from VegaHammerRightEnvCfg; __post_init__ (inherited) sets the init pose / move_scene / friction /
    # reset noise / curriculum on this robot_cfg exactly the same.
