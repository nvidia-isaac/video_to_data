"""Env for the 'screwdriver043' variant: the 043 PHILLIPS screwdriver + a cross-slot screw.

Reuses ScrewdriverEnv wholesale (scene, layout, driven-screw, obs/reward). The CROSS-slot goal
generator (tighten_traj043) is now selected purely by config: Screwdriver043EnvCfg sets
`goal_generator_module`, and the base ScrewdriverEnv loads it dynamically (geometry + goals). So this
subclass adds no behavior -- it exists only to give the 043 task its own registered env class.
"""

from __future__ import annotations

from ..screwdriver.screwdriver_env import ScrewdriverEnv
from .screwdriver043_env_cfg import Screwdriver043EnvCfg


class Screwdriver043Env(ScrewdriverEnv):
    cfg: Screwdriver043EnvCfg
