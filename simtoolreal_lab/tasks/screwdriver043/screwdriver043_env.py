"""Env for the 'screwdriver043' variant: the 043 PHILLIPS screwdriver + a cross-slot screw.

Reuses ScrewdriverEnv wholesale (scene, layout, driven-screw, obs/reward) and only swaps the
per-env goal generator to the CROSS-slot one (tighten_traj043), which aligns the 4-fold tip & slot
with a minimal-rotation tip-down + nearest-arm (mod 90deg) roll snap. The aligned-043 mesh shares
the aligned-044 local frame, so all the inherited geometry (TOOL/BLADE/TIP) carries over unchanged.
"""

from __future__ import annotations

import torch

from ..screwdriver.screwdriver_env import ScrewdriverEnv
from . import tighten_traj043
from .screwdriver043_env_cfg import Screwdriver043EnvCfg


class Screwdriver043Env(ScrewdriverEnv):
    cfg: Screwdriver043EnvCfg

    def __init__(self, cfg: Screwdriver043EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # same trajectory length (74); keep per_env_goals as allocated by the parent
        self._traj_T = tighten_traj043.T

    def _set_per_env_goals(self, env_ids, sd_pos, sd_quat_wxyz, screw_pos, yaw):
        """Identical head/slot/axis setup as the flat env, but route through the CROSS generator.
        slot = Rz(yaw)@world-x is one arm of the cross; the generator's mod-90deg snap handles the
        other three. All args are env-local torch tensors."""
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        ox, oy, oz = self._screw_head_off[0], self._screw_head_off[1], self._screw_head_off[2]
        head = torch.empty_like(screw_pos)
        head[:, 0] = screw_pos[:, 0] + cos * ox - sin * oy
        head[:, 1] = screw_pos[:, 1] + sin * ox + cos * oy
        head[:, 2] = screw_pos[:, 2] + oz
        slot = torch.stack([cos, sin, torch.zeros_like(cos)], dim=-1)        # one cross arm
        axis = torch.zeros_like(slot); axis[:, 2] = 1.0                      # screw axis = world +z
        sd_quat_xyzw = sd_quat_wxyz[:, [1, 2, 3, 0]]
        goals = tighten_traj043.compute_goals_batch(
            sd_pos.detach().cpu().numpy(), sd_quat_xyzw.detach().cpu().numpy(),
            head.detach().cpu().numpy(), slot.detach().cpu().numpy(), axis.detach().cpu().numpy(),
            contact_clearance=self._screw_contact_clearance)
        self.per_env_goals[env_ids] = torch.from_numpy(goals).to(self.device)
