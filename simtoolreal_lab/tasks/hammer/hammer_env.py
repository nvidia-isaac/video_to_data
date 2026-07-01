"""Env for the 'hammer' task: claw_hammer driving a prismatic-jointed nail/screw.

Reuses ScrewdriverEnv wholesale (scene, layout randomization, physical-screw articulation, obs/
reward). Everything task-specific is config: HammerEnvCfg swaps in the claw_hammer tool, the
PRISMATIC screw assembly (`physical_screw=True`), and the nail-in goal generator.

The one piece of added behavior is the CLOSED-LOOP strike goal (`responsive_goals=True`): each
control step it re-aims the hammer at the nail's CURRENT head (which sinks as the nail is driven),
so repeated blows keep landing on the sinking nail and drive it in -- the open-loop trajectory
targets the original head and whiffs after the first hit. It overrides `_set_responsive_goal`
(the screwdriver's slot-rotation version) with a hammer-strike version.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_from_matrix, quat_mul

from ..screwdriver.screwdriver_env import ScrewdriverEnv
from .hammer_env_cfg import HammerEnvCfg


class HammerEnv(ScrewdriverEnv):
    cfg: HammerEnvCfg

    def __init__(self, cfg: HammerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # closed-loop strike state (used only when responsive_goals=True)
        self._cl_phase = torch.zeros(self.num_envs, device=self.device)              # swing oscillation phase
        self._cl_over = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # latched 'over the nail'
        self._nail_joint_init = float(cfg.nail_start_height)                          # initial prismatic joint (raised)
        # per-env "screw driven into the hole" success flag (nail prismatic joint reached its limit);
        # set in _get_dones when cfg.terminate_on_nail_driven is configured (used by BC data collection).
        self.nail_driven = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # distances (m) recomputed each step in _get_dones (used for the tighter success + logged):
        #   nail_strike_dist = hammer striking face -> nail head (the hammer must be CONTACTING it)
        #   nail_hand_dist   = nearest FINGERTIP -> nail head (detect the HAND nailing it, not the hammer)
        self.nail_strike_dist = torch.full((self.num_envs,), 9.9, device=self.device)
        self.nail_hand_dist = torch.full((self.num_envs,), 9.9, device=self.device)
        self.nail_since_displace = torch.full((self.num_envs,), 1e9, device=self.device)  # steps since last teleport
        # TIGHTEST: the nail must move ONLY while the hammer touches it. Track the joint each step; any
        # movement while the head is FAR is not a hammer strike -> fail. _nail_joint_prev = last step's
        # joint; nail_max_dmove_far = per-episode max |Δjoint| while the hammer was far (logged for calib).
        self._nail_joint_prev = torch.full((self.num_envs,), self._nail_joint_init, device=self.device)
        self.nail_max_dmove_far = torch.zeros(self.num_envs, device=self.device)
        self.nail_move_far_log = torch.zeros(self.num_envs, device=self.device)   # persisted copy for diag

    def _get_dones(self):
        terminated, time_out = super()._get_dones()   # refreshes object_pos/quat + screw_head_world + fingertips
        c = self.cfg
        if c.terminate_on_nail_driven is not None and c.physical_screw and self.screw_asm is not None:
            joint = self.screw_asm.data.joint_pos[:, 0]
            seated = joint <= c.terminate_on_nail_driven          # nail prismatic joint at/near its limit
            # distance from the hammer's striking face to the nail's CURRENT (sinking) head. The face
            # = object root + TIP (tool-local striking-face center); the head sinks with the joint.
            N = self.num_envs
            tip_w = self.object_pos + self.scene.env_origins + quat_apply(self.object_quat, self._tip_local.expand(N, 3))
            head_w = self.screw_head_world.clone()
            head_w[:, 2] = head_w[:, 2] + (joint - self._nail_joint_init)
            self.nail_strike_dist = torch.norm(tip_w - head_w, dim=-1)
            # nearest fingertip -> nail head (env-local; fingertip_pos and head share the env frame).
            head_local = head_w - self.scene.env_origins
            self.nail_hand_dist = torch.norm(self.fingertip_pos - head_local.unsqueeze(1), dim=-1).amin(dim=1)
            # nail movement since last step + whether the hammer is "near" (within the contact distance).
            near_thresh = c.nail_strike_contact_dist if c.nail_strike_contact_dist is not None else 0.03
            hammer_near = self.nail_strike_dist <= near_thresh
            dmove = (joint - self._nail_joint_prev).abs()
            self.nail_max_dmove_far = torch.where(hammer_near, self.nail_max_dmove_far,
                                                  torch.maximum(self.nail_max_dmove_far, dmove))
            self.nail_move_far_log = self.nail_max_dmove_far.clone()   # persisted (not reset) for the diagnostic
            self._nail_joint_prev = joint.clone()
            driven = seated.clone()
            # TIGHTER success: require the head to be CONTACTING the hammer (face within this distance)
            # at the seated moment -> a genuine strike, not the nail drifting in / a kick pushing it.
            if c.nail_strike_contact_dist is not None:
                driven = driven & (self.nail_strike_dist <= c.nail_strike_contact_dist)
            # (optional) RULE OUT hand-nailing: a fingertip at the nail when it seats -> the HAND drove
            # it in. Off by default (the head-contact check above is the preferred hammer-vs-hand test).
            if c.nail_hand_reject_dist is not None:
                driven = driven & (self.nail_hand_dist >= c.nail_hand_reject_dist)
            # reject a "success" that lands within ~1s of a teleport: the teleport may have dropped the
            # hammer onto the screw and faked a seating, so it's not a genuine recovered strike.
            self.nail_since_displace = self._steps_since_displace.clone()   # for the diagnostic / logging
            if c.tool_displacement and c.tool_displace_success_block_steps > 0:
                driven = driven & (self._steps_since_displace >= c.tool_displace_success_block_steps)
            self.nail_driven = driven                             # SUCCESS = seated AND the hammer did it
            # The nail is irrecoverably in the hole once seated. If it seated WITHOUT the hammer (seated
            # but not a valid strike), the episode is ruined -> TERMINATE it now as a FAILURE and reset,
            # instead of wasting the rest of the budget with the screw already down. (No-op unless a
            # stricter success condition is set, since then seated == driven.)
            seated_not_by_hammer = seated & ~driven
            # TIGHTEST: the nail joint moved while the hammer was NOT near -> that movement wasn't a hammer
            # strike (hand / teleport / instability pushed it) -> fail the episode, caught at the FIRST
            # spurious move (before it even fully seats).
            bad_move = torch.zeros_like(seated)
            if c.nail_move_eps is not None:
                bad_move = (dmove > c.nail_move_eps) & ~hammer_near
            terminated = terminated | self.nail_driven | seated_not_by_hammer | bad_move
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)   # parent set the layout (fresh screw_head_world / _nominal_slot)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        self._nail_joint_prev[ids] = self._nail_joint_init   # avoid a spurious step-1 "move"
        self.nail_max_dmove_far[ids] = 0.0
        if not self.cfg.responsive_goals:
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # re-seed the closed-loop goal with the FRESH layout (the parent's reset goal ran before the
        # layout randomizer, so it used a stale head) + reset the per-env strike state.
        self._cl_phase[env_ids] = 0.0
        self._cl_over[env_ids] = False
        self._set_responsive_goal(env_ids, force_over=True)

    # ------------------------------------------------------------------ closed-loop hammer strike
    def _set_responsive_goal(self, env_ids, force_over: bool = False):
        """Re-aim the hammer at the nail's CURRENT head each step. Two regimes:
          - not yet over the nail (horizontally): hover the striking face at `resp_approach_height`
            above the head, head raised -> lifts + centers the hammer over the nail first.
          - over the nail (latched): swing the head about the grip so the face arcs DOWN onto the
            current head (driving it `screw_contact_clearance` past it) and back up, repeatedly.
        The head TRACKS the sinking nail (physical prismatic joint), so each blow re-lands on it.
        A CARROT (same as the screwdriver responsive) leads the goal only a small step ahead of the
        current tool pose, keeping it in the policy's incremental-tracking range."""
        c = self.cfg
        eo = self.scene.env_origins[env_ids]
        q = self.object_quat[env_ids]
        m = q.shape[0]
        tip = self.object_pos[env_ids] + quat_apply(q, self._tip_local.expand(m, 3))  # env-local tip

        # current nail head (env-local): the reset head, lowered by however far the joint has been driven
        head = (self.screw_head_world[env_ids] - eo).clone()
        if c.physical_screw and self.screw_asm is not None:
            joint = self.screw_asm.data.joint_pos[env_ids, 0]
            head[:, 2] = head[:, 2] + (joint - self._nail_joint_init)

        # face-down orientation R0: columns [slot, +z, slot x z]  (maps TOOL -y -> -z, BLADE +x -> slot)
        slot = self._nominal_slot[env_ids]
        zc = torch.zeros_like(slot); zc[:, 2] = 1.0
        colz = torch.cross(slot, zc, dim=-1)
        R0m = torch.stack([slot, zc, colz], dim=-1)             # (m,3,3)
        R0q = quat_from_matrix(R0m)
        tipw = quat_apply(R0q, self._tip_local.expand(m, 3))    # tip relative to root, at R0 (world)
        axis = torch.cross(tipw, zc, dim=-1)                    # swing axis (horizontal, +phi raises head)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)

        # latch 'over the nail' horizontally (so the swing lifting the tip can't un-latch it)
        dxy = torch.norm((tip - head)[:, :2], dim=-1)
        if force_over:
            self._cl_phase[env_ids] = 0.0
            self._cl_over[env_ids] = False
        else:
            self._cl_over[env_ids] = self._cl_over[env_ids] | (dxy < c.resp_over_radius)
        over = self._cl_over[env_ids]

        # advance the strike oscillation only once over the nail; phi: raised(=swing) -> 0(hit) -> raised
        self._cl_phase[env_ids] = self._cl_phase[env_ids] + over.float() * c.hammer_cl_phase_step
        phi = c.hammer_cl_swing * 0.5 * (1.0 + torch.cos(self._cl_phase[env_ids]))
        newR = quat_mul(quat_from_angle_axis(phi, axis), R0q)

        # target for the face (at the base orientation R0): drive PAST the head once over
        # (screw_contact_clearance is negative -> below the head); else hover above it.
        dz = torch.where(over,
                         torch.full((m,), self._screw_contact_clearance, device=self.device),
                         torch.full((m,), c.resp_approach_height, device=self.device))
        tgt_tip = head.clone(); tgt_tip[:, 2] = tgt_tip[:, 2] + dz
        root_c = tgt_tip - tipw                                 # grip/root so the R0 face lands at tgt_tip

        # CARROT: lead only resp_pos_hop ahead of the current tool root toward (root_c, newR)
        cur_pos = self.object_pos[env_ids]
        cur_quat = self.object_quat[env_ids]
        dvec = root_c - cur_pos
        dist = dvec.norm(dim=-1, keepdim=True)
        self.goal_pos[env_ids] = cur_pos + dvec * torch.clamp(c.resp_pos_hop / (dist + 1e-6), max=1.0)
        sgn = torch.where((cur_quat * newR).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        gq = (1.0 - c.resp_rot_alpha) * cur_quat + c.resp_rot_alpha * sgn * newR
        self.goal_quat[env_ids] = gq / (gq.norm(dim=-1, keepdim=True) + 1e-9)
