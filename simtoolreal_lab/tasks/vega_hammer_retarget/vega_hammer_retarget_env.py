"""Vega hammer env that runs the pretrained left-Sharpa policy on the RIGHT hand via shadow-IIWA
retarget + sagittal mirror. The env's canonical 29 = right arm + right hand (so HammerEnv's scene /
reward / success / goal all work off the right palm + object). Each control step:

  obs  : a MIRRORED, shadow-IIWA observation is handed to the policy (the env's _get_observations is
         overridden) -- arm joint state from a virtual IIWA arm the policy drives, hand state from the
         real right hand (un-mirrored to left), palm/object/keypoints from the real scene reflected x.
  act  : the policy's left-hand action is mirrored back -> the 7 arm dims integrate the shadow IIWA
         arm -> its palm EE pose -> reflected -> Vega right-arm differential IK; the 22 hand dims drive
         the real right hand. The left arm is parked (held).

This bridges the IIWA-trained policy onto the Vega arm (different kinematics) by treating its arm
command as an end-effector pose. v1: right hand only (left parked, not yet holding the fixture).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import (
    combine_frame_transforms, quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, subtract_frame_transforms,
)

from ..hammer.hammer_env import HammerEnv
from ..vega_hammer.shadow_iiwa import (
    IIWA_ARM_LOWER, IIWA_ARM_UPPER, SIGN_HAND, iiwa_palm_fk, mirror_quat_wxyz, mirror_vec_x, shadow_arm_step,
)
from ..vega_sharpa_robot import VEGA_ARM_JOINTS, VEGA_FINGERTIP_BODIES, VEGA_HAND_JOINTS, VEGA_PALM_BODY
from .vega_hammer_retarget_env_cfg import VegaHammerRetargetEnvCfg

_STARTARMHIGHER = [-1.571, 1.571 - math.radians(10), 0.0, 1.376 + math.radians(10), 0.0, 1.485, 1.308]
_IIWA_BASE = (0.0, 0.8, 0.0)   # original IIWA env ROBOT_BASE_POS (palm_pos obs is env-local)


class VegaHammerRetargetEnv(HammerEnv):
    cfg: VegaHammerRetargetEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        # shadow IIWA arm (7) the policy drives, init to startArmHigher
        self._sah = torch.tensor(_STARTARMHIGHER, device=dev).repeat(N, 1)
        self.shadow_tgt = self._sah.clone()
        self.shadow_prev = self._sah.clone()
        self.shadow_vel = torch.zeros((N, 7), device=dev)
        ps, qs = iiwa_palm_fk(self._sah)
        self.shadow_palm0 = ps.clone()           # shadow palm anchor (IIWA link_0 frame)
        self.shadow_rot0 = qs.clone()
        self._iiwa_base = torch.tensor(_IIWA_BASE, device=dev)
        self._sign_hand = SIGN_HAND.to(dev)
        self._qarm_lo = IIWA_ARM_LOWER.to(dev); self._qarm_hi = IIWA_ARM_UPPER.to(dev)
        # right-arm IK
        self._arm_ids = self.canonical_dof_ids_t[self.arm_slice]      # 7 right-arm joint indices
        self._ee_idx = self.palm_body_id                              # R_arm_l7 body index
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}),  # 5x default: smooth IK that the stiff arm tracks (lambda<0.05 + stiff gains oscillates)
            num_envs=N, device=dev)
        # Vega right EE (R_arm_l7) anchor pose in the robot base frame (set at reset)
        self.vega_ee0_pos = torch.zeros((N, 3), device=dev)
        self.vega_ee0_quat = torch.zeros((N, 4), device=dev); self.vega_ee0_quat[:, 0] = 1.0
        self._capture_vega_anchor(None)

        # ---------------- LEFT instance: hold the thread_tester via the SAME policy, NO mirror ----------
        self._left_player = None                                  # set by the deploy/collect script
        if getattr(self.cfg, "left_hold", False):
            self.shadow_tgt_L = self._sah.clone()
            self.shadow_prev_L = self._sah.clone()
            self.shadow_vel_L = torch.zeros((N, 7), device=dev)
            pL, qL = iiwa_palm_fk(self._sah)
            self.shadow_palm0_L = pL.clone(); self.shadow_rot0_L = qL.clone()
            self._left_arm_ids = torch.tensor(self.robot.find_joints(VEGA_ARM_JOINTS, preserve_order=True)[0],
                                              device=dev, dtype=torch.long)         # L_arm_j1..7
            self._left_hand_ids = torch.tensor(self.robot.find_joints(VEGA_HAND_JOINTS, preserve_order=True)[0],
                                               device=dev, dtype=torch.long)        # left_* 22 (canonical order)
            self._left_palm_id = self.robot.body_names.index(VEGA_PALM_BODY)        # L_arm_l7
            self._left_ft_ids = torch.tensor([self.robot.body_names.index(b) for b in VEGA_FINGERTIP_BODIES],
                                             device=dev, dtype=torch.long)
            self._left_hand_lo = self.dof_lower[:, self.hand_slice]                 # Q hand limits (same morphology)
            self._left_hand_hi = self.dof_upper[:, self.hand_slice]
            self._prev_left_hand = self.robot.data.joint_pos[:, self._left_hand_ids].clone()
            # left arm+hand ids (29) + their joint limits (for the joint-teleport clamp)
            self._left_ids = torch.cat([self._left_arm_ids, self._left_hand_ids])  # 29: arm7 + hand22
            jl = self.robot.data.joint_pos_limits
            self._left_lo = jl[:, self._left_ids, 0]
            self._left_hi = jl[:, self._left_ids, 1]
            self._ik_L = DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}),  # 5x default: smooth IK that the stiff arm tracks (lambda<0.05 + stiff gains oscillates)
                num_envs=N, device=dev)
            self.vega_ee0_pos_L = torch.zeros((N, 3), device=dev)
            self.vega_ee0_quat_L = torch.zeros((N, 4), device=dev); self.vega_ee0_quat_L[:, 0] = 1.0
            # 4 keypoints in the fixture LOCAL frame, LOCATED FROM THE BAR MESH: 2 = diagonal corners of
            # the FAR-END face (farthest from the screw), 2 = the MIDDLE of the bar. Falls back to the cfg
            # values if the mesh query fails.
            kp_local = self._left_kp_from_mesh()
            self._left_kp_local = kp_local.unsqueeze(0).expand(N, -1, -1)         # (N,4,3)
            self._left_grasp_off = kp_local.mean(0).repeat(N, 1)                 # keypoint centroid (N,3, debug print)
            self._left_hold_off = torch.tensor(self.cfg.left_hold_offset, device=dev)
            self._left_obj_scales = torch.tensor(self.cfg.left_object_scale, device=dev).repeat(N, 1)
            self._capture_vega_anchor_left(None)

    def attach_left_policy(self, player):
        """Give the env the 2nd pretrained-policy instance (own rnn state) that drives the LEFT arm/hand
        to hold the thread_tester. Called by the deploy/collect script after the env is built."""
        self._left_player = player

    def _left_kp_from_mesh(self) -> torch.Tensor:
        """Locate the 4 left keypoints FROM the thread_test BAR MESH (env_0; identical across envs),
        in the assembly-root (fixture) frame: 2 = diagonal corners of the FAR-END face (the bar end
        farthest from the screw), 2 = the bar MIDDLE. Falls back to the cfg extents on any failure."""
        c = self.cfg
        try:
            from pxr import Usd, UsdGeom
            stage = self.sim.stage
            root = None
            for p in ("/World/envs/env_0/ScrewAsm", "/World/envs/env_0/ThreadTest"):
                pr = stage.GetPrimAtPath(p)
                if pr and pr.IsValid():
                    root = pr; break
            if root is None:
                raise RuntimeError("no fixture prim")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            meshes = []
            for pr in Usd.PrimRange(root):
                if pr.IsA(UsdGeom.Mesh):
                    b = cache.ComputeRelativeBound(pr, root).ComputeAlignedRange()
                    mn, mx = b.GetMin(), b.GetMax()
                    meshes.append((mx[0] - mn[0], [mn[0], mn[1], mn[2]], [mx[0], mx[1], mx[2]]))
            if not meshes:
                raise RuntimeError("no meshes")
            meshes.sort(key=lambda m: m[0])
            _, bmn, bmx = meshes[-1]                                    # bar = largest x-extent mesh
            screw_cx = 0.5 * (meshes[0][1][0] + meshes[0][2][0]) if len(meshes) >= 2 else bmn[0]
            # far face = the bar x-end farther from the screw center
            far_x = bmx[0] if abs(bmx[0] - screw_cx) >= abs(bmn[0] - screw_cx) else bmn[0]
            mid_x = 0.5 * (bmn[0] + bmx[0])
            ky = max(abs(bmn[1]), abs(bmx[1]))
            zb, zt = bmn[2], bmx[2]
            kp = torch.tensor([[far_x, ky, zt], [far_x, -ky, zb], [mid_x, ky, zt], [mid_x, -ky, zb]],
                              device=self.device)
            print(f"[VegaRetarget] left keypoints from bar mesh: far_x={far_x:.3f} mid_x={mid_x:.3f} "
                  f"y=+/-{ky:.3f} z=[{zb:.3f},{zt:.3f}] screw_cx={screw_cx:.3f}", flush=True)
            return kp
        except Exception as e:
            print(f"[VegaRetarget] mesh keypoint query failed ({type(e).__name__}: {e}); using cfg extents", flush=True)
            fx, mx_, ky = c.left_kp_far_x, c.left_kp_mid_x, c.left_kp_y
            zc, zh = c.left_kp_z_center, c.left_kp_z_half
            return torch.tensor([[fx, ky, zc + zh], [fx, -ky, zc - zh], [mx_, ky, zc + zh], [mx_, -ky, zc - zh]],
                                device=self.device)

    def _ensure_ee_marker(self):
        """Frame marker at the RIGHT-arm TARGET EE pose (axes show position+orientation) + a small sphere
        at the ACHIEVED EE, so the commanded vs realized EE can be compared frame-by-frame."""
        if getattr(self, "_ee_marker", None) is not None:
            return
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import FRAME_MARKER_CFG
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/ee_target"
        cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        self._ee_marker = VisualizationMarkers(cfg)
        self._ee_ach_marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ee_achieved", markers={
            "s": sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))}))

    def _update_ee_marker(self, tgt_pos_b, tgt_quat_b, ee_pos_w, ee_quat_w):
        self._ensure_ee_marker()
        rp = self.robot.data.root_pos_w; rq = self.robot.data.root_quat_w
        w_pos, w_quat = combine_frame_transforms(rp, rq, tgt_pos_b, tgt_quat_b)   # target EE base->world
        self._ee_marker.visualize(translations=w_pos, orientations=w_quat)        # frame axes @ target
        self._ee_ach_marker.visualize(translations=ee_pos_w)                      # orange sphere @ achieved

    def _ensure_left_markers(self):
        """Lazily create debug markers: GREEN = the 4 left keypoints (the empty-half box the left policy
        targets), RED = the screw head, YELLOW = the left palm center."""
        if getattr(self, "_viz_markers", None) is not None:
            return
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/left_dbg", markers={
            "kp": sim_utils.SphereCfg(radius=0.014, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            "screw": sim_utils.SphereCfg(radius=0.014, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
            "palm": sim_utils.SphereCfg(radius=0.014, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.0))),
        })
        self._viz_markers = VisualizationMarkers(cfg)

    def _update_left_markers(self):
        self._ensure_left_markers()
        eo = self.scene.env_origins
        kp_w = (self._last_tt_kp + eo.unsqueeze(1)).reshape(-1, 3)              # (N*4,3) GREEN
        kp_w = kp_w.clone(); kp_w[:, 2] += 0.06                                 # raise for visibility (xy = true keypoints)
        screw_w = self.screw_head_world                                        # (N,3) RED (already world)
        palm_w = self._last_left_palm + eo                                     # (N,3) YELLOW
        trans = torch.cat([kp_w, screw_w, palm_w], dim=0)
        idx = torch.cat([torch.zeros(kp_w.shape[0], dtype=torch.long, device=self.device),
                         torch.ones(self.num_envs, dtype=torch.long, device=self.device),
                         torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)])
        self._viz_markers.visualize(translations=trans, marker_indices=idx)

    def _capture_vega_anchor(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        rp = self.robot.data.root_pos_w; rq = self.robot.data.root_quat_w
        ee_p = self.robot.data.body_pos_w[:, self._ee_idx]; ee_q = self.robot.data.body_quat_w[:, self._ee_idx]
        p_b, q_b = subtract_frame_transforms(rp, rq, ee_p, ee_q)
        self.vega_ee0_pos[ids] = p_b[ids]
        self.vega_ee0_quat[ids] = q_b[ids]

    def _capture_vega_anchor_left(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        rp = self.robot.data.root_pos_w; rq = self.robot.data.root_quat_w
        ee_p = self.robot.data.body_pos_w[:, self._left_palm_id]; ee_q = self.robot.data.body_quat_w[:, self._left_palm_id]
        p_b, q_b = subtract_frame_transforms(rp, rq, ee_p, ee_q)
        self.vega_ee0_pos_L[ids] = p_b[ids]
        self.vega_ee0_quat_L[ids] = q_b[ids]

    def _thread_tester_pose(self):
        """thread_tester (fixture) world pose (N,3)+(N,4 wxyz). physical_screw -> the screw_asm fixed
        base; else the kinematic thread_test rigid object. Returns env-local pos."""
        eo = self.scene.env_origins
        if getattr(self, "screw_asm", None) is not None:
            return self.screw_asm.data.root_pos_w - eo, self.screw_asm.data.root_quat_w
        if getattr(self, "thread_test", None) is not None:
            return self.thread_test.data.root_pos_w - eo, self.thread_test.data.root_quat_w
        return self.goal_pos.clone(), self.goal_quat.clone()   # degenerate fallback

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        self.shadow_tgt[ids] = self._sah[ids]
        self.shadow_prev[ids] = self._sah[ids]
        self.shadow_vel[ids] = 0.0
        self._ik.reset(env_ids)
        self._capture_vega_anchor(env_ids)
        if getattr(self.cfg, "left_hold", False):
            self.shadow_tgt_L[ids] = self._sah[ids]
            self.shadow_prev_L[ids] = self._sah[ids]
            self.shadow_vel_L[ids] = 0.0
            self._prev_left_hand[ids] = self.robot.data.joint_pos[ids][:, self._left_hand_ids]
            self._ik_L.reset(env_ids)
            self._capture_vega_anchor_left(env_ids)

    # ------------------------------------------------------------------ action (retarget)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        a = actions.clone().clamp(-1.0, 1.0)
        c = self.cfg

        # --- perturbation bookkeeping (teleport mask + time-since), shared by tool/joint/burst ---
        if c.tool_displacement or c.joint_displacement or c.random_action:
            self._teleported_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._steps_since_displace = self._steps_since_displace + 1.0

        # --- force/torque kicks on the object once lifted (recipe's --force_perturbation) ---
        if (c.force_perturbation or c.domain_randomization) and c.perturb_force_scale > 0.0:
            lifted = self.lifted_object.view(self.num_envs, 1, 1).float()
            fkick = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).view(self.num_envs, 1, 1).float()
            tkick = (torch.rand(self.num_envs, device=self.device) < self.random_torque_prob).view(self.num_envs, 1, 1).float()
            forces = torch.randn((self.num_envs, 1, 3), device=self.device) * self.object_mass * c.perturb_force_scale * fkick * lifted
            torques = torch.randn((self.num_envs, 1, 3), device=self.device) * self.object_mass * c.perturb_torque_scale * tkick * lifted
            self.object.set_external_force_and_torque(forces, torques)

        # --- random-action burst (BOTH arms+hands): start/continue a per-env burst of random actions; the
        # collector DROPS burst steps (flagged in _random_action_this_step). The random action is fed
        # through the SAME retarget (shadow->IK / hand-scale) for the right here and the left below. ---
        bursting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if c.random_action:
            start = ((self._burst_steps_left <= 0)
                     & (torch.rand(self.num_envs, device=self.device) < c.random_action_prob))
            if start.any():
                n = (torch.randn(self.num_envs, device=self.device).abs()
                     * c.random_action_steps_std).round().clamp(min=1.0)
                self._burst_steps_left = torch.where(start, n, self._burst_steps_left)
                self.random_action_events += start.long()
            bursting = self._burst_steps_left > 0
            self._random_action_this_step = bursting.clone()
            self._burst_steps_left = torch.where(bursting, self._burst_steps_left - 1.0, self._burst_steps_left)
            self._steps_since_displace = torch.where(bursting, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)
            rand_r = (torch.randn(self.num_envs, self.num_dofs, device=self.device) * c.random_action_std).clamp(-1.0, 1.0)
            a = torch.where(bursting.unsqueeze(-1), rand_r, a)
        a_arm, a_hand = a[:, :7], a[:, 7:]

        # --- shadow IIWA arm: integrate -> palm FK ---
        new_tgt = shadow_arm_step(self.shadow_prev, a_arm, c.dof_speed_scale, c.control_dt, c.arm_moving_average)
        self.shadow_vel = (new_tgt - self.shadow_prev) / c.control_dt
        self.shadow_prev = new_tgt
        self.shadow_tgt = new_tgt
        palm, rot = iiwa_palm_fk(new_tgt)                              # IIWA link_0 frame

        # --- mirror the shadow palm DELTA (from its anchor) onto the Vega right EE anchor (base frame) ---
        dpos = mirror_vec_x(palm - self.shadow_palm0)
        drot = mirror_quat_wxyz(quat_mul(rot, quat_conjugate(self.shadow_rot0)))
        tgt_pos = self.vega_ee0_pos + dpos
        tgt_quat = quat_mul(drot, self.vega_ee0_quat)

        # --- Vega right-arm differential IK to that EE pose ---
        rp = self.robot.data.root_pos_w; rq = self.robot.data.root_quat_w
        ee_p = self.robot.data.body_pos_w[:, self._ee_idx]; ee_q = self.robot.data.body_quat_w[:, self._ee_idx]
        ee_p_b, ee_q_b = subtract_frame_transforms(rp, rq, ee_p, ee_q)
        self._ik.set_command(torch.cat([tgt_pos, tgt_quat], dim=-1))
        jac = self.robot.root_physx_view.get_jacobians()[:, self._ee_idx - 1, :, self._arm_ids]
        q_arm = self.robot.data.joint_pos[:, self._arm_ids]
        q_des_arm = self._ik.compute(ee_p_b, ee_q_b, jac, q_arm)

        # --- right hand: mirror the policy hand action, scale to limits + EMA (compat hand branch) ---
        h = self.hand_slice
        a_hand_m = self._sign_hand * a_hand
        lo, hi = self.dof_lower[:, h], self.dof_upper[:, h]
        scaled = 0.5 * (a_hand_m + 1.0) * (hi - lo) + lo
        hand_tgt = c.hand_moving_average * scaled + (1.0 - c.hand_moving_average) * self.prev_targets[:, h]
        hand_tgt = torch.clamp(hand_tgt, lo, hi)

        self.cur_targets[:, self.arm_slice] = q_des_arm
        self.cur_targets[:, h] = hand_tgt
        self.prev_targets = self.cur_targets.clone()
        self.expert_targets = self.cur_targets.clone()

        # diagnostic: log + visualize the RIGHT-arm target EE pose (to separate EE-pose jumps from IK jumps).
        # logs ALL envs (-> one run yields N trajectories) + episode_length_buf (to mark episode resets).
        if getattr(self, "_log_ee", False):
            self._ee_log.append({
                "tgt_pos": tgt_pos.detach().cpu().clone(),              # (N,3) Vega EE target (base frame)
                "tgt_quat": tgt_quat.detach().cpu().clone(),           # (N,4)
                "ee_pos": ee_p_b.detach().cpu().clone(),               # (N,3) ACHIEVED Vega EE (base frame)
                "ee_quat": ee_q_b.detach().cpu().clone(),              # (N,4)
                "q_des": q_des_arm.detach().cpu().clone(),             # (N,7) IK output (commanded arm joints)
                "q_arm": q_arm.detach().cpu().clone(),                 # (N,7) current arm joints
                "ep_step": self.episode_length_buf.detach().cpu().clone(),  # (N,) resets to 0 on episode reset
            })
            self._update_ee_marker(tgt_pos, tgt_quat, ee_p, ee_q)

        # LEFT arm/hand: hold the thread_tester via the 2nd policy instance (no mirror). Applied via
        # set_joint_position_target on the left joint ids (the canonical _apply_action only sets the right).
        if getattr(self.cfg, "left_hold", False) and self._left_player is not None:
            self._step_left_instance(bursting)

        # --- teleport perturbations (AFTER targets are set; they override the held config for fired envs) ---
        if c.tool_displacement:
            self._tool_teleport()                                     # teleport the HAMMER (object)
        if c.joint_displacement:
            self._joint_teleport()                                    # teleport RIGHT + LEFT arm+hand joints

        if getattr(self, "_dbg_retarget", False):
            self._dbg_step = getattr(self, "_dbg_step", 0) + 1
            if self._dbg_step <= 10 or self._dbg_step % 20 == 0:
                i = 0
                ik_move = (q_des_arm[i] - q_arm[i]).abs().max().item()
                ee_err = (tgt_pos[i] - ee_p_b[i]).norm().item()
                p2o = (self.object_pos[i] - self.palm_center[i])               # palm->object (env-local)
                print(f"[rdbg t={self._dbg_step}] |a_arm|max={a_arm[i].abs().max():.2f} "
                      f"shadowdq={(new_tgt[i]-self._sah[i]).abs().max():.3f} dpos={dpos[i].norm():.3f} "
                      f"ee_err={ee_err:.3f} ik_dq={ik_move:.3f} "
                      f"palm->obj={[round(float(v),3) for v in p2o]} |={p2o.norm():.3f} "
                      f"objz={self.object_pos[i,2]:.3f}", flush=True)

    # ------------------------------------------------------------------ obs (mirrored shadow)
    def _get_observations(self) -> dict:
        ko = self.cfg.clamp_abs_observations
        h = self.hand_slice
        # arm joint_pos: shadow IIWA (unscaled via IIWA Q-limits); hand: real right -> left (sign) -> unscale
        q_arm_u = (2.0 * self.shadow_tgt - self._qarm_hi - self._qarm_lo) / (self._qarm_hi - self._qarm_lo)
        ql_hand = self._sign_hand * self.joint_pos[:, h]
        q_hand_u = (2.0 * ql_hand - self.dof_upper[:, h] - self.dof_lower[:, h]) / (self.dof_upper[:, h] - self.dof_lower[:, h])
        unscaled_q = torch.cat([q_arm_u, q_hand_u], dim=-1)
        joint_vel = torch.cat([self.shadow_vel, self._sign_hand * self.joint_vel[:, h]], dim=-1)
        prev_t = torch.cat([self.shadow_prev, self._sign_hand * self.prev_targets[:, h]], dim=-1)

        palm, rot = iiwa_palm_fk(self.shadow_tgt)
        palm_pos = palm + self._iiwa_base                              # shadow palm in IIWA env-local
        palm_rot = rot[:, [1, 2, 3, 0]]                                # xyzw
        object_rot = mirror_quat_wxyz(self.object_quat)[:, [1, 2, 3, 0]]
        pc = self.palm_center.unsqueeze(1)
        ft_rel = mirror_vec_x(self.fingertip_pos - pc).reshape(self.num_envs, -1)
        kp_rel_palm = mirror_vec_x(self.object_keypoints - pc).reshape(self.num_envs, -1)
        if self.cfg.actor_infer_goal_from_screw:
            kp_rel_goal = mirror_vec_x(self.screw_keypoints - pc).reshape(self.num_envs, -1)
        else:
            kp_rel_goal = mirror_vec_x(self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1)

        obs = torch.cat([unscaled_q, joint_vel, prev_t, palm_pos, palm_rot, object_rot,
                         ft_rel, kp_rel_palm, kp_rel_goal, self.object_scales], dim=-1).clamp(-ko, ko)
        if self.cfg.eval_append_expl_coef:
            obs = torch.cat([obs, torch.full((self.num_envs, 1), float(self.cfg.expl_exploit_coef), device=self.device)], dim=-1)
        out = {"policy": obs}
        if self.cfg.state_space and self.cfg.state_space > 0:
            terms = [obs]
            if self.cfg.actor_infer_goal_from_screw:
                terms.append((self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1))
            terms += [self.object_linvel, self.object_angvel, self.lifted_object.float().unsqueeze(-1),
                      self.keypoints_max_dist.unsqueeze(-1), self.closest_keypoint_max_dist.unsqueeze(-1),
                      self.closest_fingertip_dist, self.successes.unsqueeze(-1)]
            out["critic"] = torch.cat(terms, dim=-1).clamp(-ko, ko)
        return out

    # ================================================================== LEFT instance (thread_tester hold)
    def _left_palm_center(self):
        eo = self.scene.env_origins
        p = self.robot.data.body_pos_w[:, self._left_palm_id] - eo
        q = self.robot.data.body_quat_w[:, self._left_palm_id]
        return p + quat_apply(q, self.palm_offset), q

    def _build_left_obs(self) -> torch.Tensor:
        """The policy's 140(+coef) obs for the LEFT instance: NO mirror; object = thread_tester; goal =
        thread_tester keypoints + a constant small horizontal offset (so the policy grasps + holds)."""
        N, ko = self.num_envs, self.cfg.clamp_abs_observations
        lo, hi = self._left_hand_lo, self._left_hand_hi
        q_arm_u = (2.0 * self.shadow_tgt_L - self._qarm_hi - self._qarm_lo) / (self._qarm_hi - self._qarm_lo)
        q_hand = self.robot.data.joint_pos[:, self._left_hand_ids]                 # real Vega LEFT hand (no sign)
        q_hand_u = (2.0 * q_hand - hi - lo) / (hi - lo)
        unscaled_q = torch.cat([q_arm_u, q_hand_u], dim=-1)
        joint_vel = torch.cat([self.shadow_vel_L, self.robot.data.joint_vel[:, self._left_hand_ids]], dim=-1)
        prev_t = torch.cat([self.shadow_prev_L, self._prev_left_hand], dim=-1)

        palm, rot = iiwa_palm_fk(self.shadow_tgt_L)
        palm_pos = palm + self._iiwa_base
        palm_rot = rot[:, [1, 2, 3, 0]]                                            # xyzw, NO mirror
        palm_center, _ = self._left_palm_center()
        tt_pos, tt_quat = self._thread_tester_pose()
        # 4 keypoints on the SCREW-FREE portion (2 far-face diagonal + 2 middle), in the fixture frame
        tt_kp = tt_pos.unsqueeze(1) + quat_apply(tt_quat.unsqueeze(1).expand(-1, 4, -1), self._left_kp_local)  # (N,4,3)
        self._last_tt_kp = tt_kp                                                   # (env-local) for debug markers
        self._last_left_palm = palm_center
        object_rot = tt_quat[:, [1, 2, 3, 0]]                                      # NO mirror
        ft_p = self.robot.data.body_pos_w[:, self._left_ft_ids] - self.scene.env_origins.unsqueeze(1)
        ft_q = self.robot.data.body_quat_w[:, self._left_ft_ids]
        ft = ft_p + quat_apply(ft_q, self.fingertip_offset)                        # tip offset (matches right)
        ft_rel = (ft - palm_center.unsqueeze(1)).reshape(N, -1)                    # NO mirror
        kp_rel_palm = (tt_kp - palm_center.unsqueeze(1)).reshape(N, -1)
        goal_kp = tt_kp + self._left_hold_off                                      # constant small horizontal offset
        kp_rel_goal = (tt_kp - goal_kp).reshape(N, -1)                             # = -offset (constant)

        obs = torch.cat([unscaled_q, joint_vel, prev_t, palm_pos, palm_rot, object_rot,
                         ft_rel, kp_rel_palm, kp_rel_goal, self._left_obj_scales], dim=-1).clamp(-ko, ko)
        if self.cfg.eval_append_expl_coef:
            obs = torch.cat([obs, torch.full((N, 1), float(self.cfg.expl_exploit_coef), device=self.device)], dim=-1)
        return obs

    def _step_left_instance(self, bursting=None):
        c = self.cfg
        o_L = {"obs": self._build_left_obs()}
        a = self._left_player.get_action(o_L["obs"], is_deterministic=True).clamp(-1.0, 1.0)
        if bursting is not None and bool(bursting.any()):           # random-action burst on the LEFT too
            randL = (torch.randn_like(a) * c.random_action_std).clamp(-1.0, 1.0)
            a = torch.where(bursting.unsqueeze(-1), randL, a)
        a_arm, a_hand = a[:, :7], a[:, 7:]
        # left shadow IIWA arm (no mirror) -> palm FK
        new_tgt = shadow_arm_step(self.shadow_prev_L, a_arm, c.dof_speed_scale, c.control_dt, c.arm_moving_average)
        self.shadow_vel_L = (new_tgt - self.shadow_prev_L) / c.control_dt
        self.shadow_prev_L = new_tgt; self.shadow_tgt_L = new_tgt
        palm, rot = iiwa_palm_fk(new_tgt)
        dpos = palm - self.shadow_palm0_L                                          # NO mirror
        drot = quat_mul(rot, quat_conjugate(self.shadow_rot0_L))                   # NO mirror
        tgt_pos = self.vega_ee0_pos_L + dpos
        tgt_quat = quat_mul(drot, self.vega_ee0_quat_L)
        # Vega LEFT-arm IK
        rp = self.robot.data.root_pos_w; rq = self.robot.data.root_quat_w
        ee_p = self.robot.data.body_pos_w[:, self._left_palm_id]; ee_q = self.robot.data.body_quat_w[:, self._left_palm_id]
        ee_p_b, ee_q_b = subtract_frame_transforms(rp, rq, ee_p, ee_q)
        self._ik_L.set_command(torch.cat([tgt_pos, tgt_quat], dim=-1))
        jac = self.robot.root_physx_view.get_jacobians()[:, self._left_palm_id - 1, :, self._left_arm_ids]
        q_arm = self.robot.data.joint_pos[:, self._left_arm_ids]
        q_des_arm = self._ik_L.compute(ee_p_b, ee_q_b, jac, q_arm)
        # left hand: scale + EMA (no mirror)
        lo, hi = self._left_hand_lo, self._left_hand_hi
        scaled = 0.5 * (a_hand + 1.0) * (hi - lo) + lo
        hand_tgt = c.hand_moving_average * scaled + (1.0 - c.hand_moving_average) * self._prev_left_hand
        hand_tgt = torch.clamp(hand_tgt, lo, hi)
        self._prev_left_hand = hand_tgt
        self.robot.set_joint_position_target(q_des_arm, joint_ids=self._left_arm_ids)
        self.robot.set_joint_position_target(hand_tgt, joint_ids=self._left_hand_ids)
        if getattr(self, "_viz_left_kp", False):
            self._update_left_markers()
        if getattr(self, "_dbg_retarget", False) and getattr(self, "_dbg_step", 0) % 50 == 0:
            i = 0
            kc = self._last_tt_kp[i].mean(0)                                    # keypoint-box center (env-local)
            screw = self.screw_head_world[i] - self.scene.env_origins[i]
            palm = self._last_left_palm[i]
            print(f"[Ldbg t={getattr(self,'_dbg_step',0)}] kp_center->screw={(kc-screw).norm()*1000:.0f}mm "
                  f"palm->screw={(palm-screw).norm()*1000:.0f}mm palm->kp_center={(palm-kc).norm()*1000:.0f}mm "
                  f"kpc={[round(float(v),3) for v in kc]} palm={[round(float(v),3) for v in palm]}", flush=True)

    # ================================================================== perturbations (teleports)
    def _tool_teleport(self):
        """Teleport the HAMMER (object) by a random delta pose (+zero velocity) so the grasp slips/drops;
        the expert then recovers. Half-normal magnitude over [min,max]. Replicates SimToolRealEnv."""
        c = self.cfg
        self._displace_cooldown = torch.clamp(self._displace_cooldown - 1.0, min=0.0)
        gate = torch.ones_like(self.lifted_object) if c.tool_displace_pregrasp else self.lifted_object
        fire = ((torch.rand(self.num_envs, device=self.device) < c.tool_displace_prob)
                & gate & (self._displace_cooldown <= 0))
        ids = torch.nonzero(fire).flatten()
        if ids.numel() > 0:
            m = ids.numel()
            pos = self.object.data.root_pos_w[ids].clone(); quat = self.object.data.root_quat_w[ids].clone()
            pmag = (c.tool_displace_pos_min + (c.tool_displace_pos - c.tool_displace_pos_min) * 0.5
                    * torch.randn(m, device=self.device).abs()).clamp(max=c.tool_displace_pos)
            pdir = torch.randn(m, 3, device=self.device); pdir = pdir / (pdir.norm(dim=-1, keepdim=True) + 1e-9)
            rmag = (c.tool_displace_rot_min + (c.tool_displace_rot - c.tool_displace_rot_min) * 0.5
                    * torch.randn(m, device=self.device).abs()).clamp(max=c.tool_displace_rot)
            ax = torch.randn(m, 3, device=self.device); ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-9)
            new_pose = torch.cat([pos + pdir * pmag.unsqueeze(-1), quat_mul(quat_from_angle_axis(rmag, ax), quat)], dim=-1)
            self.object.write_root_pose_to_sim(new_pose, ids)
            self.object.write_root_velocity_to_sim(torch.zeros((m, 6), device=self.device), ids)
            self._displace_cooldown[ids] = float(c.tool_displace_cooldown)
            self.tool_displace_events[ids] += 1
        self._teleported_this_step |= fire
        self._steps_since_displace = torch.where(fire, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)

    def _joint_teleport(self):
        """Joint teleport for BOTH arms+hands -- producing GENUINE policy recovery (not an IK snap-back).
        ARM: perturb the SHADOW IIWA arm joints (the policy's EE-pose proprioception -- joint_pos[:7],
        palm_pos -- AND the IK target both derive from the shadow, so the policy SEES the displacement and
        re-plans; the real Vega arm then IK-tracks the perturbed shadow). HAND: teleport the REAL hand
        joints (+hold) -- the hand obs IS the real hand, so the policy sees it and re-grasps. expert_targets
        stays clean (recorded action = the policy's recovery command)."""
        c = self.cfg
        self._joint_displace_cooldown = torch.clamp(self._joint_displace_cooldown - 1.0, min=0.0)
        jfire = ((torch.rand(self.num_envs, device=self.device) < c.joint_displace_prob)
                 & (self._joint_displace_cooldown <= 0))
        ids = torch.nonzero(jfire).flatten()
        if ids.numel() > 0:
            m = ids.numel()
            s_arm = (c.joint_displace_arm_scale_min + (c.joint_displace_arm_scale - c.joint_displace_arm_scale_min)
                     * 0.5 * torch.randn(m, 1, device=self.device).abs()).clamp(max=c.joint_displace_arm_scale)
            s_hand = (c.joint_displace_hand_scale_min + (c.joint_displace_hand_scale - c.joint_displace_hand_scale_min)
                      * 0.5 * torch.randn(m, 1, device=self.device).abs()).clamp(max=c.joint_displace_hand_scale)
            h = self.hand_slice
            rhand_ids = self.canonical_dof_ids_t[h]                              # right-hand articulation ids
            # --- RIGHT: SHADOW arm EE-pose perturbation (policy sees it via shadow obs + IK target) ---
            self.shadow_tgt[ids] = torch.clamp(self.shadow_tgt[ids] + torch.randn(m, 7, device=self.device) * s_arm,
                                               self._qarm_lo, self._qarm_hi)
            self.shadow_prev[ids] = self.shadow_tgt[ids]
            # --- RIGHT: real HAND teleport (+ hold via cur/prev_targets so it doesn't snap back) ---
            cur_h = self.robot.data.joint_pos[ids][:, rhand_ids]
            new_h = torch.clamp(cur_h + torch.randn(m, rhand_ids.numel(), device=self.device) * s_hand,
                                self.dof_lower[ids][:, h], self.dof_upper[ids][:, h])
            self.robot.write_joint_state_to_sim(new_h, torch.zeros_like(new_h), joint_ids=rhand_ids, env_ids=ids)
            self.cur_targets[ids, 7:] = new_h; self.prev_targets[ids, 7:] = new_h
            # --- LEFT: SHADOW arm perturbation + real left-HAND teleport ---
            self.shadow_tgt_L[ids] = torch.clamp(self.shadow_tgt_L[ids] + torch.randn(m, 7, device=self.device) * s_arm,
                                                 self._qarm_lo, self._qarm_hi)
            self.shadow_prev_L[ids] = self.shadow_tgt_L[ids]
            cur_hL = self.robot.data.joint_pos[ids][:, self._left_hand_ids]
            new_hL = torch.clamp(cur_hL + torch.randn(m, self._left_hand_ids.numel(), device=self.device) * s_hand,
                                 self._left_hand_lo[ids], self._left_hand_hi[ids])
            self.robot.write_joint_state_to_sim(new_hL, torch.zeros_like(new_hL), joint_ids=self._left_hand_ids, env_ids=ids)
            self.robot.set_joint_position_target(new_hL, joint_ids=self._left_hand_ids, env_ids=ids)  # hold
            self._prev_left_hand[ids] = new_hL
            self._joint_displace_cooldown[ids] = float(c.joint_displace_cooldown)
            self.joint_displace_events[ids] += 1
        self._teleported_this_step |= jfire
        self._steps_since_displace = torch.where(jfire, torch.zeros_like(self._steps_since_displace), self._steps_since_displace)
