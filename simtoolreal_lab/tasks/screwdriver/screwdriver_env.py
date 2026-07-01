"""SimToolReal 'screwdriver' DirectRLEnv (IIWA14 + left Sharpa).

Subclasses `SimToolRealEnv`. Identical robot / table / observation / action / reward /
goal logic; the manipulated object is the screwdriver. Adds a PASSIVE `flat_screw` (posed
inserted in a thread_test hole) and a kinematic `thread_test` fixture.

Every reset, `_randomize_layout` randomizes the xy-plane pose (x, y, yaw) of the thread_test
+ screw (kept rigidly together so the screw stays in its hole) and the screwdriver tool,
ensuring the tool's footprint never overlaps the thread_test bar.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_from_matrix, quat_mul, sample_uniform

from ..simtoolreal.simtoolreal_env import SimToolRealEnv
import importlib

from .screwdriver_env_cfg import ScrewdriverEnvCfg


class ScrewdriverEnv(SimToolRealEnv):
    cfg: ScrewdriverEnvCfg

    def __init__(self, cfg: ScrewdriverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # cache nominal (env-local) poses used by the layout randomizer
        f = lambda x: torch.tensor(x, device=self.device, dtype=torch.float)
        self._tt_def_pos = f(cfg.thread_test_cfg.init_state.pos)
        self._tt_def_quat = f(cfg.thread_test_cfg.init_state.rot)
        self._screw_def_pos = f(cfg.screw_cfg.init_state.pos)
        self._screw_def_quat = f(cfg.screw_cfg.init_state.rot)
        self._obj_def_quat = f(cfg.object_cfg.init_state.rot)
        self._obj_def_z = float(cfg.object_cfg.init_state.pos[2])
        self._pivot = f(cfg.layout_pivot_xy)
        self._screw_head_off = f(cfg.screw_head_offset_nominal)  # head rel. to root (world @ nominal)
        self._screw_contact_clearance = float(cfg.screw_contact_clearance)  # tip-vs-head clearance (m)
        # pluggable goal generator: a module exporting TOOL/BLADE/TIP/T/compute_goals_batch
        # (tighten_traj for the 044 flat slot, tighten_traj043 for the 043 cross slot, ...).
        self._goal_gen = importlib.import_module(cfg.goal_generator_module)
        # per-env tighten goal sequences (xyzw), regenerated each reset to target each env's screw
        self._traj_T = self._goal_gen.T
        self.per_env_goals = torch.zeros((self.num_envs, self._traj_T, 7), device=self.device)
        # optional pluggable goal-pose NOISE (training diversity; off by default). Precompute the
        # per-goal-index pos/rot sigma from the schedule module + the generator's phase counts.
        self._noise_pos_sig = self._noise_rot_sig = None
        if getattr(cfg, "goal_noise_module", None):
            gg = self._goal_gen
            # phase counts for the phase-aware schedules; task-agnostic schedules ignore them. getattr
            # with a 0 default so a generator that doesn't expose phases still works (task-agnostic).
            phase_counts = tuple(getattr(gg, k, 0) for k in ("N_LIFT", "N_REORIENT", "N_OVER", "N_LOWER", "N_TURN"))
            ps, rs = importlib.import_module(cfg.goal_noise_module).sigma_schedule(self._traj_T, phase_counts)
            scale = float(getattr(cfg, "goal_noise_scale", 1.0))
            self._noise_pos_sig = (torch.tensor(ps, device=self.device, dtype=torch.float) * scale).view(1, self._traj_T, 1)
            self._noise_rot_sig = (torch.tensor(rs, device=self.device, dtype=torch.float) * scale).view(1, self._traj_T)

        # optional goal-TRAJECTORY DIVERSIFICATION (training; off by default). (a) per-episode random
        # GENERATION params via the generator's sample_diversify_params (different trajectory SHAPES);
        # (b) a SMOOTH correlated offset added to the APPROACH phases (lift+reorient+over+lower), 0 during
        # the terminal strike/turn phase so the strike/insertion stays clean. Precompute the approach
        # envelope: a parabolic bump 4x(1-x) over the approach phases (0 at the start, 1 mid, 0 by the strike).
        self._diversify = bool(getattr(cfg, "goal_diversify", False))
        self._diversify_scale = float(getattr(cfg, "goal_diversify_scale", 1.0))
        self._diversify_offset_std = float(getattr(cfg, "goal_diversify_offset_std", 0.03))
        self._diversify_sampler = getattr(self._goal_gen, "sample_diversify_params", None)
        self._diversify_env = None
        if self._diversify:
            gg = self._goal_gen
            n_app = sum(int(getattr(gg, k, 0)) for k in ("N_LIFT", "N_REORIENT", "N_OVER", "N_LOWER"))
            env_w = torch.zeros(self._traj_T, device=self.device)
            if n_app > 1:
                x = torch.arange(n_app, device=self.device, dtype=torch.float) / (n_app - 1)  # 0..1
                env_w[:n_app] = 4.0 * x * (1.0 - x)                                            # 0 -> 1 -> 0 bump
            self._diversify_env = env_w.view(1, self._traj_T, 1)

        # --- driven-screw state (screw turns + sinks as the screwdriver tightens it) ---
        # local tool geometry from the goal generator: tool -> +x, blade/arm -> +z, tip at +x
        self._tool_local = f(self._goal_gen.TOOL)   # (3,)
        self._blade_local = f(self._goal_gen.BLADE)
        self._tip_local = f(self._goal_gen.TIP)
        z = lambda: torch.zeros(self.num_envs, device=self.device)
        self.screw_nom_pos = torch.zeros((self.num_envs, 3), device=self.device)   # world
        self.screw_nom_quat = torch.zeros((self.num_envs, 4), device=self.device)  # wxyz
        self.screw_head_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.screw_turn = z()          # accumulated screw rotation about +z (rad)
        self.screw_sink = z()          # accumulated axial sink (m)
        self.driver_roll_prev = z()    # previous screwdriver blade azimuth about +z
        self.driver_roll_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # responsive-goal state: nominal slot dir per env (= Rz(layout_yaw) @ world-x); + fixed L^T
        self._nominal_slot = torch.zeros((self.num_envs, 3), device=self.device)
        self._nominal_slot[:, 0] = 1.0
        self._L_T = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], device=self.device)
        self._neg_z = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        # per-env "screw tightened" success flag (physical screw_spin rotated >= terminate_on_screw_rotated
        # from its start); set in _get_dones when that cfg is configured (used by BC data collection).
        self.screw_rotated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._screw_spin_init = torch.zeros(self.num_envs, device=self.device)  # screw_spin angle at reset
        self.tip_in_slot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # tip engaged this step
        self.screw_cw_rot = torch.zeros(self.num_envs, device=self.device)  # clockwise (tightening) rotation (rad)
        self._inbox_max_cw = torch.zeros(self.num_envs, device=self.device)  # max cw rotation reached WHILE tip-in-slot
        self.tip_dist = torch.full((self.num_envs,), 9.9, device=self.device)  # tip-to-head dist (m)
        # freeze-until-grasp: hold the object STILL at its rest pose (kills the resting SDF-on-table jitter)
        # until a fingertip reaches it, then latch RELEASED -> full unmodified dynamics for the grasp/turn.
        self._obj_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obj_frozen_pos = torch.zeros((self.num_envs, 3), device=self.device)   # world
        self._obj_frozen_quat = torch.zeros((self.num_envs, 4), device=self.device)  # world (wxyz)
        self._obj_frozen_quat[:, 0] = 1.0
        z3 = torch.zeros(self.num_envs, device=self.device)
        self.slot_along, self.slot_across, self.slot_depth = z3.clone(), z3.clone(), z3.clone()  # slot-box coords (m)

    def _recolor_screw(self, prim_path: str, color):
        """Bind a solid diffuse material to the screw prim (env_0, before clone -> replicates to all
        envs). stronger_than_descendants overrides the asset's baked material on the mesh below."""
        mat_path = f"{prim_path}/RecolorMat"
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.4)
        mat.func(mat_path, mat)
        sim_utils.bind_visual_material(prim_path, mat_path)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        # Mirror SimToolRealEnv._setup_scene, plus the passive screw + thread_test fixture.
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        table_color = getattr(self.cfg, "table_color", None)
        if table_color is not None:  # tint the work-table to match a solid ground_color (uniform scene)
            self.cfg.table_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(table_color), roughness=1.0)
        self._apply_table_dist()     # move the table further from the robot (cfg.table_dist), objects unchanged
        self.table = RigidObject(self.cfg.table_cfg)
        self.screw = self.thread_test = self.screw_asm = None
        # spawn_passive_screw=False (TRAINING): skip the screw + thread_test entirely. The reward is
        # pure tool pose-reaching; the goals only need screw_head_world (a computed point, still set in
        # _randomize_layout). The screwdriver (Object) + its SDF collider are UNCHANGED, so the
        # screwdriver's own collision is identical between train and eval -- only the passive
        # screw/fixture (never in the reward) are dropped, for speed + clean tracking.
        screw_color = getattr(self.cfg, "screw_color", None)
        if self.cfg.spawn_passive_screw:
            if self.cfg.physical_screw:
                # merged articulation: thread_test fixed base + revolute-jointed screw (spins via contact)
                act = self.cfg.screw_asm_cfg.actuators["spin"]  # apply joint resistance knobs
                act.damping = self.cfg.screw_joint_damping
                act.friction = self.cfg.screw_joint_friction
                act.armature = self.cfg.screw_joint_armature
                self.screw_asm = Articulation(self.cfg.screw_asm_cfg)
                if screw_color is not None:
                    # recolor ONLY the driven screw/nail link (not the board); the screw link is
                    # "<ScrewAsm>/screw" (see build_screw_asm043_prismatic.py).
                    self._recolor_screw(self.cfg.screw_asm_cfg.prim_path.replace("env_.*", "env_0") + "/screw", screw_color)
            else:
                self.screw = RigidObject(self.cfg.screw_cfg)
                self.thread_test = RigidObject(self.cfg.thread_test_cfg)
                if screw_color is not None:  # recolor the kinematic screw (screwdriver task default)
                    self._recolor_screw(self.cfg.screw_cfg.prim_path.replace("env_.*", "env_0"), screw_color)
        self._build_per_env_camera()  # created before clone so it replicates per env
        self._build_wrist_camera()    # palm-facing wrist cam (mounted on the link), also pre-clone
        ground_color = getattr(self.cfg, "ground_color", None)
        if ground_color is not None:
            # plain solid-color floor + backdrop (e.g. white) -- a clean uniform camera background, no
            # texture, no HDRI. Floor is a static collidable cuboid; top surface at z=0 (like the grid).
            col = tuple(ground_color)
            floor = sim_utils.CuboidCfg(
                size=(60.0, 60.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col, roughness=1.0),
            )
            floor.func("/World/ground", floor, translation=(0.0, 0.0, -0.05))
            # BACKDROP wall behind the robot (per-env, cloned) -- this is what the camera actually sees
            # as the background (the floor is mostly out of the manipulation-view frame).
            wall = sim_utils.CuboidCfg(
                size=(8.0, 0.1, 5.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col, roughness=1.0),
            )
            wall.func("/World/envs/env_0/Backdrop", wall, translation=(0.0, 1.3, 1.5))
        elif getattr(self.cfg, "ground_mdl", None):
            # textured floor (in-house Isaac Lab MDL material) instead of the default grid -- changes
            # the camera background. A static collidable cuboid; top surface at z=0 (like the grid).
            floor = sim_utils.CuboidCfg(
                size=(60.0, 60.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=self.cfg.ground_mdl, project_uvw=True,
                    texture_scale=tuple(self.cfg.ground_texture_scale)),
            )
            floor.func("/World/ground", floor, translation=(0.0, 0.0, -0.05))
            # textured BACKDROP wall behind the robot (per-env, cloned) -- this is what the camera
            # actually sees as the background (the floor is mostly out of the manipulation-view frame).
            wall = sim_utils.CuboidCfg(
                size=(8.0, 0.1, 5.0),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=self.cfg.ground_mdl, project_uvw=True,
                    texture_scale=tuple(self.cfg.ground_texture_scale)),
            )
            wall.func("/World/envs/env_0/Backdrop", wall, translation=(0.0, 1.3, 1.5))
        else:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        if self.cfg.spawn_passive_screw:
            if self.cfg.physical_screw:
                self.scene.articulations["screw_asm"] = self.screw_asm
            else:
                self.scene.rigid_objects["screw"] = self.screw
                self.scene.rigid_objects["thread_test"] = self.thread_test
        self._register_per_env_camera()
        self._register_wrist_camera()
        if getattr(self.cfg, "dome_texture", None):
            # HDRI environment light (existing Isaac Lab/Sim sky asset): lights the scene AND shows as
            # the camera background. color=white so the HDRI sets the look; intensity scales brightness.
            light = sim_utils.DomeLightCfg(
                intensity=getattr(self.cfg, "dome_intensity", 2000.0),
                color=(1.0, 1.0, 1.0),
                texture_file=self.cfg.dome_texture,
                texture_format="latlong",
            )
        elif ground_color is not None and sum(ground_color) / 3.0 > 0.5:
            # bright (white) background: a white (not gray) dome so the floor/backdrop read as true
            # white rather than light gray. Dark grounds use the standard neutral dome below.
            light = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
        else:
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Robot, tool object, goals, reward state reset by the parent (the parent also writes
        # an initial object pose; _randomize_layout overrides it below when enabled).
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self.cfg.randomize_layout:
            self._randomize_layout(env_ids)
        else:
            self._place_layout_default(env_ids)
        # freeze-until-grasp: clear the release latch + cache the object's reset pose (held still until grasp)
        self._obj_released[env_ids] = False
        self._obj_frozen_pos[env_ids] = self.object.data.root_pos_w[env_ids].clone()
        self._obj_frozen_quat[env_ids] = self.object.data.root_quat_w[env_ids].clone()
        # reset the driven-screw coupling state (screw back to nominal, no turn/sink yet)
        self.screw_turn[env_ids] = 0.0
        self.screw_sink[env_ids] = 0.0
        self._inbox_max_cw[env_ids] = 0.0   # reset the in-slot clockwise-rotation accumulator
        self.driver_roll_init[env_ids] = False
        # per-env goals are computed in the layout step above; (re)set the initial goal from them
        # (the parent set goal-0 from the shared trajectory before per_env_goals existed).
        if self.cfg.demo_mode or self.cfg.use_tighten_goals:
            self._resample_goals(env_ids, base="object_init")

    # ------------------------------------------------------------------ driven screw
    def _get_dones(self):
        out = super()._get_dones()       # computes intermediate values (object_pos/quat fresh)
        if getattr(self.cfg, "freeze_until_grasp", False):
            self._freeze_object_until_grasp()  # hold the object still until a fingertip reaches it
        if self.cfg.screw_turns_with_driver and not self.cfg.physical_screw:
            self._update_driven_screw()  # kinematic coupling (superseded by physical_screw)
        if self.cfg.responsive_goals:
            self._update_responsive_goals()  # closed-loop goal follows the screw + tip-in-slot state
        c = self.cfg
        if c.terminate_on_screw_rotated is not None and c.physical_screw and self.screw_asm is not None:
            # success = the screw has rotated >= the threshold in the TIGHTENING (clockwise-from-top)
            # direction AND the tip is in the slot. CW-from-top = NEGATIVE angle about +z (right-hand
            # rule), so screw_tighten_sign defaults to -1; cw_rot is the clockwise rotation (rad, >=0
            # when tightening, <0 if it backs off). Only legit clockwise screwing, with the tip engaged.
            signed = self.screw_asm.data.joint_pos[:, 0] - self._screw_spin_init
            self.screw_cw_rot = c.screw_tighten_sign * signed
            self.tip_in_slot = self._tip_in_slot()
            # credit ONLY clockwise rotation accrued WHILE the tip is in the slot box -- track the max
            # in-slot cw rotation. This rejects coast: if the blade slips out and the free screw spins
            # on, those steps aren't in-slot so they don't count. Success when the in-slot max hits the
            # threshold (i.e. the screw was tightened >= threshold by the blade staying engaged).
            self._inbox_max_cw = torch.where(
                self.tip_in_slot, torch.maximum(self._inbox_max_cw, self.screw_cw_rot), self._inbox_max_cw)
            self.screw_rotated = self._inbox_max_cw >= c.terminate_on_screw_rotated
            terminated, time_out = out
            return terminated | self.screw_rotated, time_out
        return out

    def _freeze_object_until_grasp(self):
        """Hold the manipulated object STILL at its reset pose (kills the resting SDF-on-table contact
        jitter) until a fingertip reaches it; once a fingertip is within `grasp_release_dist`, latch
        RELEASED so physics runs UNMODIFIED for the grasp + manipulation (no dynamics change after
        contact). Called post-physics each control step, so the held pose is what the obs/render see."""
        ftd = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1).amin(dim=1)  # min over 5 tips
        self._obj_released = self._obj_released | (ftd < self.cfg.grasp_release_dist)
        ids = torch.nonzero(~self._obj_released).flatten()
        if ids.numel() > 0:
            pose = torch.cat([self._obj_frozen_pos[ids], self._obj_frozen_quat[ids]], dim=-1)
            self.object.write_root_pose_to_sim(pose, ids)
            self.object.write_root_velocity_to_sim(torch.zeros((ids.numel(), 6), device=self.device), ids)

    def _tip_in_slot(self) -> torch.Tensor:
        """Per-env bool: the screwdriver tip is inside the screw's SLOT bounding box, computed on the fly.
        The slot is an oriented box centered at the screw head: it TURNS with the screw (its long axis =
        the nominal slot dir rotated by the screw's current spin), with a narrow width across it and a
        depth range below the head top. Far more precise than a sphere -- the tip must be near the screw
        axis, within the narrow slot, AND down inside it (not hovering above)."""
        c = self.cfg
        eo = self.scene.env_origins
        q = self.object_quat
        N = self.num_envs
        tool = quat_apply(q, self._tool_local.expand(N, 3))                       # tool axis -> world
        tip = self.object_pos + eo + quat_apply(q, self._tip_local.expand(N, 3))  # tip world
        d = tip - self.screw_head_world                                           # tip rel. slot center
        # slot orientation turns with the screw: nominal slot dir (at reset) rotated by the current spin
        if c.physical_screw and self.screw_asm is not None:
            th = self.screw_asm.data.joint_pos[:, 0]
        else:
            th = self.screw_turn
        ct, st = torch.cos(th), torch.sin(th)
        nx, ny = self._nominal_slot[:, 0], self._nominal_slot[:, 1]
        sx, sy = nx * ct - ny * st, nx * st + ny * ct                             # along-slot dir (xy unit)
        self.slot_along = d[:, 0] * sx + d[:, 1] * sy                             # along the slot (long)
        self.slot_across = -d[:, 0] * sy + d[:, 1] * sx                           # across the slot (narrow)
        self.slot_depth = d[:, 2]                                                 # +up; into slot -> negative
        self.tip_dist = torch.norm(d, dim=-1)                                     # exposed for diagnostics
        in_box = ((self.slot_along.abs() < c.slot_half_length)
                  & (self.slot_across.abs() < c.slot_half_width)
                  & (self.slot_depth < c.slot_top_tol)
                  & (self.slot_depth > -c.slot_depth))
        return in_box & (-tool[:, 2] > c.screw_engage_tipdown)

    def _update_driven_screw(self):
        """Rotate + sink the (kinematic) screw to TRACK the screwdriver's rotation about its axis,
        whenever the screwdriver tip is engaged in the slot (tip near head + tip pointing down)."""
        c = self.cfg
        eo = self.scene.env_origins
        q = self.object_quat                                   # (N,4) wxyz, screwdriver
        N = self.num_envs
        blade = quat_apply(q, self._blade_local.expand(N, 3))  # blade wide axis -> world
        tool = quat_apply(q, self._tool_local.expand(N, 3))    # tool axis (origin->tip) -> world
        tip = self.object_pos + eo + quat_apply(q, self._tip_local.expand(N, 3))  # tip world

        # engagement: tip near the screw head AND tool pointing down (into the slot)
        tip_dist = torch.norm(tip - self.screw_head_world, dim=-1)
        engaged = (tip_dist < c.screw_engage_radius) & (-tool[:, 2] > c.screw_engage_tipdown)

        # screwdriver azimuth about +z (the blade's direction in the xy-plane)
        roll = torch.atan2(blade[:, 1], blade[:, 0])
        # first engaged frame: seed prev so the first delta is ~0
        seed = engaged & (~self.driver_roll_init)
        self.driver_roll_prev = torch.where(seed, roll, self.driver_roll_prev)
        self.driver_roll_init = self.driver_roll_init | engaged
        delta = torch.atan2(torch.sin(roll - self.driver_roll_prev), torch.cos(roll - self.driver_roll_prev))
        delta = torch.clamp(delta, -0.3, 0.3)                  # reject wrap/teleport spikes
        self.driver_roll_prev = roll

        step = delta * engaged.float()
        self.screw_turn = self.screw_turn + step
        # sink monotonically with how far it has been turned (tightening); capped
        self.screw_sink = torch.clamp(self.screw_turn.abs() * (c.screw_thread_pitch / (2 * torch.pi)),
                                      0.0, c.screw_max_sink)

        # New kinematic screw pose = rotate the body about its OWN axis (the vertical line through
        # the head xy), not about the body origin. The screw root is offset from the shaft axis
        # (head = root + screw_head_offset), so we must orbit the root about the axis too, else the
        # geometry swings outside the hole. orientation: Rz(turn) * nominal; sink along -z.
        turn = self.screw_turn
        ct, st = torch.cos(turn), torch.sin(turn)
        half = turn * 0.5
        rz = torch.zeros((N, 4), device=self.device)
        rz[:, 0] = torch.cos(half)
        rz[:, 3] = torch.sin(half)
        new_quat = quat_mul(rz, self.screw_nom_quat)
        pivot = self.screw_head_world[:, :2]                 # screw axis (xy)
        off = self.screw_nom_pos[:, :2] - pivot              # root relative to the axis
        new_pos = self.screw_nom_pos.clone()
        new_pos[:, 0] = pivot[:, 0] + ct * off[:, 0] - st * off[:, 1]
        new_pos[:, 1] = pivot[:, 1] + st * off[:, 0] + ct * off[:, 1]
        new_pos[:, 2] = self.screw_nom_pos[:, 2] - self.screw_sink
        self.screw.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1))
        self.screw.write_root_velocity_to_sim(torch.zeros((N, 6), device=self.device))

    # ------------------------------------------------------------------ responsive (closed-loop) goal
    def _align_tool_blade(self, wb):
        """Rotation mapping TOOL(local +x)->-z (tip down) and BLADE(local +z)->wb (target slot dir)."""
        m = wb.shape[0]
        wa = self._neg_z.expand(m, 3)
        wbp = wb - (wb * wa).sum(-1, keepdim=True) * wa
        wbp = wbp / (wbp.norm(dim=-1, keepdim=True) + 1e-9)
        col2 = torch.cross(wa, wbp, dim=-1)
        W = torch.stack([wa, wbp, col2], dim=-1)            # (m,3,3) columns [wa, wb_perp, wa x wb_perp]
        return torch.bmm(W, self._L_T.expand(m, 3, 3))      # W @ L^T

    def _set_responsive_goal(self, env_ids, force_over=False):
        """Recompute the goal from the CURRENT state: blade follows the screw's current angle; if the
        tip is in the slot -> rotate ahead (turn the screw), else -> guide back over + lower (re-insert)."""
        c = self.cfg
        eo = self.scene.env_origins[env_ids]
        q = self.object_quat[env_ids]
        m = q.shape[0]
        tip = self.object_pos[env_ids] + quat_apply(q, self._tip_local.expand(m, 3))
        tool = quat_apply(q, self._tool_local.expand(m, 3))
        head = self.screw_head_world[env_ids] - eo
        ns = self._nominal_slot[env_ids]
        if c.physical_screw:
            ang = self.screw_asm.data.joint_pos[env_ids, 0]
        elif c.screw_turns_with_driver:
            ang = self.screw_turn[env_ids]
        else:
            ang = torch.zeros(m, device=self.device)
        d3 = torch.norm(tip - head, dim=-1)                  # 3-D tip-to-head
        dxy = torch.norm((tip - head)[:, :2], dim=-1)        # horizontal tip-to-slot
        contact = torch.full_like(d3, self._screw_contact_clearance)
        approach = torch.full_like(d3, c.resp_approach_height)
        if force_over:                                       # reset: ignore current tip, go over the slot
            engaged = torch.zeros(m, dtype=torch.bool, device=self.device)
            height = approach
        else:
            engaged = (d3 < c.resp_engage_radius) & (-tool[:, 2] > c.resp_tipdown)
            # lower into the slot once the tip is HORIZONTALLY over it (xy-aligned), regardless of height
            height = torch.where(engaged | (dxy < c.resp_over_radius), contact, approach)
        target_ang = torch.where(engaged, ang + c.resp_rotate_step, ang)   # engaged -> rotate ahead
        tac, tas = torch.cos(target_ang), torch.sin(target_ang)
        tslot = torch.stack([tac * ns[:, 0] - tas * ns[:, 1], tas * ns[:, 0] + tac * ns[:, 1],
                             torch.zeros_like(tac)], dim=-1)
        goal_R = self._align_tool_blade(tslot)
        tipoff = torch.bmm(goal_R, self._tip_local.expand(m, 3).unsqueeze(-1)).squeeze(-1)
        axz = torch.zeros_like(head); axz[:, 2] = 1.0
        tgt_pos = head + height.unsqueeze(-1) * axz - tipoff   # responsive TARGET root pose
        tgt_quat = quat_from_matrix(goal_R)
        # CARROT: lead the goal only a small step ahead of the CURRENT tool pose toward the target,
        # so it stays in the policy's incremental-tracking range (it can't track a far/reoriented jump).
        cur_pos = self.object_pos[env_ids]
        cur_quat = self.object_quat[env_ids]
        dvec = tgt_pos - cur_pos
        dist = dvec.norm(dim=-1, keepdim=True)
        self.goal_pos[env_ids] = cur_pos + dvec * torch.clamp(c.resp_pos_hop / (dist + 1e-6), max=1.0)
        sgn = torch.where((cur_quat * tgt_quat).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        gq = (1.0 - c.resp_rot_alpha) * cur_quat + c.resp_rot_alpha * sgn * tgt_quat
        self.goal_quat[env_ids] = gq / (gq.norm(dim=-1, keepdim=True) + 1e-9)

    def _update_responsive_goals(self):
        self._set_responsive_goal(self.robot._ALL_INDICES, force_over=False)

    # ------------------------------------------------------------------ per-env tighten goals
    def _resample_goals(self, env_ids, base):
        """In demo_mode, advance each env through ITS OWN tighten trajectory (per_env_goals)."""
        if self.cfg.responsive_goals:
            # closed-loop: the per-step _update_responsive_goals drives the goal. On reset set the
            # initial "over the slot" goal; ignore success-advance (base="goal") -> no-op.
            if base == "object_init":
                self._set_responsive_goal(env_ids, force_over=True)
            return
        if self.cfg.demo_mode or self.cfg.use_tighten_goals:
            idx = (self.successes[env_ids].long()) % self._traj_T
            g = self.per_env_goals[env_ids, idx]  # (n,7) xyz + xyzw
            self.goal_pos[env_ids] = g[:, 0:3]
            self.goal_quat[env_ids] = g[:, [6, 3, 4, 5]]  # xyzw -> wxyz
            return
        super()._resample_goals(env_ids, base)

    def _set_per_env_goals(self, env_ids, sd_pos, sd_quat_wxyz, screw_pos, yaw):
        """Generate each env's tighten trajectory targeting its screw head + slot (slot along
        world x in the canonical pose, rotated by the layout yaw). All args env-local torch."""
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        ox, oy, oz = self._screw_head_off[0], self._screw_head_off[1], self._screw_head_off[2]
        head = torch.empty_like(screw_pos)
        head[:, 0] = screw_pos[:, 0] + cos * ox - sin * oy
        head[:, 1] = screw_pos[:, 1] + sin * ox + cos * oy
        head[:, 2] = screw_pos[:, 2] + oz
        slot = torch.stack([cos, sin, torch.zeros_like(cos)], dim=-1)        # Rz(yaw) @ world-x
        axis = torch.zeros_like(slot); axis[:, 2] = 1.0                      # screw axis = world +z
        sd_quat_xyzw = sd_quat_wxyz[:, [1, 2, 3, 0]]
        # (a) trajectory-diversity GENERATION-PARAM noise: per-env random shape params (the hammer's
        # lift_height/swing_angle/n_strikes), fresh each reset. Off / no sampler -> generator defaults.
        extra = {}
        if self._diversify and self._diversify_sampler is not None:
            extra = self._diversify_sampler(len(env_ids), self._diversify_scale)
        goals = self._goal_gen.compute_goals_batch(
            sd_pos.detach().cpu().numpy(), sd_quat_xyzw.detach().cpu().numpy(),
            head.detach().cpu().numpy(), slot.detach().cpu().numpy(), axis.detach().cpu().numpy(),
            contact_clearance=self._screw_contact_clearance,
            turn_degrees=getattr(self.cfg, "tighten_turn_degrees", 180.0), **extra)
        g = torch.from_numpy(goals).to(self.device)                     # (n, T, 7) xyz + xyzw
        # pluggable goal-pose NOISE (training diversity): fresh per-env N(0,sigma) per goal index,
        # LARGE early (lift/over) -> ~0 late (insert/rotate). Off (None) at eval/viz -> clean goals.
        if self._noise_pos_sig is not None:
            n, T = g.shape[0], g.shape[1]
            g[:, :, 0:3] += torch.randn(n, T, 3, device=self.device) * self._noise_pos_sig
            ax = torch.randn(n, T, 3, device=self.device)
            ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-9)
            ang = torch.randn(n, T, device=self.device) * self._noise_rot_sig
            dq = quat_from_angle_axis(ang.reshape(-1), ax.reshape(-1, 3)).reshape(n, T, 4)   # wxyz
            gq = g[:, :, [6, 3, 4, 5]].reshape(-1, 4)                    # xyzw -> wxyz
            gq = quat_mul(dq.reshape(-1, 4), gq).reshape(n, T, 4)
            g[:, :, 3:7] = gq[:, :, [1, 2, 3, 0]]                        # wxyz -> xyzw
        # (b) trajectory-diversity SMOOTH PATH offset: a low-frequency correlated offset (K=3 random
        # control points linearly interpolated to T) added to the APPROACH positions, x the bump envelope
        # (0 at the start & by the strike) -> a varied-but-clean approach path. Fresh per env per reset.
        if self._diversify and self._diversify_env is not None:
            n, T = g.shape[0], g.shape[1]
            K = 3
            ctrl = torch.randn(n, K, 3, device=self.device) * (self._diversify_offset_std * self._diversify_scale)
            pos = torch.linspace(0.0, K - 1, T, device=self.device)
            lo = pos.floor().long().clamp(max=K - 1); hi = (lo + 1).clamp(max=K - 1)
            frac = (pos - lo.float()).view(1, T, 1)
            curve = ctrl[:, lo, :] * (1.0 - frac) + ctrl[:, hi, :] * frac    # (n, T, 3) smooth low-freq curve
            g[:, :, 0:3] = g[:, :, 0:3] + curve * self._diversify_env        # envelope: 0 at start & strike
        self.per_env_goals[env_ids] = g

    def _place_layout_default(self, env_ids):
        """Place thread_test + screw at their nominal inserted poses (no randomization)."""
        n = len(env_ids)
        eo = self.scene.env_origins[env_ids]
        z6 = torch.zeros((n, 6), device=self.device)
        if self.cfg.spawn_passive_screw:
            if self.cfg.physical_screw:
                # place the articulation root (thread_test base) at nominal; reset the screw_spin joint
                root = torch.cat([self._tt_def_pos.unsqueeze(0).expand(n, 3) + eo,
                                  self._tt_def_quat.unsqueeze(0).expand(n, 4)], dim=-1)
                self.screw_asm.write_root_pose_to_sim(root, env_ids)
                self.screw_asm.write_root_velocity_to_sim(z6, env_ids)
                jp = self.screw_asm.data.default_joint_pos[env_ids]  # configured init (e.g. raised nail)
                self.screw_asm.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
            else:
                for asset, pos, quat in (
                    (self.thread_test, self._tt_def_pos, self._tt_def_quat),
                    (self.screw, self._screw_def_pos, self._screw_def_quat),
                ):
                    pose = torch.cat([pos.unsqueeze(0).expand(n, 3) + eo, quat.unsqueeze(0).expand(n, 4)], dim=-1)
                    asset.write_root_pose_to_sim(pose, env_ids)
                    asset.write_root_velocity_to_sim(z6, env_ids)
        # cache the screw's nominal world pose + head for the driven-screw coupling (yaw=0 here)
        self.screw_nom_pos[env_ids] = self._screw_def_pos.unsqueeze(0).expand(n, 3) + eo
        self.screw_nom_quat[env_ids] = self._screw_def_quat.unsqueeze(0).expand(n, 4)
        self.screw_head_world[env_ids] = (self._screw_def_pos + self._screw_head_off).unsqueeze(0).expand(n, 3) + eo
        self._nominal_slot[env_ids] = 0.0
        self._nominal_slot[env_ids, 0] = 1.0   # slot along world-x (yaw=0)
        if self.cfg.demo_mode and self.demo_start_pose is not None:
            sd = self.demo_start_pose  # (7,) xyz_xyzw; screwdriver init was set to this by the parent
            self._set_per_env_goals(
                env_ids, sd[:3].unsqueeze(0).expand(n, 3), sd[[6, 3, 4, 5]].unsqueeze(0).expand(n, 4),
                self._screw_def_pos.unsqueeze(0).expand(n, 3), torch.zeros(n, device=self.device))

    def _randomize_layout(self, env_ids):
        """Randomize xy-plane poses every reset: thread_test + screw move as one rigid group
        (yaw about the bar center + xy shift, so the screw stays in its hole); the screwdriver
        gets a random xy + yaw rejection-sampled to avoid overlapping the thread_test bar."""
        n = len(env_ids)
        d = self.device
        c = self.cfg
        eo = self.scene.env_origins[env_ids]  # (n,3) world; z component is 0
        z6 = torch.zeros((n, 6), device=d)

        # --- group transform for thread_test + screw ---
        yaw = sample_uniform(c.layout_yaw_range[0], c.layout_yaw_range[1], (n,), d)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        cx = sample_uniform(*c.layout_threadtest_center_x_range, (n,), d)  # new bar-center xy
        cy = sample_uniform(*c.layout_threadtest_center_y_range, (n,), d)
        qz = torch.zeros((n, 4), device=d)
        qz[:, 0] = torch.cos(yaw / 2)
        qz[:, 3] = torch.sin(yaw / 2)

        def place(def_pos, def_quat):
            # same rigid xy transform for every group member: rotate (def - pivot) by yaw about
            # the pivot, re-anchor at the sampled bar center; z and relative pose unchanged.
            ox, oy = def_pos[0] - self._pivot[0], def_pos[1] - self._pivot[1]
            pos = torch.empty((n, 3), device=d)
            pos[:, 0] = cx + (cos * ox - sin * oy)
            pos[:, 1] = cy + (sin * ox + cos * oy)
            pos[:, 2] = def_pos[2]
            quat = quat_mul(qz, def_quat.unsqueeze(0).expand(n, 4))
            return torch.cat([pos + eo, quat], dim=-1)

        tt_world = place(self._tt_def_pos, self._tt_def_quat)
        screw_world = place(self._screw_def_pos, self._screw_def_quat)  # (n,7) world (for head/goals)
        if self.cfg.spawn_passive_screw:                    # skipped when training screw-free
            if self.cfg.physical_screw:
                # place the articulation root (thread_test base) under the group transform; reset joint
                self.screw_asm.write_root_pose_to_sim(tt_world, env_ids)
                self.screw_asm.write_root_velocity_to_sim(z6, env_ids)
                jp = self.screw_asm.data.default_joint_pos[env_ids]  # configured init (e.g. raised nail)
                self.screw_asm.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
            else:
                self.thread_test.write_root_pose_to_sim(tt_world, env_ids)
                self.thread_test.write_root_velocity_to_sim(z6, env_ids)
                self.screw.write_root_pose_to_sim(screw_world, env_ids)
                self.screw.write_root_velocity_to_sim(z6, env_ids)
        # cache the screw's nominal world pose + head (head_off rotated by the group yaw)
        self.screw_nom_pos[env_ids] = screw_world[:, :3]
        self.screw_nom_quat[env_ids] = screw_world[:, 3:7]
        ho = self._screw_head_off
        head = torch.empty((n, 3), device=d)
        head[:, 0] = screw_world[:, 0] + cos * ho[0] - sin * ho[1]
        head[:, 1] = screw_world[:, 1] + sin * ho[0] + cos * ho[1]
        head[:, 2] = screw_world[:, 2] + ho[2]
        self.screw_head_world[env_ids] = head
        self._nominal_slot[env_ids, 0] = cos    # slot = Rz(group-yaw) @ world-x
        self._nominal_slot[env_ids, 1] = sin
        self._nominal_slot[env_ids, 2] = 0.0

        # --- screwdriver: rejection-sample xy so its footprint (circle r) clears the bar OBB ---
        hx, hy = c.layout_threadtest_half_extents
        r = c.layout_min_clearance
        sx = torch.zeros(n, device=d)
        sy = torch.zeros(n, device=d)
        best_d = torch.full((n,), -1.0, device=d)   # best (max) clearance seen, for the fallback
        best_x = torch.zeros(n, device=d)
        best_y = torch.zeros(n, device=d)
        todo = torch.ones(n, dtype=torch.bool, device=d)
        for _ in range(c.layout_max_reject_iters):
            ax = sample_uniform(*c.layout_screwdriver_x_range, (n,), d)
            ay = sample_uniform(*c.layout_screwdriver_y_range, (n,), d)
            # candidate -> bar-local frame (rotate by -yaw about bar center), distance to rect
            dx, dy = ax - cx, ay - cy
            lx = cos * dx + sin * dy
            ly = -sin * dx + cos * dy
            ddx = lx - lx.clamp(-hx, hx)
            ddy = ly - ly.clamp(-hy, hy)
            dist = torch.sqrt(ddx * ddx + ddy * ddy + 1e-12)
            # track the least-overlapping in-box candidate (keeps the tool inside its +/-0.1 box)
            improve = dist > best_d
            best_d = torch.where(improve, dist, best_d)
            best_x = torch.where(improve, ax, best_x)
            best_y = torch.where(improve, ay, best_y)
            # commit the first sample that clears the required margin
            ok = dist > r
            upd = todo & ok
            sx = torch.where(upd, ax, sx)
            sy = torch.where(upd, ay, sy)
            todo = todo & ~ok
            if not bool(todo.any()):
                break
        # envs with no clearing sample: use the least-overlapping spot within the box
        sx = torch.where(todo, best_x, sx)
        sy = torch.where(todo, best_y, sy)

        syaw = sample_uniform(c.layout_yaw_range[0], c.layout_yaw_range[1], (n,), d)
        sqz = torch.zeros((n, 4), device=d)
        sqz[:, 0] = torch.cos(syaw / 2)
        sqz[:, 3] = torch.sin(syaw / 2)
        obj_quat = quat_mul(sqz, self._obj_def_quat.unsqueeze(0).expand(n, 4))
        obj_pos = torch.empty((n, 3), device=d)
        obj_pos[:, 0], obj_pos[:, 1], obj_pos[:, 2] = sx, sy, self._obj_def_z
        self.object.write_root_pose_to_sim(torch.cat([obj_pos + eo, obj_quat], dim=-1), env_ids)
        self.object.write_root_velocity_to_sim(z6, env_ids)
        self.object_init_pos[env_ids] = obj_pos  # env-local, for the lifting-reward reference

        # per-env tighten goals: target THIS env's screw head + slot (slot = Rz(group-yaw) @ world-x)
        if self.cfg.demo_mode or self.cfg.use_tighten_goals:
            screw_local = place(self._screw_def_pos, self._screw_def_quat)[:, :3] - eo
            self._set_per_env_goals(env_ids, obj_pos, obj_quat, screw_local, yaw)
