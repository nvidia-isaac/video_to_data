# `tasks/v2p/mdp/actions` — V2P action terms

Five `ActionTerm` implementations covering the two hand-control modes and the two virtual object controllers that the VOC curriculum decays. Every term in this package reads from `DualHandsObjectTrackingCommand` (the central data hub — see `../commands/CLAUDE.md`), and every term implements the Isaac Lab `ActionTerm` contract: `__init__`, `action_dim`, `raw_actions`, `processed_actions`, `IO_descriptor`, `process_actions`, `reset`, `apply_actions`. Heavy math lives in `@torch.jit.script` helpers at the top of each file; the class methods just stage tensors and call into them.

## File map

| File | Term | Cfg | Used by |
|---|---|---|---|
| `actions_cfg.py` | (all `ActionTermCfg`s) | — | env configs |
| `action_track_residual.py` | `JointResidualWithTrackingAction` | `JointResidualWithTrackingActionCfg` | `V2PHandEnvCfg` (default) |
| `action_direct.py` | `JointDirectPositionAction` | `JointDirectPositionActionCfg` | `V2PHandTrackingEnvCfg`, `SharpaV2PDirectEnvCfg` |
| `action_chunks.py` | `JointPositionActionChunk` | `JointPositionActionChunkCfg` | (none currently registered — ACT-style multi-step output) |
| `virtual_rigid_object_control.py` | `VirtualRigidObjectControl` | `VirtualRigidObjectControlCfg` | injected by `apply_scene_config` per rigid object |
| `virtual_articulated_object_control.py` | `VirtualArticulatedObjectControl` | `VirtualArticulatedObjectControlCfg` | injected by `apply_scene_config` per articulated object |

`__init__.py` re-exports the cfg classes from `actions_cfg`.

## Hand actions — the common shape

Both `JointResidualWithTrackingAction` and `JointDirectPositionAction` control a single floating hand (right or left, picked from `cfg.asset_name.split("_")[0]`). They share the same surface:

- **`action_dim = 3 + 3 + num_finger_joints`** — wrist position Δ (3) + wrist orientation Δ as Euler XYZ (3) + finger joint values (one per joint).
- **`_processed_actions` is `action_dim + 1`** because wrist orientation is stored as a quaternion `(qw, qx, qy, qz)`, not Euler. Layout: `[wrist_xyz (3), wrist_wxyz (4), finger_targets (J_f)]`.
- **`_raw_actions`** is the policy output before any scaling/EMA/clip — exposed for the action-norm reward.
- **`process_actions(actions)`** runs once per env step before sim. Pipeline (in the JIT helper):
  1. Store raw policy output.
  2. EMA: `actions = ema_factor * prev_actions + (1 - ema_factor) * actions * scale`.
  3. Per-block clip: `[-clip, clip]` from `cfg.{wrist_position,wrist_orientation,finger_joint}_clip`.
  4. Combine with reference / accumulate (the two terms differ here — see below).
- **`apply_actions()`** runs every sim substep (decimation × physics dt). Reads live sim state, computes wrist wrench + finger target via a JIT helper, writes them:
  - `robot.set_external_force_and_torque(forces, torques, body_ids=wrist_body_id, is_global=False)` — wrench is in **body frame**.
  - `_asset.set_joint_position_target(finger_target, joint_ids=finger_joint_ids)` — fingers are driven by sim's built-in PD.
- **`reset(env_ids)`** zeroes `raw_actions`, `prev_actions`, `wrist_forces`, `wrist_torques`, `finger_joint_pos`, and clears existing external wrench on the wrist body.
- **Scale / clip tensors** are precomputed `(num_envs, action_dim)` constants — the first 3 cols hold `wrist_position_*`, next 3 hold `wrist_orientation_*`, remaining hold `finger_joint_*`.
- **Wrist COM offset** is cached at init (`wrist_com_pose_b = robot.data._root_physx_view.get_coms()[:, 0]`, **xyzw**) and used to convert the COM-frame velocity that PhysX reports back to the link frame before the body-frame PD.
- **Gravity compensation** is read from `robot.root_physx_view.get_gravity_compensation_forces()` and added to the wrench in the body frame.

### `JointResidualWithTrackingAction` — residual on top of reference

The default action for `V2PHandEnvCfg`. The policy outputs a **residual**; the term applies it on top of the command's reference wrist pose + finger joint targets every step:

```text
wrist_target_pos_e   = command.{side}_hand_wrist_pose_command_e[:, :3] + Δpos
wrist_target_wxyz_e  = command.{side}_hand_wrist_pose_command_e[:, 3:7] ⊗ quat_xyz(Δrot_euler)
finger_target        = command.{side}_hand_finger_joint_pos_command + Δjoints
```

Then `apply_actions` runs a body-frame PD against the live wrist pose, with the gains from `cfg.tracking_controller_{linear,angular}_{stiffness,damping}` and `cfg.max_{force,torque}`. The PD output **is** the controller that "tracks" the reference; the policy just nudges where that tracker is pointed.

**Finger PD-safe clamp**: before sending finger targets to the sim PD, the JIT helper computes the position window in which `Kp·(q_target - q) + Kd·(v_target - v)` would stay inside `joint_effort_limits` (with `v_target=0`) and clamps targets into that window. `clip_actions_to_torque_limit` is the eager-mode equivalent kept for legacy callers.

Defaults from `V2PHandEnvCfg`: `wrist_position_scale=0.05 m`, `wrist_orientation_scale=0.15 rad`, `finger_joint_scale=0.15 rad`, `ema_factor=0.3`, linear `K=50, D=10`, angular `K=12, D=0.1`, `max_force=max_torque=60`.

### `JointDirectPositionAction` — policy IS the target

Used by `V2PHandTrackingEnvCfg` and `SharpaV2PDirectEnvCfg`. No reference tracking:

```text
wrist_target_pos_e   = _processed_actions[:, :3]  + clip(scale * action[:3])    # accumulated
wrist_target_wxyz_e  = _processed_actions[:, 3:7] ⊗ quat_xyz(clip(scale * action[3:6]))  # accumulated
finger_target        = clip(scale * action[6:], joint_pos_limits)               # direct
```

The wrist target is **accumulated across steps** — each step adds a clipped delta to the previous target. Fingers are clipped to joint-position limits inside the JIT helper (not the PD-torque window — direct mode does not need the residual term's torque-safety calc).

Because Isaac Lab calls `ActionTerm.reset()` **before** the command term resets, the wrist target can't be seeded from the command at reset time (the command still holds stale references). `reset()` instead flags `_needs_target_init[env_ids]=True`; the next `process_actions` initializes the wrist target from the live `command.{side}_hand_wrist_{position_e,wxyz_e}` (which the command has now refreshed) and clears the flag. This deferred init is the single most subtle interaction in this file — preserve it.

Defaults from `SharpaV2PDirectEnvCfg`: same per-block scales, but stiffer PD gains (linear `K=1000, D=100`; angular `K=40, D=0.01`), `ema_factor=0.0`, `finger_joint_clip=100.0`.

## Virtual object controllers — the VOC curriculum's payload

Both virtual object terms have **`action_dim = 0`**: the policy never controls the object. `process_actions(actions)` is a no-op. They exist so `apply_actions()` can drag the object toward the reference pose with a PD wrench, scaled by the curriculum.

Per-step path (both terms):

1. Resolve the object: `object_idx = command.cfg.object_body_names.index(cfg.asset_name)`, `object = command.objects[object_idx]`.
2. PD wrench in body frame against `command.object_body_{position,wxyz}_command_e[:, object_idx]`.
3. Gravity compensation on the object's mass (`force += -9.81 * mass * projected_gravity_b`).
4. **Scale by `command.virtual_object_controller_scale_factor_per_env`** — the per-env curriculum live value (see `../commands/CLAUDE.md` § VOC curriculum contract).
5. Clamp to `cfg.max_{force,torque}`, apply with `object.set_external_force_and_torque(..., is_global=False)`.

When `scale_factor` is 1.0 (early training / right after reset) the controller fully drags the object onto the reference. When the curriculum decays to 0 it disappears — the policy is responsible for the object.

### `VirtualRigidObjectControl`

One per rigid object in the scene. Just the PD wrench above on the object's root body. `IO_descriptor` advertises `num_bodies` for logging.

### `VirtualArticulatedObjectControl`

Same root-body PD wrench, **plus** joint-effort tracking against the demo articulation (`command.retargeted_object_articulation[command.timestep_counter]`). Restrictions enforced in `__init__`:

- Single articulated object only (`num_bodies == command.object_position_e.shape[1]`).
- Body-name order in the live USD must match `command.retargeted_object_body_names`.
- Single joint only (`num_joints == 1`).

The joint effort reuses `linear_stiffness` (Kp) and `angular_damping` (Kd) — that's intentional (matches the eager pre-JIT path); don't "fix" it without checking the regression. Applied via `object.set_joint_effort_target(effort)` + `object.write_data_to_sim()`. Wrench is written to the root body slot of a `(num_envs, num_bodies, 3)` zeroed tensor so the per-body API isn't confused by stray non-root forces.

## `JointPositionActionChunk` — ACT-style multi-step output (unused)

Not bound by any registered Gym ID today, but lives here for ACT-style policy experiments. The policy outputs a chunk of shape `(num_envs, horizon, num_joints)`; `process_actions` applies an affine `prev_target + raw * scale`, optionally clipped, and rolls `_prev_targets` forward. `apply_actions` writes `_processed_actions` as the joint position target. Doesn't touch the command term — it operates directly on `cfg.joint_names` of the bound articulation.

## What this layer reads from the command

(See `../commands/CLAUDE.md` for full property surface. Quick map:)

| Action | Reads (per-side / per-object) |
|---|---|
| `JointResidualWithTrackingAction` | `{side}_robot`, `{side}_wrist_body_id`, `{side}_finger_joint_{names,ids}`, `{side}_hand_wrist_{position_e,wxyz_e}`, `{side}_hand_wrist_pose_command_e`, `{side}_hand_finger_joint_pos_command` |
| `JointDirectPositionAction` | Same minus the command targets (wrist target is accumulated internally; finger target is direct) |
| `VirtualRigidObjectControl` | `objects[idx]`, `object_position_e[:, idx]`, `object_orientation_e[:, idx]`, `object_body_{position,wxyz}_command_e[:, idx]`, `virtual_object_controller_scale_factor_per_env` |
| `VirtualArticulatedObjectControl` | Above + `retargeted_object_articulation[timestep_counter]`, `retargeted_object_body_names` |

Action terms never write back to the command term — the command owns its tensors and is read-only from here.

## Lifecycle gotchas

- **Reset ordering**: `ActionTerm.reset` runs **before** `CommandTerm._resample_command`. `JointResidualWithTrackingAction` doesn't care (its targets come from the command live, by the time `process_actions` runs the command is fresh). `JointDirectPositionAction` does care — see the `_needs_target_init` deferred-init dance above. Add new state to `reset` accordingly.
- **`set_external_force_and_torque` is body-frame.** Every wrist/object wrench in this package is computed in body frame and passed with `is_global=False`. If you ever switch a controller to world frame, you must also remove the `quat_apply_inverse(..., wrist_wxyz, ...)` projections.
- **COM vs link frame.** PhysX reports velocities at the COM; we correct to the link frame with `cross(angvel, R(quat) @ -wrist_com_pos_b)` before computing body-frame velocity. The JIT helpers **mutate the caller's `wrist_link_vel_w` tensor in place** to match the legacy eager path — pass a copy if the original needs to survive.
- **Finger clip differs by term.** Residual clamps to **PD-torque-safe** position windows (so external wrench stays inside `joint_effort_limits`). Direct clamps to **joint position limits**. Same parameter name (`finger_joint_clip`), different semantics — that's intentional and reflects the controllers' very different gain settings.
- **`_processed_actions[:, 3] = 1.0` at init** seeds the wrist quaternion to identity (`w=1`). Without it, the first `quat_mul` would corrupt the orientation. Preserve it if you refactor the buffer layout.
- **Object terms have `action_dim=0`.** Don't add observation hooks to `raw_actions` expecting a policy slot; there is none. They are present in the action manager only so `apply_actions()` runs and the wrench gets applied.
- **`VirtualArticulatedObjectControl` is single-object, single-joint.** The asserts in `__init__` are real constraints — multi-joint articulations (e.g. drawers with nested hinges) need a new term, not a relaxation of these asserts.
