# `tasks/v2p/mdp/commands` — V2P command terms

Isaac Lab `CommandTerm` implementations for the dual-hand V2P envs. Two terms:

- **`hand_object_commands.py::DualHandsObjectTrackingCommand`** — manipulation (hands + object + contact). Used by `V2PHandEnvCfg` (`Sharpa-V2P-v0`, `-Direct-v0`, `-AutoCurr-v0`).
- **`commands.py::DualHandsTrackingCommand`** — hand-only tracking, no object. Used by `V2PHandTrackingEnvCfg` (`Sharpa-V2P-Tracking-v0`). Same motion-loading + reset logic; everything below applies after stripping object/contact pieces.

The rest of this doc covers `DualHandsObjectTrackingCommand` since it is the heavy class and the contract that downstream rewards / observations / actions depend on.

## What this term is

The **central data hub** for the dual-hand manipulation env. It owns the loaded motion parquet, advances per-env timestep counters, computes goal targets in the object-local frame (so wrist commands follow object drift), reads sim state every step, and computes wrench-space contact supports against the demo's precomputed references. **Every reward in `mdp/rewards.py`, every observation in `mdp/observations.py`, and the VOC curriculum read from this term.** Action terms (`JointResidualWithTrackingAction`, `JointDirectPositionAction`, the virtual object controllers) read the reset-wrist buffers and the VOC scale factor written by it.

## Init pipeline (`__init__`)

Nine private helpers, called in order — each leaves the listed attributes on `self`:

1. **`_init_scene_references`** — Resolves `objects` (Articulation/RigidObject list, length `num_bodies`), and per side: `{side}_robot`, `{side}_finger_joint_ids/names`, `{side}_wrist_body_id/name`, `{side}_fingertip_body_ids/names`, `object_to_{side}_hand_contact_sensors`.
2. **`_load_motion_data`** — Loads `ManoSharpaData` parquet (`cfg.motion_folder` / `motion_filters` / `motion_id`), then `interpolate_robot_motion_data` resamples to `target_num_frames = int(1 / (step_dt * cfg.motion_speed))`. Stored as `self._retargeted_motion_data`.
3. **`_init_buffers`** — Per-env counters (`timestep_counter`, `tracking_lengths`, `steps_since_last_reset`, `all_env_ids`), reset-wrist buffers (`reset_{side}_wrist_position_e/wxyz`) consumed by actions, VOC scale factors (`virtual_object_controller_scale_factor` (1,) — the **curriculum target** — and `virtual_object_controller_scale_factor_per_env` (N, 1) — the **current per-env value** that decays toward it), constant unit vectors (`X/Y/Z_UNIT_VEC`, `QUAT_UNIT_VEC`, `KEYPOINT_VECS` (6 axis-aligned)), and bookkeeping (`contact_sensor_history_length`, `num_robot_contacts_{left,right}`, `env_origins_expanded`).
4. **`_init_hand_data`** — Per-side GPU tensors of retargeted wrist / finger / fingertip / hand-frame trajectories (`retargeted_{side}_wrist_position`, `retargeted_{side}_wrist_wxyz`, `retargeted_{side}_finger_joints`, etc.).
5. **`_init_object_data`** — `retargeted_object_body_{position,wxyz,names}` (horizon, N_body, ·) and `retargeted_object_articulation` (horizon, N_joints). Asserts body-name count matches the live USD object.
6. **`_init_contact_data`** — Per-side per-link contact arrays from the parquet (horizon, num_contact_links, 3): `retargeted_{side}_link_contact_{positions,normals}_e`, `retargeted_{side}_object_contact_{positions,normals}_e`, `retargeted_{side}_object_contact_part_ids` (0 = no contact, otherwise 1-indexed body ID).
7. **`_precompute_contact_positions_normals_in_object_frame`** — Transforms env-frame contact positions/normals into the **contacted body's local frame** using `subtract_frame_transforms`. Required so the wrench supports stay correct as the object moves.
8. **`_precompute_hand_keypoints_in_object_frame`** — Same idea for wrist / fingertip / hand-frame keypoints (`retargeted_{side}_wrist_position_o`, `_wxyz_o`, etc.). Lets `right_hand_wrist_pose_command_e` recompute the goal **from the current object pose** each step.
9. **`_precompute_contact_wrench_support_values`** — Samples `cfg.num_wrench_space_basis_samples` (default 512) random directions per body (via `sample_wrench_space_basis_scaled`), builds friction-cone edge cosines/sines (`cfg.num_friction_cone_edges`=8 by default), then per (frame, body) computes the demo's wrench-space support `s_basis(t, body, b)` for each basis direction. Stored in `retargeted_{side}_contact_wrench_supports` (horizon, num_bodies, 512). Also allocates the per-env `{side}_contact_wrench_supports` buffer that `refresh_tensors` fills with the live values.
10. **`_init_metrics`** — Zero-inits W&B-logged metric buffers (`{side}_hand_wrist_position_error`, `_wxyz_error`, `_finger_joints_error`, `object_body_{position,wxyz}_error`, `object_articulation_error`, `virtual_object_controller_scale_factor`). When `cfg.enable_additional_metrics=True` also allocates wrench CV / coverage / bbox / lift / contact-frac / VOC-decay-marker diagnostics.

`_set_contact_vis_impl` is invoked at the end of init if `cfg.debug_vis=True`.

## Lifecycle: reset vs step

### `_resample_command(env_ids)` — runs on env reset

1. **Pick a random trajectory frame**: `timestep_counter[env_ids] = randint(0, horizon - 1)`. Overrides:
   - `cfg.always_reset_to_first_frame=True` → always `0`.
   - Late-curriculum eval-gap fix: once `virtual_object_controller_scale_factor < 0.1`, each env independently with probability `cfg.reset_to_first_frame_prob` is forced to `tc=0`. Closes the train-vs-eval start-distribution gap (eval always starts from frame 0).
2. **Recompute `tracking_lengths[env_ids] = horizon - tc`**, zero `steps_since_last_reset`, **set `virtual_object_controller_scale_factor_per_env[env_ids] = 1.0`** (VOC is always on at full strength for `cfg.virtual_object_control_decay_steps` after reset, then decays toward the curriculum target).
3. **`resample_compute_tensors_jit`** computes per-env reset object pose/velocity, wrist pose/velocity, and finger joints (with `cfg.reset_finger_openness` randomization: each env samples `factor ~ U[0, reset_finger_openness]`, finger targets are `factor * reference`).
4. **Write to sim**: `objects[*].write_root_pose_to_sim` + `write_root_velocity_to_sim` (+ `write_joint_state_to_sim` for articulated objects), `{side}_robot.write_root_pose_to_sim` + `write_root_velocity_to_sim` + `write_joint_state_to_sim`. **Reset wrist poses are also cached** in `reset_{side}_wrist_{position_e,wxyz}` so action terms can initialize their PD targets without re-reading sim state.
5. **`env.sim.forward()` + `env.scene.update(dt=physics_dt)`** to refresh kinematics, then mark `_tensors_dirty = True`.

### `_update_command()` — runs every env step

1. Increment `steps_since_last_reset`.
2. **Decay the per-env VOC scale** from `1.0` toward `virtual_object_controller_scale_factor` (the curriculum target):
   - `"linear"` — interpolate over `cfg.virtual_object_control_decay_steps`.
   - `"step"` / `"fixed_schedule"` / `"custom_schedule"` — hold at 1.0 then snap to target at step `decay_steps`.
   - Anything else → `ValueError`.
3. **`timestep_counter` advances only for envs past the reset phase** (`steps_since_last_reset >= decay_steps`). Variable episode lengths are an explicit consequence — accept it; do not "fix" it without checking the reward stats.
4. Mark `_tensors_dirty = True`.

### `refresh_tensors()` — lazy per-phase cache

Cleared by `_init_buffers`, `_update_command`, `_resample_command`. Called by every reward/observation property that needs derived tensors. First call per phase: (a) fills `{side}_contact_wrench_supports` (Python loop over bodies, JIT inner kernels), (b) calls `refresh_jit` to materialize a flat tuple of cached tensors:

| Cached attribute | Shape | Used by |
|---|---|---|
| `_cached_{side}_force_sq_per_link` | (N, num_robot_contacts) | `right_force_sq_per_link` property → `contact_force_reward` |
| `_cached_{side}_link_in_contact` | (N, num_robot_contacts), bool | same |
| `_cached_{side}_in_contact`, `_cached_in_contact` | (N,) | `contact_wrench_*` rewards, eval gates |
| `_cached_ref_{L,R}`, `_cached_mask_{L,R}` | (N, num_bodies, B) | `contact_wrench_reward`, `_cumulative_reward` |
| `_cached_ref_active_{per_cell,per_body,global}` | various | wrench rewards |
| `_cached_{side}_wrench_{cmd,cur}_active{_per_body}` | (N,) / (N, num_bodies) | `contact_wrench_support_reward`, `unintended_contact_penalty`, `missed_contact_penalty` |
| `_cached_object_{position,wxyz}_e_sq` | (N, num_bodies, ·) | `object_position_tracking_exp`, `object_wxyz_tracking_exp` |

Within one reward / observation phase, repeat reads are free (just attribute access). Adding a new derived tensor to the cache requires a parallel update to `refresh_jit` and the relevant property.

## Property API (consumer-facing surface)

Properties are stable names rewards/observations bind against. They fall into three frame conventions: **`_w`** (world), **`_e`** (per-env, env-origin subtracted), **`_o`** (object-body local).

**Commands (goal targets)** — recomputed from the **current** object pose each step so hand goals follow object drift:

| Property | Shape | Notes |
|---|---|---|
| `command` | (N, –) | Concatenated goal-delta vector exposed to the policy via `isaac_mdp.generated_commands`: relative wrist position/orientation for both hands, finger joint deltas, object position/orientation deltas. |
| `right_hand_wrist_pose_command_e`, `left_*` | (N, 7) | Wrist goal `[xyz, qw,qx,qy,qz]`; `combine_frame_transforms(object_pose_e, wrist_pose_o)` with `quat_unique` if `cfg.make_quat_unique`. |
| `{side}_hand_finger_joint_pos_command` | (N, J_f) | Per-side finger joint goal. |
| `{side}_hand_fingertip_position_command_{o,e}` | (N, F, 3) | Fingertip goals in object-local / env frame. |
| `object_body_{position,wxyz}_command_e` | (N, num_bodies, 3 / 4) | Object goal poses per body. |
| `{side}_hand_object_contact_command_positions_o`, `_normals_o`, `_positions_and_normals_e` | (N, num_contact_links, 3 / 6) | Per-link demo contact targets. |
| `{side}_hand_contact_wrench_supports_command` | (N, num_bodies, B) | Demo wrench-space support (precomputed slice indexed by `timestep_counter`). |

**State (current sim values)** — read from `{side}_robot.data` and `objects[*].data`:

| Property | Shape | Notes |
|---|---|---|
| `{side}_hand_wrist_position_{w,e}`, `_wxyz_e` | (N, 3 / 4) | Wrist link pose. |
| `{side}_hand_wrist_velocity_b` | (N, 6) | Body-frame wrist twist. |
| `{side}_hand_finger_joint_{pos,vel}` | (N, J_f) | Finger state. |
| `{side}_hand_fingertip_position_{w,e}`, `_orientation_e` | (N, F, 3 / 4) | Fingertip pose. |
| `object_{position,orientation}_e`, `_position_w`, `object_com_position_and_wxyz_w` | (N, num_bodies, ·) | Object body + COM poses. |
| `{side}_hand_object_contact_{positions,forces}_{w,e}` | (N, num_bodies, num_robot_contacts, 3) | Per-link contact positions/forces; force tensor is contact-sensor history along dim 1. |
| `{side}_hand_contact_wrench_supports` | (N, num_bodies, B) | Live wrench-space support, filled by `refresh_tensors`. |

**Derived / cached** — exposed via simple property forwarders to the `_cached_*` attributes (see the refresh table above): `{side}_force_sq_per_link`, `{side}_link_in_contact`, `{side}_in_contact`, `in_contact`, `ref_{left,right}`, `mask_{left,right}`, `ref_active_per_{cell,body,global}`, `{side}_wrench_{cmd,cur}_active`, `{side}_wrench_{cmd,cur}_active_per_body`, `object_{position,wxyz}_e_sq`.

**Helpers**:

- `get_command_contact_part_id(side)` → (N,) — clamped 0-indexed body ID for the side's demo contact at the current frame.
- `get_command_contact_object_position_orientation(side)` → `(position, orientation)` in env frame for that body. Used by the wrist-pose commands.

## VOC curriculum contract

This term **does not** own the curriculum logic — `mdp/curriculum.py` does. But it owns the state both sides read/write:

- **`virtual_object_controller_scale_factor` (1,)** — the curriculum **target**. Curriculum classes (`FixedTimestepCurriculum`, `VirtualObjectControlCurriculum`) write into this tensor to decay it. `_update_command` reads it as the floor that per-env VOC decays toward.
- **`virtual_object_controller_scale_factor_per_env` (N, 1)** — the **live** scale that the virtual object controllers (`VirtualRigidObjectControl`, `VirtualArticulatedObjectControl`) multiply their wrench output by. `_resample_command` slams it to `1.0`; `_update_command` decays it toward the target over `cfg.virtual_object_control_decay_steps` per the `decay_mode`.

Net effect: every env starts each episode with full VOC assistance, decays away over a configurable window, then sits at the curriculum's current target until the next reset.

## Key cfg knobs

(See `commands_cfg.py::DualHandsObjectTrackingCommandCfg` for the full list. Highlights:)

| Field | Effect |
|---|---|
| `motion_folder`, `motion_filters`, `motion_id` | Selects the `ManoSharpaData` parquet row. |
| `motion_speed=0.5` | Resampling factor when interpolating to sim FPS. |
| `reset_finger_openness=0.7` | Upper bound for the uniform reset-time finger-openness factor. `0` = always open hand on reset, `1` = always reference. |
| `initial_virtual_object_control_curriculum_scale=1.0` | Initial curriculum target. |
| `virtual_object_control_decay_steps=20`, `..._decay_mode="step"` | Per-episode VOC ramp-down. |
| `recompute_hand_keypoints_from_object=True` | Wrist/fingertip commands follow object drift. Turn off only if you want hand commands frozen at retarget-time poses. |
| `always_reset_to_first_frame`, `reset_to_first_frame_prob` | Reset-distribution overrides; the latter only fires once VOC has nearly decayed (< 0.1). |
| `num_friction_cone_edges=8`, `num_wrench_space_basis_samples=512`, `friction_coefficients=0.1` | Wrench-space basis parameters. Larger `num_wrench_space_basis_samples` → tighter support function, more memory + compute. |
| `make_quat_unique=True` | Force positive real part on wrist-pose commands. |
| `enable_additional_metrics=False` | Allocates and updates CV / coverage / bbox / lift / contact-frac diagnostics for W&B. Off by default to save compute. |

## Consumers (who reads what)

| Reader | Reads |
|---|---|
| `mdp/rewards.py::object_keypoints_tracking_exp` | `object_position_e`, `object_orientation_e`, `object_body_{position,wxyz}_command_e`, `KEYPOINT_VECS` |
| `hand_keypoints_tracking_exp`, `hand_joint_pos_tracking_exp` | `{side}_hand_wrist_pose_command_e`, `_fingertip_position_command_e`, `_finger_joint_pos_command`, current state mirrors |
| `dexmachina_contact_tracking_reward` | `{side}_hand_object_contact_positions_e`, `_command_positions_and_normals_e`, `retargeted_{side}_object_contact_is_valid`, `get_command_contact_part_id` |
| `contact_wrench_*_reward`, `unintended_contact_penalty`, `missed_contact_penalty` | `{side}_hand_contact_wrench_supports{,_command}`, `ref_*`, `{side}_wrench_*_active*`, `in_contact` |
| `contact_force_reward` | `{side}_force_sq_per_link`, `{side}_link_in_contact` |
| `mdp/observations.py` (`wrist_position_e`, `wrist_orientation_e`, `wrist_velocity_b`, `finger_joint_pos/vel`, `object_position_e`, `object_orientation_e`, `contact_position_direction_in_wrist`, `object_t_wrist`, `object_p_fingertip`) | Same state/command properties. |
| `mdp/terminations.py::timestep_timeout` | `timestep_counter`, `retargeted_horizon` |
| `hand_wrist_away_from_trajectory`, `object_away_from_trajectory` | wrist / object current vs command poses |
| Action terms (`JointResidualWithTrackingAction`, `JointDirectPositionAction`) | `reset_{side}_wrist_{position_e,wxyz}` on reset; wrist + finger goals during step |
| `VirtualRigidObjectControl`, `VirtualArticulatedObjectControl` | `virtual_object_controller_scale_factor_per_env`, object body command poses |
| `mdp/curriculum.py` (`FixedTimestepCurriculum`, `VirtualObjectControlCurriculum`) | Writes `virtual_object_controller_scale_factor`; optionally reads the additional metrics |

## Gotchas

- **Object-local goal computation depends on the contact part ID being correct**. `retargeted_{side}_object_contact_part_ids` uses 1-indexed body IDs with 0 = no contact; the property helpers clamp to `[0, num_bodies - 1]` before indexing. If a new dataset writes the wrong part IDs, wrist goals will silently snap to the wrong body and rewards will look "stuck".
- **`recompute_hand_keypoints_from_object=True` is load-bearing.** With the object under VOC assistance early in training, the object drifts noticeably from the reference; hand commands must drift with it or the policy is being asked to track the demo wrist in the wrong frame.
- **Wrench tensors are 1-indexed by body until the property reads.** Producers and readers in this file consistently subtract 1 before indexing; if you add a new wrench tensor, do the same.
- **`refresh_tensors` is the only legal way to read the cached tensors.** Don't manually call `refresh_jit` or `_compute_contact_wrench_supports` — invariants depend on `_tensors_dirty` toggling between reset/step boundaries.
- **`enable_additional_metrics=False` skips `_precompute_bbox_corner_vecs` and the W&B diagnostics**. Flipping it on at runtime is not supported — set in cfg.
- **`always_reset_to_first_frame` and `reset_to_first_frame_prob` interact**. The latter is gated on `virtual_object_controller_scale_factor < 0.1`, so it has no effect during the VOC-assisted phase of training.
