# `tasks/v2p` — Dual-hand V2P manipulation envs

Isaac Lab `ManagerBasedRLEnv` configurations for **dual floating-hand** manipulation against retargeted human motion data. The policy learns residual (or direct) joint targets that drive each hand's tracking controller, while the object is taught via a virtual-object-control (VOC) curriculum that gradually transfers responsibility from a kinematic helper to the policy itself.

```
v2p/
├── v2p_hand_env_cfg.py           # Manipulation base: V2PHandEnvCfg (hands + object)
├── v2p_hand_tracking_env_cfg.py  # Hand-only base: V2PHandTrackingEnvCfg (no object)
├── mdp/                          # Actions / commands / observations / rewards / terminations / events / curriculum
└── config/sharpa_wave/           # Sharpa Wave hand binding + PPO runner
```

## Registered Gym IDs

| ID | Base cfg | Notes |
|---|---|---|
| `Sharpa-V2P-v0` / `-v0-Play` | `SharpaV2PEnvCfg` | Default manipulation env (residual action, fixed-timestep VOC curriculum). |
| `Sharpa-V2P-AutoCurr-v0` | `SharpaV2PAutoCurrEnvCfg` | Same env, but uses `VirtualObjectControlCurriculum` (adaptive gates). |
| `Sharpa-V2P-Direct-v0` / `-Play` | `SharpaV2PDirectEnvCfg` | Replaces residual action with `JointDirectPositionAction`. |
| `Sharpa-V2P-Tracking-v0` / `-Play` | `SharpaV2PTrackingEnvCfg` | Hand-only tracking (no object, no contact rewards). |

Play variants set `scene.num_envs = 16`; all others default to `4096`.

## Env config hierarchy

```
ManagerBasedRLEnvCfg
└── V2PHandEnvCfg                    (v2p_hand_env_cfg.py — base manipulation)
    └── SharpaV2PEnvCfg              (binds Sharpa Wave robots + motion file)
        ├── SharpaV2PEnvCfgPlay
        ├── SharpaV2PAutoCurrEnvCfg  (swaps curriculum)
        └── SharpaV2PDirectEnvCfg    (swaps action)

ManagerBasedRLEnvCfg
└── V2PHandTrackingEnvCfg            (v2p_hand_tracking_env_cfg.py — hand only)
    └── SharpaV2PTrackingEnvCfg
```

Object assets and motion-file fields are not declared in the base configs — `SceneConfig.from_motion_file(env_cfg.motion_file)` + `apply_scene_config` patch them in at construction time (objects, fixed support surfaces, episode length, collision-group membership, etc.). See `tasks/scene_utils/`.

## Scene (`V2PSceneCfg` / `V2PTrackingSceneCfg`)

- `terrain`: plane (`prim_path="/World/ground"`)
- `right_robot`, `left_robot`: `ArticulationCfg = MISSING` — bound by the robot-specific cfg (e.g. `SharpaV2PTrackingEnvCfg` sets `RIGHT_SHARPA_WAVE_CFG` / `LEFT_SHARPA_WAVE_CFG`), overridden by `apply_scene_config` during scene construction.
- `light`, `sky_light`
- Object(s) injected by `apply_scene_config` during scene construction.

## Sim settings

| Cfg | `sim.dt` | `decimation` | control rate | episode length (s) |
|---|---|---|---|---|
| `V2PHandEnvCfg` | 0.01 (100 Hz) | 5 | 20 Hz | 20.0 (overridden by SceneConfig) |
| `V2PHandTrackingEnvCfg` | 0.005 (200 Hz) | 4 | 50 Hz | 41.4 |

## Commands

The command term is the **central data hub** — it loads the motion parquet, interpolates to sim FPS, holds reset state, advances the timestep counter, computes wrench-space contact supports, and exposes both goal targets and current state to actions / observations / rewards.

### `DualHandsObjectTrackingCommand` (`mdp/commands/hand_object_commands.py`)

Used by `V2PHandEnvCfg`. See `mdp/commands/CLAUDE.md` for details.

### `DualHandsTrackingCommand` (`mdp/commands/commands.py`)

Hand-only tracking variant used by `V2PHandTrackingEnvCfg`. Same motion-loading logic without object/contact state.

## Actions (`mdp/actions/`)

Five action terms — two hand controllers (residual / direct), two virtual object controllers (rigid / articulated) scaled by the VOC curriculum, and an unused ACT-style chunk term. See `mdp/actions/CLAUDE.md` for details.

| Term | Used by | Behavior |
|---|---|---|
| `JointResidualWithTrackingAction` | `V2PHandEnvCfg` (default) | Policy outputs **residual** wrist Δpose + finger Δjoints on top of the reference-tracking controller. |
| `JointDirectPositionAction` | `V2PHandTrackingEnvCfg`, `SharpaV2PDirectEnvCfg` | Policy outputs **direct** wrist deltas (accumulated) and finger targets (set directly); the controller's gains stay but it tracks the policy's pose, not the reference. |
| `VirtualRigidObjectControl` / `VirtualArticulatedObjectControl` | injected by `apply_scene_config` per object | A hidden controller that drags the object toward its reference pose; scaled by the VOC curriculum so its contribution decays during training. |

## Observations (`mdp/observations.py`)

Single `PolicyCfg` group, including:

| Term | Source | Dim per env |
|---|---|---|
| `wrist_position_e` | command | `3 × 2` |
| `wrist_orientation_e` | command (wxyz) | `4 × 2` |
| `wrist_velocity_b` | command (body frame) | `6 × 2` |
| `finger_joint_pos` (scaled to limits) | command | `J_f × 2` |
| `finger_joint_vel` | command | `J_f × 2` |
| `object_position_e` | command | `3 × B` |
| `object_orientation_e` | command (wxyz) | `4 × B` |
| `command` (`isaac_mdp.generated_commands`) | command term | concatenated goal pose vector |
| `actions` (`isaac_mdp.last_action`) | action manager | action dim |
| `processed_right_actions`, `processed_left_actions` | per-action processed targets | per-hand control dim |
| `contact_position_direction_in_wrist` | command + contact sensors | per-link contact positions + force directions in wrist frame |

`object_t_wrist` and `object_p_fingertip` are defined but commented out by default.

The hand-only tracking env drops object/contact terms and adds `prev_action` instead of `processed_action`.

## Rewards (`mdp/rewards.py`)

Default weights in `V2PHandEnvCfg` (most tracking-rewards start at `0.0` and are **ramped up by the curriculum**):

| Term | Weight | Purpose |
|---|---|---|
| `action_rate_l2` | -5e-3 | Smoothness on raw action deltas. |
| `action_l1` (`action_norm`) | -2e-3 | L2 norm on raw actions for both hands. |
| `object_keypoints_tracking_exp` | 0.0 → curriculum | `exp(-‖kp_cur − kp_cmd‖² / var)` on 6 principal-axis keypoints in the object frame. |
| `hand_keypoints_tracking_exp` | 0.0 → curriculum | `exp(-…)` on wrist + fingertip positions vs command. |
| `hand_joint_pos_tracking_exp` | 0.0 → curriculum | `exp(-…)` on finger joint positions vs command. |
| `termination_penalty` | -100.0 | Indicator-style penalty when episode is `terminated`. |
| `contact_wrench_support_reward` | 10.0 | JIT'd alignment of agent's wrench-space supports against the demo's per-body, per-basis-direction supports. |
| `contact_wrench_continuous_reward` | 0.0 | `contact_wrench_reward` + `exp(-avg_min_dist / approach_var)` approach bonus when not yet in contact. |
| `contact_wrench_cumulative_reward` | 0.0 | Per-body streak reward (`tanh(streak / streak_scale)`) for sustained correct contact. |
| `unintended_contact_penalty` | -2.5 | Penalize contact on bodies the demo does not touch. |
| `missed_contact_penalty` | -0.25 | Penalize lack of contact on bodies the demo does touch. |
| `contact_force_reward` | 0.0 | `exp(-‖F‖² / var)` on links currently in contact (paired with the contact gate). |
| `dexmachina_contact_tracking_reward` | 0.0 | Chamfer-style match between policy contact points and demo contact points per part. |

Commented-out (available but off-by-default): `contact_force_range_reward`, `contact_force_rate_reward`, `contact_slippage_reward`, joint-limit penalties.

Most rewards read the `DualHandsObjectTrackingCommand` and compute their values based on the command's state, see `mdp/commands/CLAUDE.md` for details.

## Terminations (`mdp/terminations.py`)

| Term | Type | Condition |
|---|---|---|
| `time_out` (`timestep_timeout`) | `time_out=True` (no penalty) | `command.timestep_counter >= retargeted_horizon - 1`. |
| `hand_wrist_away_from_trajectory` | terminal | `‖wrist − wrist_cmd‖ > threshold` for **either** hand. |
| `object_away_from_trajectory` | terminal | object position > position_threshold **or** orientation error > orientation_threshold off command. |

## Events (`mdp/events.py` + Isaac Lab built-ins)

| Term | Mode | Effect |
|---|---|---|
| `setup_collision_groups` (`configure_collision_groups`) | `prestartup` | Builds USD `RobotGroup`, `ObjectGroup`, `FixedObjectGroup`. By default robots vs. fixed objects are filtered (no collisions); robots vs. objects are kept on. Object/fixed-object lists are patched in by `apply_scene_config`. |
| `right_physics_material`, `left_physics_material` | `startup` | `randomize_rigid_body_material` on each robot: static/dynamic friction ∈ [2.0, 2.01], restitution 0, 64 buckets. |

## Curriculum (`mdp/curriculum.py`)

Two interchangeable implementations:

### `FixedTimestepCurriculum` (default, `V2PHandEnvCfg.curriculum`)

Hard-coded schedule of `(env_step, voc_scale, reward_weights)` tuples. 

### `VirtualObjectControlCurriculum` (adaptive, `Sharpa-V2P-AutoCurr-v0`)

Decays VOC scale only when **all** of these are satisfied:
1. Past `initial_wait_env_steps` (2000) and `wait_env_steps_since_last_decay` (1000).
2. Episode reward deque (`deque_maxlen=500`) is full.
3. Mean episode length / max episode length > 0.95.
4. Per-term mean rewards exceed `reward_thresholds` (e.g. `contact_wrench_support_reward ≥ 1.6`).
5. (Optional) `metric_thresholds` and `metric_upper_thresholds` on command metrics.

Default decay mode is exponential (`factor=0.9`). Custom paired VOC + reward schedules are supported via `custom_voc_schedule` / `custom_reward_schedules`.

## PPO setup (`config/sharpa_wave/agents/rsl_rl_ppo_cfg.py`)

### `SharpaV2PPPORunnerCfg` (residual + tracking envs)

```python
num_steps_per_env     = 24
max_iterations        = 20000
save_interval         = 200
empirical_normalization = True
policy                = MLP [1024, 512, 256, 128] (actor + critic), ELU
init_noise_std        = 0.1
clip_param            = 0.1
value_loss_coef       = 1.0,  use_clipped_value_loss = True
entropy_coef          = 0.001
num_learning_epochs   = 5,    num_mini_batches = 4
learning_rate         = 1e-3, schedule = "adaptive", desired_kl = 0.005
gamma                 = 0.99, lam = 0.95
max_grad_norm         = 1.0
```

### `SharpaV2PDirectPPORunnerCfg` (direct action variant)

Same network shape; the differences are tuned for a stronger exploration signal:

- `max_iterations = 5000`
- `empirical_normalization = False`
- `init_noise_std = 1.0`
- `clip_param = 0.2`, `entropy_coef = 0.005`, `desired_kl = 0.01`
- `wandb_project = "v2p_hands_direct"`
