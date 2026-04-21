# V2P Whole Body

Whole-body humanoid manipulation environments using SONIC controller with RL residuals. The robot tracks reference motions from either third-person video reconstruction (ReconBody) or EE-based motion planning (ReconHand).

## Environments

| Gym ID | Config | Description |
|--------|--------|-------------|
| `SonicG1-v0` | `G1SonicEnvCfg` | Base G1 env. JOINT_RESIDUAL action, unified observation space, all reward weights at zero. For custom reward configs via Hydra. |
| `SonicG1-ReconBody-v0` | `G1SonicReconBodyEnvCfg` | Body-accurate reference (MHR/third-person video). Rewards: anchor, joint, object, EE tracking + force closure. Residual scale 0.15. |
| `SonicG1-ReconHand-v0` | `G1SonicReconHandEnvCfg` | Hand-accurate reference (planner pipeline). Rewards: hand keypoints, finger joints, contact tracking. Residual scale 0.5 (from exp201). |

## Architecture

```
V2PEnvCfg (base_env_cfg.py)
    scene, commands, events, sim settings

G1SonicEnvCfg (g1_sonic_env_cfg.py)
    robot (G1 + dex hands), SONIC action, unified obs, terminations,
    contact sensors, FrameTransformers, action scale

G1SonicReconBodyEnvCfg          G1SonicReconHandEnvCfg
    body tracking rewards           hand/contact rewards
    residual_scale=0.15             residual_scale=0.5
```

## Data Flow

All reference data loads from a single Hive-partitioned parquet:
```
whole_body/{dataset}/sequence_id={seq}/robot_name={robot}/data.parquet
```

The parquet contains body qpos, EE targets, hand keypoints, finger joints, contacts, object trajectory, and binary contact labels. `SceneConfig.from_motion_file()` auto-discovers objects, support surfaces, and episode length.

## Key Components

- **TrackingCommand** (`mdp/commands/`): Central data hub. Loads parquet, provides command targets and sim state. Handles VOC decay, freeze periods, shoulder spread, action history.
- **SONIC Action** (`mdp/actions/`): JOINT_RESIDUAL — SONIC encodes reference trajectory, RL adds residuals after decode. Supports finger residuals.
- **Observations** (`mdp/observations/`): Egocentric body-frame state, 6D rotations throughout, future frame deltas, hand-object transforms, action history.
- **Rewards** (`mdp/rewards/`): Tracking rewards (Gaussian kernel) + contact/wrench rewards. ReconBody uses force closure; ReconHand uses wrench support matching.
- **Events** (`mdp/events/`): Reset to trajectory frame with configurable freeze, shoulder spread, root Z clamp, yaw-only quaternion.
- **Terminations** (`mdp/terminations/`): Freeze-aware for EE/object terms. Anchor terminations always active.

## PPO Configs

| Config | init_std | entropy | lr | network | kl |
|--------|----------|---------|-------|---------|-----|
| `G1SonicRslRlPpoCfg` (base) | 0.5 | 5e-4 | 5e-4 | [1024,512,256,128] | 0.02 |
| `G1SonicReconHandRslRlPpoCfg` | 0.5 | 1e-4 | 5e-4 | [1024,512,256,128] | 0.02 |

## Running

```bash
# Training
python experiments/run_experiment.py recon_body_apple --local

# Eval with checkpoint
isaaclab.sh -p scripts/rsl_rl/eval.py \
    --task SonicG1-ReconBody-v0 \
    --motion_file whole_body/mhr/sequence_id=apple_pick_optimized/robot_name=g1 \
    --checkpoint <path>/model_N.pt \
    --enable_cameras
```
