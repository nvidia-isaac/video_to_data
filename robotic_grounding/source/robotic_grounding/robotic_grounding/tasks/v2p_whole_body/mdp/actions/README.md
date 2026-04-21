# Actions

## Action Types

| Type | Class | File | Description |
|------|-------|------|-------------|
| `HIERARCHICAL` | `SONICHierarchicalAction` | `sonic_hierarchical_action.py` | RL outputs joint commands + base orientation fed INTO SONIC encoder. |
| `HIERARCHICAL_RESIDUAL` | `SONICHierarchicalResidualAction` | `sonic_hierarchical_residual_action.py` | RL residuals added to commanded joints BEFORE SONIC. |
| `JOINT_RESIDUAL` | `SONICJointResidualAction` | `sonic_joint_residual_action.py` | RL residuals added AFTER SONIC output. Default for whole-body envs. |
| `LATENT_RESIDUAL` | `SONICLatentResidualAction` | `sonic_latent_residual_action.py` | RL residuals in SONIC latent (token) space. |
| `LATENT` | `SONICLatentAction` | `sonic_latent_action.py` | RL directly outputs full latent state for decoder. No encoder pass. |
| `LATENT_HAND_POLICY` | `SONICLatentHandPolicyAction` | `sonic_latent_hand_policy_action.py` | Latent body + pretrained hand policy for fingers. |

## Configuration

`SONICActionCfg` (`sonic_action_cfg.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_type` | `SONICActionType` | `HIERARCHICAL` | Selects action class |
| `policy_dir` | str | required | Path to SONIC ONNX models (`encoder_batched.onnx`, `decoder_batched.onnx`) |
| `sonic_joint_names` | list[str] | required | Joints controlled by SONIC (29 for G1) |
| `command_name` | str | `"motion"` | Tracking command term name |
| `use_default_offset` | bool | True | Add default joint positions to output |
| `scale` | dict | — | Per-joint action scale (PD gains) |
| `residual_scale` | float | 0.1 | Scale on RL residuals (`JOINT_RESIDUAL`) |
| `finger_residual` | bool | False | RL also outputs finger residuals (`JOINT_RESIDUAL`) |
| `finger_residual_scale` | float | -1.0 | Separate finger scale (-1 = use `residual_scale`) |
| `use_tanh` | bool | True | Tanh squashing on residuals |
| `residual_joint_names` | list[str] \| None | None | Restrict residuals to subset of SONIC joints |
| `hand_policy_class` | type \| None | None | Pretrained hand policy (`LATENT_HAND_POLICY`) |
| `hand_policy_cfg` | object | None | Hand policy config (`LATENT_HAND_POLICY`) |

## Base Class

`SONICActionBase` (`sonic_actions.py`):
- Loads `encoder_batched.onnx` + `decoder_batched.onnx` via `SonicPolicy`
- Splits joints into SONIC-controlled vs direct (fingers)
- Builds observation dicts for tokenizer and decoder from command term
- Handles default joint position offsets
- Calls `command.update_action_history(processed_actions)` each step

Key methods:
- `get_sonic_joint_indices()` / `get_sonic_joint_ids()` — joint index lookups
- `get_last_sonic_actions()` — for decoder observation (last SONIC output)
