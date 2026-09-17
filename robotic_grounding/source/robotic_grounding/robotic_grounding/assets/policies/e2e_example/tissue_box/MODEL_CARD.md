# Model Overview

### Description:

Video-to-Data (V2D) Vega Sharpa Tissue-Box Expert Policy v1.0.0 controls the 58 actuated arm and hand joints of the Vega/Dexmate Sharpa embodiment to follow a reconstructed tissue-box lift, hold, and place-back reference motion in simulation. NVIDIA developed the checkpoint as part of the V2D robotic-grounding expert-policy workflow using Proximal Policy Optimization (PPO). It is intended for simulation research, reproducible reference-motion tracking, and generation of expert rollouts for downstream robot-policy training.

The repository distributes `vega_sharpa_policy.onnx`, a deterministic actor export from PPO training. The training-time critic, optimizer state, learned stochastic action scale, and training metadata are not packaged in the ONNX artifact.

**Model Owner:** NVIDIA Corporation.

This model is ready for commercial or non-commercial use.

### License/Terms of Use:

GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

### Deployment Geography:

Global.

### Use Case:

Robotics developers and researchers can use the model with the corresponding V2D and Robotic Grounding code to replay the tissue-box manipulation in Isaac Lab, evaluate the documented simulation contract, or generate expert trajectories. The checkpoint is specific to the Vega/Dexmate Sharpa embodiment, the provided tissue-box asset, and the associated retargeted reference motion.

The checkpoint has not been validated for direct or unsupervised physical-robot control. A physical deployment requires task-specific simulation validation, staged hardware testing, collision and force safeguards, human supervision, and an independent assessment of the target system.

### Release Date:

GitHub 09/15/2026 via [https://github.com/nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data).

## Model Architecture:

**Architecture Type:** Other - Feed-forward multilayer perceptron (MLP).

**Network Architecture:** Separate actor and critic networks with hidden dimensions 512, 256, and 128 and exponential linear unit (ELU) activations. The actor maps a 445-dimensional policy observation to the mean of a 58-dimensional action distribution. The critic consumes the same observation group and predicts a scalar value. The action standard deviation is represented by 58 learned scalar parameters during training; deterministic inference uses the actor mean.

**Base Model:** None. The policy was trained from a zero-actor initialization centered on the reference action rather than fine-tuned from a pretrained model.

**Number of Model Parameters:** The packaged deterministic ONNX actor contains 400,058 weights and biases. The full training-time actor-critic architecture contains approximately `7.9 × 10^5` trainable parameters, including the learned action-standard-deviation parameters. The training-time count is derived from the preserved network configuration and input/output dimensions; optimizer state is not counted as trainable model parameters.

**Training Algorithm:** PPO with a 0.1 clipping parameter, adaptive learning rate initialized at `5 × 10^-4`, discount factor 0.99, generalized-advantage-estimation factor 0.95, five learning epochs, four minibatches, entropy coefficient 0.001, and maximum gradient norm 1.0. Training used reference tracking, hand-keypoint tracking, object-keypoint tracking, and force-closure rewards, with penalties for termination, action magnitude, and action rate.

## Input(s):

**Input Type(s):** Tabular numerical robot and task state.

**Input Format(s):** Tensor. The ONNX graph input is named `obs` and has data type `float32`.

**Input Parameters:** The ONNX graph has fixed shape `(1, 445)`. The V2D `OnnxSourcePolicy` wrapper accepts a two-dimensional `(num_envs, 445)` policy-observation tensor and evaluates fixed-batch rows through the graph. The last dimension is ordered as follows:

| Observation term | Size | Semantics |
|---|---:|---|
| `joint_pos_rel` | 58 | Joint positions relative to the robot defaults, in radians |
| `joint_vel_rel` | 58 | Joint velocities, in radians per second |
| `motion_joint_pos_delta` | 174 | Joint-position deltas for three future reference frames, in radians |
| `motion_ee_pos_delta` | 18 | World-frame end-effector position deltas for two wrists over three future reference frames |
| `motion_ee_quat_delta` | 36 | Relative end-effector orientations in a six-dimensional rotation representation for two wrists over three future frames |
| `left_hand_object_transform`, `right_hand_object_transform` | 7 each | Left-then-right wrist position and WXYZ quaternion in the tracked object frame |
| `object_pose_delta` | 7 | Reference-object pose delta in the current object frame: position plus WXYZ quaternion |
| `wrist_position_e`, `wrist_wxyz_e` | 6 and 8 | Right-then-left wrist positions and WXYZ quaternions in the environment-local frame |
| `object_position_e`, `object_wxyz_e` | 3 and 4 | Object position and WXYZ quaternion in the environment-local frame |
| `trajectory_progress` | 1 | Normalized reference progress |
| `last_action` | 58 | Previous raw dimensionless policy action |

**Other Properties Related to Input:** The released V2D integration passes the raw concatenated policy observation directly to the ONNX graph as `float32`; neither the graph nor `OnnxSourcePolicy` applies a separate empirical-normalization transform. Observation preparation must preserve the exact ordering, units, reference trajectory, and environment configuration from `VegaSharpa-WholeBody-Manip-v0`. Although individual joint-state, wrist-pose, and object-pose terms retain uniform-noise configurations, corruption is disabled for the resolved policy observation group. It does not accept raw perception streams, natural language, or arbitrary robot-state schemas.

## Output(s)

**Output Type(s):** Tabular numerical robot joint-control action.

**Output Format(s):** Tensor. The ONNX graph output is named `actions` and has data type `float32`.

**Output Parameters:** The ONNX graph has fixed shape `(1, 58)`. The V2D wrapper concatenates per-row outputs into a two-dimensional `(num_envs, 58)` tensor containing dimensionless reference-centered residual commands for 14 arm joints and 44 finger joints in the robot asset's resolved joint order.

**Other Properties Related to Output:** The environment multiplies the raw policy output by 0.15 radians, applies an exponential moving average (`0.3 × previous + 0.7 × current`), clips the filtered residual to `[-1, 1]` radians, adds it to the current reference joint position, and clips the absolute target to the robot's joint limits. The target is sent to the articulation's position controller at the documented 0.05-second policy interval, or 20 Hz.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware, such as GPU cores, and software frameworks, such as CUDA libraries, the model achieves faster training and inference times compared with CPU-only solutions.

## Software Integration

**Runtime Engine(s):** ONNX Runtime through `groot_finetune.source_policy.OnnxSourcePolicy` in NVIDIA Isaac Lab and the V2D Robotic Grounding task environment. The repository image installs `onnxruntime-gpu==1.24.4`; the wrapper prefers `CUDAExecutionProvider` for a CUDA simulator device and uses `CPUExecutionProvider` when available as a fallback. The graph uses ONNX opset 11. PyTorch and RSL-RL apply to the source training-checkpoint lineage, not the packaged ONNX loader.

**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere and NVIDIA Lovelace. Other NVIDIA RTX GPUs are expected to be compatible when supported by the released Isaac Lab, ONNX Runtime, and CUDA versions but have not been validated for this
  complete simulation stack.

**Supported Operating System(s):** Linux with glibc 2.35 using the dependency and container configuration distributed with the associated repository revision.

The policy must be integrated with the matching robot asset, joint ordering, reconstructed tissue-box assets, retargeted motion, observation manager, and action post-processing. Changes to any part of that contract require revalidation.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

**v1.0.0:** Packaged deterministic actor `vega_sharpa_policy.onnx`, identified by SHA-256 `bf3259ec2b86dd696e6b89e6275d85985ad30dafeaf53351971b16cdb8dad8b2`. It exposes fixed `float32` tensors `obs` `(1, 445)` and `actions` `(1, 58)`, uses ONNX opset 11, and has no external initializer files. The ONNX metadata identifies PyTorch 2.7.1 as the producer.

## Training, Testing, and Evaluation Datasets:

This policy was trained with online synthetic simulation rollouts rather than a fixed supervised dataset. All rollouts were conditioned on one reconstructed and retargeted human tissue-box manipulation trajectory.

### Training Dataset

**Name:** Ego Reconstruction Retargeting Sample, Vega Sharpa trajectory `tissue_box_simple`.

**Data Modality:** Other - numerical robot state, robot action, reward, termination, and motion-trajectory data.

**Non-Audio, Image, Text Training Data Size:** One retargeted 58-joint trajectory containing 649 frames at 50 Hz, representing 12.98 seconds of source motion, plus approximately `1.97 × 10^9` synthetic simulated transition opportunities. The trajectory was replayed at half speed for the RL task. This is one human demonstration, not 649 independent demonstrations.

**Data Collection Method:** Hybrid: Manually-Collected, Synthetic.

**Data Labeling Method:** Hybrid: Automatic/Sensors, Manually-Labeled, Synthetic. Reconstruction and retargeting were automated and then inspected and corrected by people; simulator rewards and terminations were generated automatically.

**Collection and Processing:** An internally recorded NVIDIA egocentric video was processed through hand, object, support-surface, and contact reconstruction, then retargeted to the Vega/Dexmate Sharpa embodiment. Automated processing was supplemented by human inspection and corrections. The raw video is excluded from the released dataset.

**Properties:** The training lineage contains one manually collected human demonstration transformed into one 649-frame numerical robot motion trajectory. Online training generated numerical robot state, action, reward, and termination data in simulation. The source demonstration contains derived human hand shape and motion; the raw video is excluded from the released dataset and checkpoint. The data has no linguistic content.

**Online Rollouts:** PPO training used 4,096 parallel simulated environments, 24 simulation steps per environment per iteration, and 20,000 configured iterations, corresponding to approximately `1.97 × 10^9` simulated transition opportunities. Training used seed 42 on one NVIDIA L40 GPU. The policy learned from rewards and termination signals generated by the simulator; there are no conventional training, validation, and test percentages.

### Testing Dataset

**Data Collection Method:** Synthetic.

**Data Labeling Method:** Automatic/Sensors.

**Properties:** The available test evidence is the same set of 128 partial synthetic simulator episodes described under Evaluation Dataset. It contains numerical robot state, action, reference-motion, and termination data, with no personal or linguistic content. It is not a separate held-out dataset.

No separate held-out testing dataset was used. The available evaluation uses the same tissue-box asset, embodiment, reference motion, and training-distribution reset contract. Consequently, the reported metric does not measure generalization to unseen objects, motions, robot configurations, or real-world conditions.

### Evaluation Dataset

The preserved training-time evaluation ran the deterministic actor in 128 parallel simulated environments with seed 42. It was not independently repeated as an ONNX-export parity evaluation. Episodes were sampled across different reference-motion progress points under the training-distribution reset configuration. The mean observed episode length was 281.04 steps, with a range from 1 to 567 steps; these were not 128 complete manipulation cycles.

**Data Collection Method:** Synthetic.

**Data Labeling Method:** Automatic/Sensors. Isaac Lab generated episode states and termination outcomes from simulator state and configured termination rules.

**Properties:** The evaluation comprises 128 partial synthetic simulator episodes containing numerical robot state, action, reference-motion, and termination data. All episodes use the same tissue-box asset, embodiment, and reference trajectory as training. The data contains no personal or linguistic content.

| Metric | Result | Interpretation |
|---|---:|---|
| Sequence-survival rate | 117/128, or 91.41% | Reached the environment's reference-timeout termination without an earlier wrist, object-position, object-orientation, or robot-divergence termination |
| Early termination rate | 11/128, or 8.59% | Failed at least one configured trajectory-safety threshold before timeout |
| Full pickup, hold, return, and placement success | Not measured | The evaluation contract did not require completion of these task phases |

The 91.41% value must not be presented as a tissue-box pickup success rate or as evidence of physical-robot performance.

## Inference:

**Acceleration Engine:** ONNX Runtime through the repository's V2D and Isaac Lab integration; no TensorRT export is supplied.

**Test Hardware Set-Up:** NVIDIA RTX A6000 and NVIDIA L40 GPUs. Training and evaluation were validated on one NVIDIA L40 GPU; ONNX parity, a minimum inference GPU, and standalone policy latency were not independently measured.

Use the ONNX artifact only with the released configuration, preserve its observation/action contract, and reject copies whose SHA-256 does not match the published digest. The graph uses standard ONNX opset 11 and has no external initializer files.

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the [Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards](model-card-subcards/).

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
