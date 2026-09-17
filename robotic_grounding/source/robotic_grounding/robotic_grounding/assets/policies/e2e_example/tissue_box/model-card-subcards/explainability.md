# Explainability Subcard

Field | Response
:---|:---
Intended Task/Domain | Robotics; simulated reference-motion tracking for a Vega/Dexmate Sharpa tissue-box lift, hold, and place-back task.
Model Type | Deterministic ONNX actor exported from a Proximal Policy Optimization (PPO) feed-forward actor-critic multilayer perceptron.
Intended Users | Robotics developers and researchers using the matching V2D Robotic Grounding and Isaac Lab environment to replay the expert or generate expert rollout data.
Output | ONNX `actions`, a `float32` tensor with fixed shape `(1, 58)` containing reference-centered residual targets for the robot's 14 arm and 44 finger joints at 20 Hz.
Describe how the model works | The packaged actor maps the raw concatenated 445-dimensional policy observation to an action mean through three exponential linear unit (ELU) hidden layers. The environment scales the residual, applies exponential moving-average smoothing, clips the residual, adds it to the reference joint position, and clips the absolute target to the joint limits. The critic and stochastic action scale are training-only components rather than part of the packaged ONNX actor.
List the groups (or group characteristics) for which this was tested to produce comparable outcomes: | Not Applicable.
Technical Limitations & Mitigation | The policy depends on an exact 445-dimensional observation contract, 58-joint ordering, robot and object assets, and one reference trajectory. It was not tested on novel objects, motions, embodiments, broad domain randomization, sensor noise from physical hardware, or real-world contacts. Preserve the released integration, evaluate each changed condition, and use staged simulation and hardware testing before deployment.
Verified to have met prescribed NVIDIA quality standards | Yes
Performance Metrics | Training-time deterministic sequence survival was 117/128, or 91.41%, across partial training-distribution episodes. ONNX parity was not independently measured. This metric does not establish successful pickup, hold, return, or placement.
Potential Known Risks | Incorrect observations, asset mismatch, physics mismatch, checkpoint corruption, or distribution shift may cause unstable joint targets, collisions, pinching, excessive contact force, or a dropped object. The recorded sequence-survival score does not demonstrate reliable pickup or placement and is not evidence of physical safety.
Licensing | GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
