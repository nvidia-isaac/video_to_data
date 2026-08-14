# Model Overview

## Description:
SONIC (Supersizing Motion Tracking) is a humanoid behavior foundation model from NVIDIA that learns whole-body motor skills from large-scale human motion data. Using motion tracking as the training objective, a single unified reinforcement-learning policy produces natural, whole-body movement across diverse behaviors including walking, running, jumping, and manipulation.

This model card describes the SONIC checkpoints bundled in the v2d (Video-to-Data) robotic grounding pipeline. In v2d, SONIC is the **low-level whole-body tracking controller**: it consumes the dense whole-body reference produced by the v2d MotionBricks Planner (plus robot state) and outputs joint action commands for the Unitree G1 humanoid. The checkpoints are provided as two ONNX graphs — a policy **encoder** and a policy **decoder**.

These artifacts are derived from the public **[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)** model. The only modification is ONNX-graph surgery to make the encoder and decoder support a **dynamic batch dimension** (the upstream GEAR-SONIC ONNX files are fixed to batch size 1); the learned weights are unchanged.

This model is ready for commercial or non-commercial use.

### License/Terms of Use:
GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

### Deployment Geography:
Global

### Use Case:
This model is intended for robotics researchers and engineers deploying whole-body humanoid control. Use cases include reference-motion tracking, VR teleoperation, real-time robot control, whole-body manipulation, locomotion across diverse movement styles, synthetic data collection, and hardware deployment on humanoid robots. In the v2d pipeline, it executes the planner's whole-body reference on the Unitree G1.

### Release Date:
GitHub 07/08/2026 via [https://github.com/nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data)

## Model Architecture:
**Architecture Type:** Other (specify): Behavior Foundation Model (Reinforcement Learning) <br>
**Network Architecture:** Other (specify): Encoder–decoder policy. The encoder maps the observation into a finite-scalar-quantized (FSQ) latent token; the decoder maps the latent token plus an observation history into joint action commands. <br>

The checkpoints are split into two graphs:
* **Policy encoder** (`encoder_batched.onnx`) — observation → quantized latent tokens.
* **Policy decoder** (`decoder_batched.onnx`) — latent tokens + observation history → joint actions.

The encoder output is FSQ-quantized (token values are exact multiples of 1/16).

This model has 42M model parameters.

## Input:
**Input Type(s):**
* Other (specify): Robot proprioceptive state, motion-tracking reference, and control-command observations

**Input Format(s):**
* Tensor

**Input Parameters:**
* Two-Dimensional (2D)

**Other Properties Related to Input:**
* 32-bit float values.
* Encoder input (`obs_dict`): `(batch, 1762)` — flattened observation vector.
* Decoder input (`obs_dict`): `(batch, 994)` — flattened latent token plus observation history (history length 10).
* Both graphs expose a dynamic `batch` axis.
* The 1762-dim encoder observation is assembled by the v2d/robotic_grounding observation manager (the layout includes a padding block sized `1762 − 17 − 644 = 1101`).

## Output:
**Output Type(s):**
* Other (specify): Quantized latent tokens and whole-body robot joint action commands

**Output Format:**
* Tensor

**Output Parameters:**
* Two-Dimensional (2D)

**Other Properties Related to Output:**
* Encoder output (`encoded_tokens`): `(batch, 64)` — FSQ-quantized latent.
* Decoder output (`action`): `(batch, 29)` — 29 Unitree G1 body-joint action commands.
* Decoder actions drive the 29 G1 body joints; finger/hand joints are handled separately by the v2d pipeline.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):**
* Not Applicable (N/A) <br>
* TensorRT <br>

ONNX Runtime (`onnxruntime-gpu`) is used in the v2d pipeline; TensorRT is supported for low-latency deployment by upstream GEAR-SONIC.

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere
* NVIDIA Hopper
* NVIDIA Lovelace
* NVIDIA Jetson

**Supported Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
* GEAR-SONIC (part of the NVIDIA GR00T initiative) — encoder/decoder, dynamic-batch ONNX variant.

## Training and Evaluation Datasets:

### Training Dataset:

### Data Modality:
* Other: Motion-capture-derived skeletal pose trajectories <br>

### Training Data Size:
**Non-Audio, Image, Text Training Data Size:** Approximately 310,000 motion clips. <br>

**Data Collection Method** <br>
Automatic/Sensors <br>

**Labeling Method** <br>
Hybrid: Automatic/Sensors, Manually-Labeled <br>

**Properties:** Trained with reinforcement learning on the **BONES** motion-capture dataset using the Isaac Lab framework, with motion tracking as the training objective. BONES is an annotated human motion-capture dataset for humanoid robotics research, provided in a Unitree G1 MuJoCo-compatible format and spanning everyday activities, sports, communication, and dance. A subset is publicly released as **[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)** (Skeletal Everyday Embodiment Dataset), comprising 142,220 motions (71,132 original + 71,088 mirrored), ~288 hours at 120 fps from 522 performers. <br>

### Evaluation Dataset:

**Data Collection Method** <br>
Automatic/Sensors <br>

**Labeling Method** <br>
Hybrid: Automatic/Sensors, Manually-Labeled <br>

**Properties:** Evaluated on a held-out split of **[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)** (Skeletal Everyday Embodiment Dataset), measuring motion-tracking accuracy (upstream PyTorch checkpoint evaluation via `eval_agent_trl.py`). <br>

## Inference:
**Acceleration Engine:** <br>
* Other (specify): ONNX Runtime <br>
* TensorRT <br>
* PyTorch <br>
**Test Hardware:** <br>
* Desktop NVIDIA GPUs
* NVIDIA Jetson

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the [Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards](model-card-subcards/).

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
