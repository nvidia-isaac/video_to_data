# Robotic Grounding

## Prerequisites

- Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [post-installation](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) steps.

- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

- Make sure you have access to `nvcr.io/nvstaging/isaac-amr`. You can request it by asking in the `#swngc-help` Slack channel.

- Install Git LFS and `pre-commit` dependencies.
    ```bash
    bash workflow/setup_deps.sh
    ```
    This script installs `git-lfs` and `pre-commit` and ensures `workflow/run.sh` is executable. You may need to restart your shell for pipx PATH changes.

- NVIDIA Driver Version 580.126.09, CUDA Version: 13.0 Recommended. In case of visualization errors, check NVIDIA driver version.

## Docker Usage

Development should be **inside** the Container, and Git operations should be done **outside** the Container on the host machine.

```bash
./workflow/run.sh build [version] # Build Docker image and tag it with [version]
./workflow/run.sh push [version] # Push Docker image to NVIDIA registry
./workflow/run.sh pull [version] # Pull Docker image from NVIDIA registry
./workflow/run.sh start [version] [gpu] # Run and enter the Container with specific version and GPU
./workflow/run.sh shell [version] [gpu] # Enter Container from new shell with specific version and GPU
./workflow/run.sh stop [version] [gpu] # Stop the Container with specific version and GPU
```

## Development

You can launch the container with commands in the Docker Usage section.

If using VSCode or Cursor, you can use the `Attach to Running Container` feature in Dev Containers extension by `command/ctrl + shift + p`.  Inside the container, you can use Python interpreter `/workspace/isaaclab/_isaac_sim/python.sh` for debugging. The working directory is `/workspace/video_to_data/robotic_grounding`.

Currently, due to Isaac Lab's image requiring root for Omniverse, we are using the root user for the container. There can be some permission issues, but they can be bypassed with `sudo chown -R $(whoami) .` in the host machine.

### Running Debugging Env

The debug environment (`Sharpa-V2P-Debug-v0`) provides interactive GUI controls for testing contact sensors and MDP components:

- **Joint GUI Control**: Adjust all robot joints (wrist 6DoF + fingers) with P/D gain sliders
- **Object Pose GUI**: Move and rotate the object in 6DoF via floating base controls
- **Reward Visualizer**: Monitor reward terms in real-time with history plots

To run the debug environment:
```bash
python scripts/rsl_rl/dummy_agent.py --task Sharpa-V2P-Debug-v0
```

**Note**: Do not use `--headless` flag since the GUI controls require a display. The environment is configured with extended episode length (1 hour) and disabled randomization events for uninterrupted manual testing.

To create a debug environment for other tasks, use `sharpa_debug_env_cfg.py` as a template—inherit from your base environment config and override the actions with GUI-controlled versions (`JointGUIActionCfg`, `ObjectPoseGUIActionCfg`, `RewardVisualizerCfg`). Remember to disable action related MDP terms since the debug env does not have the original actions.

## Retargeting
```bash
python scripts/retarget/arctic_loader.py --save # Check the file for arguments
python scripts/retarget/arctic_to_sharpa.py --save # Check the file for arguments
python scripts/retarget/vis_retargeted.py # Need to run retargeting and save results first
```

For Arctic motion files, download them from [here](https://drive.google.com/file/d/1rL4T9N4AwQoWRqS5pOuB6a0D0B9Y8dsP/view?usp=sharing) and extract to `source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic`. (TODO: We need to figure out an appropriate way for data storage)

## RL training
```bash
python scripts/rsl_rl/dummy_agent.py  # Run environment with zero/random actions for testing setup
python scripts/rsl_rl/train.py        # Train an RL agent using RSL-RL
python scripts/rsl_rl/eval.py         # Evaluate a trained checkpoint and export policy to JIT/ONNX
python scripts/rsl_rl/play.py         # Play environment without a checkpoint (sinusoidal or zero actions)
```

## RL Tasks
- `Sharpa-V2P-v0-Play`
- `Sharpa-V2P-v0`
