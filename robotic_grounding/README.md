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

## Environment & Credentials

Retrieve your CSS/PDX keys from `~/.config/osmo/config.yaml` under `swift://pdx.s8k.io/AUTH_team-isaac`.

**Option A — Edit the credentials file directly (simple)**

Fill in your keys in `scripts/setup_css_env.sh`:

```bash
export CSS_ACCESS_KEY="<your-access-key-id>"
export CSS_SECRET_KEY="<your-secret-key>"
```

Then source it manually when needed:
```bash
source scripts/setup_css_env.sh
```

**Option B — direnv (automatic, recommended)**

[direnv](https://direnv.net/) auto-loads credentials whenever you `cd` into
`robotic_grounding/`.

Install:
```bash
sudo apt-get install direnv        # Ubuntu/Debian
# then add to ~/.bashrc:
eval "$(direnv hook bash)"
```

Create a **gitignored** `.envrc.local` inside this package
(`robotic_grounding/.envrc.local`):
```bash
# robotic_grounding/.envrc.local
export CSS_ACCESS_KEY="<your-access-key-id>"
export CSS_SECRET_KEY="<your-secret-key>"
```

Allow direnv to load it (one-time, per clone):
```bash
cd robotic_grounding && direnv allow
```

Credentials are then injected automatically on every `cd` into
`robotic_grounding/`.

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

For agent-oriented checks, use this quick path before opening a merge request. Commands assume the `robotic_grounding/` package root, and Isaac commands should run inside the container. OSMO and W&B are not required for the local smoke tests below.

## Agent Smoke Tests

### Assets and dummy agent

Motion data resolves under `source/robotic_grounding/robotic_grounding/assets/human_motion_data/`. The safest local shorthand is `<dataset>/<dataset>_processed/<sequence_id>/sharpa_wave`, for example `arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave`.

Pull retargeted outputs when they are missing:

```bash
osmo dataset info v2d_arctic_retarget_exp_200 --order desc
osmo dataset download v2d_arctic_retarget_exp_200:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic/ \
  --regex '(arctic_processed|arctic_urdfs|reconstructed_stage)/.*'
```

If OSMO is not configured, do not attempt the dataset download. Use any already
present local motion partition, or ask for the dataset version/path needed for
the task. OSMO setup lives in [workflow/README.md](workflow/README.md).

Run a GUI dummy-agent smoke test inside the container:

```bash
python scripts/rsl_rl/dummy_agent.py \
  --task Sharpa-V2P-v0-Play \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --use_primitive_urdfs
```

Run the same check headless with a short MP4:

```bash
python scripts/rsl_rl/dummy_agent.py \
  --headless \
  --task Sharpa-V2P-v0-Play \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --use_primitive_urdfs \
  --record_video \
  --output_dir /tmp/rg_dummy_agent_video \
  --video_length 300
```

Success means Isaac starts, the task registers, `SceneConfig.from_motion_file` loads the parquet partition, no missing-asset exception is raised, and the simulation advances.

### Training and OSMO

Use the `RL training` section below for a local `train.py` dry-run and one-iteration smoke test. If W&B is not configured, keep local smoke tests on TensorBoard by passing `--logger tensorboard`. Use [experiments/README.md](experiments/README.md) for OSMO dry-runs, image selection, and launch commands. The key merge-readiness checks are:

```bash
python experiments/run_experiment.py example_fixed_post --osmo --dry-run
python experiments/run_experiment.py example_AC_post --osmo --dry-run
python experiments/run_experiment.py example_pre_fixed_post --osmo --dry-run
```

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

### Hand-only (Sharpa / Arctic)
```bash
python scripts/retarget/arctic_loader.py --save # Check the file for arguments
python scripts/retarget/arctic_to_sharpa.py --save # Check the file for arguments
python scripts/retarget/vis_retargeted.py # Need to run retargeting and save results first
```

For Arctic motion files, download them from [here](https://drive.google.com/file/d/1rL4T9N4AwQoWRqS5pOuB6a0D0B9Y8dsP/view?usp=sharing) and extract to `source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic`. (TODO: We need to figure out an appropriate way for data storage)

### Whole-body (SOMA → G1)
```bash
# Retarget and save Parquet (data_folder must contain soma_params.npz, object/textured_mesh.obj)
python scripts/retarget/soma_to_g1.py <data_folder> --save

# Visualize retargeting in Viser (port 8080)
python scripts/retarget/soma_to_g1.py <data_folder> --visualize
```

### Kinematic replay (all schemas)
Replay retargeted motion in Isaac Lab. Supports both whole-body (G1) and dual floating-hand (Sharpa/Dex3) data.
Robot and object are teleported kinematically — no physics forces act on them.
```bash
# Replay G1 retargeted data (loops by default)
python scripts/replay_motion.py \
    --motion_file source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=<seq>/robot_name=g1

# Replay hand-only data
python scripts/replay_motion.py \
    --motion_file source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic_processed/sequence_id=<seq>/robot_name=sharpa_wave

# Options
python scripts/replay_motion.py --motion_file <path> --speed 0.5   # Slow motion
python scripts/replay_motion.py --motion_file <path> --no-loop     # Stop at last frame
python scripts/replay_motion.py --motion_file <path> --headless    # No GUI
```

### Support surface reconstruction
Detect where objects rest on surfaces above the ground plane and generate collision geometry for RL training.
```bash
# For hand-only datasets (auto-detects schema)
python scripts/reconstruct_support_surfaces.py --input_dir <loader_output_dir> --sequence_id <seq>

# For G1 whole-body retargeted data
python scripts/reconstruct_support_surfaces.py --input_dir source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma --sequence_id <seq>

# Or use the dataset shortcut
python scripts/reconstruct_support_surfaces.py --dataset soma_g1 --sequence_id <seq>
```

Objects resting on the ground are automatically filtered out (threshold configurable via `--ground_threshold`).

### Scene viewer (static spawn verification)
```bash
python scripts/view_scene.py --motion_file <parquet_partition_path>
```

## RL training

Commands in this section assume you are inside the container from the
`robotic_grounding/` package root. These commands do not require W&B or OSMO
when the motion data is already present locally.

```bash
# Verify the experiment runner can generate a train.py command without starting Isaac.
python experiments/run_experiment.py example_fixed_post --local --dry-run

# Run a real one-iteration train smoke test.
python scripts/rsl_rl/train.py \
  --headless \
  --task Sharpa-V2P-v0 \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --max_iterations 1 \
  --logger tensorboard \
  --run_name smoke_train \
  --use_primitive_urdfs \
  agent.num_steps_per_env=8 \
  agent.save_interval=1

# Evaluate the checkpoint produced by the smoke train.
CHECKPOINT=$(find logs/rsl_rl -path '*smoke_train*/model_*.pt' | sort -V | tail -1)
python scripts/rsl_rl/eval.py \
  --headless \
  --task Sharpa-V2P-v0 \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --eval_episodes 1 \
  --use_primitive_urdfs

# Other entry points.
python scripts/rsl_rl/dummy_agent.py  # Run an environment with zero actions.
python scripts/rsl_rl/eval.py         # Evaluate a trained checkpoint and export policy.
python scripts/rsl_rl/play.py         # Play without a checkpoint.
```

See the `Agent Smoke Tests` section above for the required asset layout, dummy-agent commands, and OSMO experiment dry-runs.

## RL Tasks
- `Sharpa-V2P-v0-Play`
- `Sharpa-V2P-v0`

## Visualizer

Browse retargeted sequences as 3D animations at **http://10.111.83.14:8080/**

To run the server yourself or generate new recordings:

```bash
# Download datasets from OSMO
python visualizer/sync_visualizer_data.py

# Start the gallery server
python visualizer/serve.py          # → http://0.0.0.0:8080

# Serve vis_retargeted.py output directly (no copy needed)
python visualizer/serve.py --html-dir /path/to/v2d_arctic_retarget_exp_200
```

See [visualizer/README.md](visualizer/README.md) for the full reference: parallel downloads, generating `.viser` files inside Docker, and running as a systemd service.
