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

If using VSCode or Cursor, you can also use the `Attach to Running Container` feature in Dev Containers extension by `command/ctrl + shift + p`.  Inside the container, you can use Python interpreter `/workspace/isaaclab/_isaac_sim/python.sh` for debugging.

Currently, due to Isaac Lab's image requiring root for Omniverse, we are using the root user for the container. This should be fixed in the future if there is a need. There can be some permission issues, but it can be bypassed with `sudo chown "$USER":domain-users <FILE_PATH>`.

## Retargeting
```bash
python scripts/retarget/arctic_to_sharpa.py # Check the file for arguments
python scripts/retarget/vis_retargeted.py # Need to run retargeting and save results first
```

For Arctic motion files, download them from [here](https://drive.google.com/file/d/1rL4T9N4AwQoWRqS5pOuB6a0D0B9Y8dsP/view?usp=sharing) and extract to `source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic`. (TODO: We need to figure out an appropriate way for data storage)

## RL training
```bash
python scripts/rsl_rl/dummy_agent.py
python scripts/rsl_rl/train.py
python scripts/rsl_rl/play.py
```

## RL Tasks
- `Sharpa-V2P-v0-Play`
- `Sharpa-V2P-v0`

## OSMO

[OSMO Wikipage](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/index.html)

### OSMO Remote Development

Launch a remote development environment on OSMO cluster:

```bash
# Build, push Docker image, and submit OSMO workflow
python scripts/run_osmo.py --experiment-name <your-name> --workflow-yaml workflow/dev_env.yaml
```

Once the workflow is running:
```bash
# Port-forwarding
osmo workflow port-forward <workflow-name> dev-env --port 6000:22

# SSH into the remote
ssh root@localhost -p 6000
```

You can now develop remotely inside the OSMO container with full GPU access.

### Launch training job
Launch a remote training job using the train workflow config:

```bash
python scripts/run_osmo.py --experiment-name <your-name> --workflow-yaml workflow/train.yaml
```
